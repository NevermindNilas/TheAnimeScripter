import logging
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch

import src.io.ffmpegSettings as ffmpegSettings
from src.constants import ADOBE
from src.infra.progressBarLogic import ProgressBarLogic
from src.io.ffmpegSettings import (
    BuildBuffer,
    closeWriterAndDrainReader,
    createWriteBuffer,
    drainReader,
)
from src.io.runOutcome import truncatedDecodeError
from src.model.download import resolveWeightPath
from src.model.registry import modelsMap
from src.stabilize.dut import config as dutCfg
from src.stabilize.dut.mesh_warp import MeshWarper
from src.stabilize.dut.motion_pro import MotionPro
from src.stabilize.dut.pwcnet import Network as PWCNet
from src.stabilize.dut.pwcnet import estimate as pwcEstimate
from src.stabilize.dut.rf_det import RFDetSO
from src.stabilize.dut.smoother import Smoother

if ADOBE:
    from src.server.aeComms import progressState, reportTerminalStatus

# Halo-chunked smoothing bounds GPU memory on long videos. See _smoothChunked
# for why normalization bounds are global; the discarded halo covers the
# Jacobi kernel's effective influence radius with a wide margin.
SMOOTH_CORE = 4096
SMOOTH_HALO = 256

# Max propagation jobs in flight behind the analysis loop. Each holds a
# ~2.4MB flow tensor on the GPU; one is enough to overlap the CPU section,
# a few more ride out per-pair time variance.
PROPAGATE_IN_FLIGHT = 4


class VideoStabilizeDUT:
    """DUT (deep unsupervised) stabilization: learned keypoints (RFDet) +
    PWC-Net flow + learned motion propagation onto a 40x30 mesh + learned
    trajectory smoothing, rendered as a per-cell homography warp at native
    resolution. Same driver contract as VideoStabilize."""

    def __init__(
        self,
        input,
        output,
        width,
        height,
        fps,
        inpoint=0,
        outpoint=0,
        encode_method="x264",
        custom_encoder="",
        benchmark=False,
        totalFrames=0,
        bitDepth: str = "8bit",
    ):
        self.input = input
        self.output = output
        self.width = width
        self.height = height
        self.fps = fps
        self.inpoint = inpoint
        self.outpoint = outpoint
        self.encode_method = encode_method
        self.custom_encoder = custom_encoder
        self.benchmark = benchmark
        self.totalFrames = totalFrames
        self.bitDepth = bitDepth

        # Grid displacement (smoothed - original) per frame, model-res px.
        self.gridDisplacement = None

        # Same contract as VideoStabilize: --stabilize bypasses start(), so
        # this is the only place a pipeline failure can be observed.
        self.processingError: Exception | None = None
        try:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "--stabilize_method dut requires a CUDA-capable GPU; "
                    "use --stabilize_method classic instead"
                )
            self.device = torch.device("cuda")
            self._loadModels()
            self.runStreamingPipeline()
        except Exception as e:
            self.processingError = e
            logging.exception(f"Something went wrong in DUT stabilization, {e}")

        if ADOBE:
            reportTerminalStatus(self.processingError, self.output, self.benchmark)

    def _loadModels(self):
        self.rfDet = RFDetSO()
        self.rfDet.loadWeights(resolveWeightPath("dut_rfdet", modelsMap("dut_rfdet")))
        self.pwcNet = PWCNet()
        self.pwcNet.loadWeights(
            resolveWeightPath("dut_pwcnet", modelsMap("dut_pwcnet"))
        )
        self.motionPro = MotionPro()
        self.motionPro.loadWeights(
            resolveWeightPath("dut_motionpro", modelsMap("dut_motionpro"))
        )
        self.smoother = Smoother()
        self.smoother.loadWeights(
            resolveWeightPath("dut_smoother", modelsMap("dut_smoother"))
        )
        for model in (self.rfDet, self.pwcNet, self.motionPro, self.smoother):
            model.to(self.device).eval()

    def runStreamingPipeline(self):
        if ADOBE:
            progressState.update({"status": "Analyzing and stabilizing video..."})

        with ProgressBarLogic(self.totalFrames * 2, title=self.input) as bar:
            self.analyzeMotion(progressBar=bar, advance=1)
            with torch.no_grad():
                self.computeTrajectoryCorrection()
            self.renderStabilized(progressBar=bar, advance=1)

    def _clearReaderCache(self):
        try:
            ffmpegSettings.CachedReader = None
            ffmpegSettings.CachedReaderMethod = None
        except Exception as e:
            logging.warning(f"Failed to clear cached decoder reader: {e}")

    def _newReadBuffer(self):
        return BuildBuffer(
            videoInput=self.input,
            inpoint=self.inpoint,
            outpoint=self.outpoint,
            resize=False,
            width=self.width,
            height=self.height,
            toTorch=False,
        )

    def analyzeMotion(self, progressBar=None, advance=1):
        if ADOBE:
            progressState.update(
                {"status": "Analyzing camera motion for stabilization..."}
            )

        self._clearReaderCache()
        self.readBuffer = self._newReadBuffer()

        with ThreadPoolExecutor(max_workers=2) as executor:
            decodeFuture = executor.submit(self.readBuffer)
            analyzeFuture = executor.submit(self._analyzeFrames, progressBar, advance)
            # Await the consumer first, then drain (see VideoStabilize).
            try:
                analyzeFuture.result()
            finally:
                drainReader(self.readBuffer)
                decodeFuture.result()

    def _toModelTensors(self, frame):
        """RGB HWC uint8 native-res frame -> (gray [1,1,h,w], bgr [1,3,h,w])."""
        small = cv2.resize(
            frame, (dutCfg.WIDTH, dutCfg.HEIGHT), interpolation=cv2.INTER_LINEAR
        )
        gray = (
            torch.from_numpy(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY))
            .to(self.device, dtype=torch.float32)
            .div_(255.0)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        # PWC-Net's published weights were trained on BGR input.
        bgr = (
            torch.from_numpy(np.ascontiguousarray(small[:, :, ::-1]))
            .to(self.device, dtype=torch.float32)
            .div_(255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        return gray, bgr

    @torch.no_grad()
    def _analyzeFrames(self, progressBar=None, advance=1):
        # The decorator, not a caller-side context manager: grad mode is
        # thread-local and this runs on a ThreadPoolExecutor worker, where an
        # outer no_grad() would not apply (autograd history chained across
        # every PWC-Net call and OOM'd a 24GB card in 480 frames).
        frameCount = 0
        prevBgr = None
        prevKpts = None
        gridMotions = []

        @torch.no_grad()  # runs on the propagator worker thread (see above)
        def propagate(flow, kpts):
            # Flow is only read at the previous frame's keypoint pixels, so
            # upstream's topk-mask multiply is a no-op and skipped here.
            return self.motionPro.inference(flow[:, 0:1], flow[:, 1:2], kpts[0]).cpu()

        # Motion propagation is half CPU work (k-means, RANSAC, tensor
        # roundtrips); running it one pair behind on its own worker lets that
        # CPU time overlap the next pair's detect/flow. The wall clock of this
        # loop is CPU-bound, not GPU-bound, so this is where the time comes
        # from. The drain loop below bounds in-flight flow tensors (~2.4MB GPU
        # each); NOT deque(maxlen=...), which would silently drop futures.
        analysisStart = time.perf_counter()
        pending = deque()
        with ThreadPoolExecutor(max_workers=1) as propagator:
            for _ in range(self.totalFrames):
                frame = self.readBuffer.read()
                if frame is None:
                    break

                gray, bgr = self._toModelTensors(frame)
                _, kpts = self.rfDet.detect(gray)

                if prevBgr is not None:
                    flow = pwcEstimate(prevBgr, bgr, self.pwcNet)
                    pending.append(propagator.submit(propagate, flow, prevKpts))
                    while len(pending) > PROPAGATE_IN_FLIGHT:
                        gridMotions.append(pending.popleft().result())

                prevBgr = bgr
                prevKpts = kpts
                frameCount += 1
                if progressBar is not None:
                    progressBar(advance)

                if self.readBuffer.isReadFinished() and self.readBuffer.isQueueEmpty():
                    break

            while pending:
                gridMotions.append(pending.popleft().result())

        self.gridMotions = gridMotions
        self.analyzedFrames = frameCount
        elapsed = time.perf_counter() - analysisStart
        logging.info(
            f"DUT analyzed {frameCount} frames for stabilization in "
            f"{elapsed:.1f}s ({frameCount / max(elapsed, 1e-6):.2f} fps)"
        )

    def _smoothChunked(self, path):
        T = path.shape[2]
        if T <= SMOOTH_CORE + 2 * SMOOTH_HALO:
            return self.smoother.smoothPath(path, repeat=50)
        # Normalization bounds must be the whole video's, not each chunk's:
        # the kernel-prediction net sees the normalized path, so chunk-local
        # bounds would change its response across the entire chunk, not just
        # at the seams. With global bounds, chunks differ from a full-sequence
        # run only by the Jacobi window truncated at the halo edge, and the
        # kernel's influence decays well inside the 256-frame discarded halo.
        minV, maxV = path.amin(), path.amax()
        out = torch.empty_like(path)
        start = 0
        while start < T:
            end = min(start + SMOOTH_CORE, T)
            s = max(0, start - SMOOTH_HALO)
            e = min(T, end + SMOOTH_HALO)
            smoothed = self.smoother.smoothPath(
                path[:, :, s:e], repeat=50, minV=minV, maxV=maxV
            )
            out[:, :, start:end] = smoothed[:, :, start - s : start - s + end - start]
            start = end
        return out

    def computeTrajectoryCorrection(self):
        if not self.gridMotions:
            self.gridDisplacement = None
            return

        motions = torch.stack(self.gridMotions, 2).to(self.device)
        self.gridMotions = None
        # Prepend the identity motion of frame 0, then accumulate to a path.
        motions = torch.cat([torch.zeros_like(motions[:, :, 0:1]), motions], 2)
        path = torch.cumsum(motions, 2)  # [1, 2, T, G_H, G_W]

        smoothed = self._smoothChunked(path)
        # The de/renormalization affine in smoothPath cancels in this
        # difference, so displacement is in model-resolution pixels.
        self.gridDisplacement = (smoothed - path).cpu()

        logging.info(f"DUT computed trajectory correction for {path.shape[2]} frames")

    def renderStabilized(self, progressBar=None, advance=1):
        if ADOBE:
            progressState.update(
                {"status": "Applying stabilization and encoding video..."}
            )

        self._clearReaderCache()
        self.readBuffer = self._newReadBuffer()

        self.writeBuffer = createWriteBuffer(
            input=self.input,
            output=self.output,
            encode_method=self.encode_method,
            custom_encoder=self.custom_encoder,
            width=self.width,
            height=self.height,
            fps=self.fps,
            grayscale=False,
            benchmark=self.benchmark,
            bitDepth=self.bitDepth,
            inpoint=self.inpoint,
            outpoint=self.outpoint,
        )
        self.writeHwcUint8 = getattr(self.writeBuffer, "acceptsHwcUint8", False)

        with ThreadPoolExecutor(max_workers=3) as executor:
            writeFuture = executor.submit(self.writeBuffer)
            decodeFuture = executor.submit(self.readBuffer)
            renderFuture = executor.submit(self._renderFrames, progressBar, advance)
            decodeFuture.result()
            renderFuture.result()
            writeFuture.result()

    @torch.no_grad()
    def _renderFrames(self, progressBar=None, advance=1):
        frameCount = 0
        try:
            warper = MeshWarper(self.width, self.height, self.device)
            # One 4.6MB-per-1000-frames upload instead of two 4.8KB copies per
            # frame inside the render loop.
            disp = (
                None
                if self.gridDisplacement is None
                else self.gridDisplacement.to(self.device)
            )
            numCorrected = 0 if disp is None else disp.shape[2]

            for i in range(self.totalFrames):
                frame = self.readBuffer.read()
                if frame is None:
                    break

                if i < numCorrected:
                    frameTensor = (
                        torch.from_numpy(frame)
                        .to(self.device, dtype=torch.float32)
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                    )
                    warped = warper.warp(frameTensor, disp[0, 0, i], disp[0, 1, i])
                    stabilized = (
                        warped.squeeze(0)
                        .permute(1, 2, 0)
                        .clamp_(0.0, 255.0)
                        .to(torch.uint8)
                        .cpu()
                        .numpy()
                    )
                else:
                    stabilized = frame

                if self.writeHwcUint8:
                    outputTensor = torch.from_numpy(np.ascontiguousarray(stabilized))
                else:
                    outputTensor = (
                        torch.from_numpy(stabilized)
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                        .to(torch.float32)
                    )
                    outputTensor.mul_(1.0 / 255.0)

                self.writeBuffer.write(outputTensor)
                frameCount += 1
                if progressBar is not None:
                    progressBar(advance)

                if self.readBuffer.isReadFinished() and self.readBuffer.isQueueEmpty():
                    break

            logging.info(f"DUT processed {frameCount} frames")
        finally:
            # Same truncated-decode + drain contract as VideoStabilize.
            decodeError = truncatedDecodeError(
                self.readBuffer, self.totalFrames, self.writeBuffer
            )
            if decodeError is not None and self.processingError is None:
                self.processingError = decodeError
            closeWriterAndDrainReader(self.writeBuffer, self.readBuffer)
