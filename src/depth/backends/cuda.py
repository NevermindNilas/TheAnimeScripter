import os
import sys

os.environ.setdefault("DA3_LOG_LEVEL", "ERROR")

import importlib
import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F

from src.constants import ADOBE
from src.depth.backends._shared import (
    MEANTENSOR,
    MEANTENSOR_HALF,
    STDTENSOR,
    STDTENSOR_HALF,
    DepthRunOutcome,
    OrderedFrameSink,
    aspectKeepingInputSize,
    calculateAspectRatio,
    disparityStretch,
    limboDisparity,
    limboResolution,
    stackFrames,
)
from src.infra.cudaStreams import waitForCaller
from src.infra.isCudaInit import CudaChecker
from src.infra.logAndPrint import logAndPrint
from src.infra.progressBarLogic import ProgressBarLogic
from src.io.ffmpegSettings import (
    BuildBuffer,
    createWriteBuffer,
)
from src.model.download import resolveWeightPath
from src.model.registry import modelsMap

if ADOBE:
    from src.server.aeComms import progressState

checker = CudaChecker()


def _resizeDepth(depth, height, width):
    """One raw DA3 depth map [h, w] at the frame's own (height, width), fp32.

    The decoder already scales to the model's size, so this only resizes after
    its OpenCV fallback (which does not scale): bilinear with half-pixel
    centres, as the cv2 INTER_LINEAR it replaces.
    """
    depth = depth.float()
    if tuple(depth.shape[-2:]) != (height, width):
        depth = F.interpolate(
            depth[None, None], (height, width), mode="bilinear", align_corners=False
        )[0, 0]
    return depth


def _disparityGray(depth):
    """DA3 depth [..., H, W] -> [0, 1] disparity with a 2/98-percentile stretch.

    The ``*_v3`` semantics: 10 valid pixels or fewer is a black map, and a map
    whose bounds coincide is widened by 1e-6 either side.
    """
    return disparityStretch(depth, minValid=10, widenFlat=True, depthFloor=None)


def _grayMaps(depths):
    """``_disparityGray`` of each map, batched when they share a size."""
    if len({tuple(d.shape) for d in depths}) == 1:
        return list(_disparityGray(torch.stack(depths)))
    return [_disparityGray(d) for d in depths]


class DepthCuda(DepthRunOutcome):
    def __init__(
        self,
        input,
        output,
        width,
        height,
        fps,
        half,
        inpoint=0,
        outpoint=0,
        encode_method="x264",
        depth_method="small",
        benchmark=False,
        totalFrames=0,
        bitDepth: str = "16bit",
        depthQuality: str = "high",
        compileMode: str = "default",
        depth_batch: int = 1,
    ):
        self.input = input
        self.output = output
        self.width = width
        self.height = height
        self.fps = fps
        self.half = half
        self.inpoint = inpoint
        self.outpoint = outpoint
        self.encode_method = encode_method
        self.depth_method = depth_method
        self.benchmark = benchmark
        self.totalFrames = totalFrames
        self.bitDepth = bitDepth
        self.depthQuality = depthQuality
        self.compileMode = compileMode
        self.depthBatch = max(1, int(depth_batch))

        self.handleModels()

        try:
            self.readBuffer = BuildBuffer(
                videoInput=self.input,
                inpoint=self.inpoint,
                outpoint=self.outpoint,
                half=self.half,
                resize=True,
                width=self.newWidth,
                height=self.newHeight,
            )

            self.writeBuffer = createWriteBuffer(
                self.encode_method,
                input=self.input,
                output=self.output,
                width=self.width,
                height=self.height,
                fps=self.fps,
                grayscale=True,
                benchmark=self.benchmark,
                bitDepth=self.bitDepth,
            )

            with ThreadPoolExecutor(max_workers=3) as executor:
                executor.submit(self.writeBuffer)
                executor.submit(self.readBuffer)
                executor.submit(self.guardedProcess)

        except Exception as e:
            self.recordFailure(e)
            logging.exception(f"Something went wrong, {e}")

        self.reportOutcome()

    def handleModels(self):
        if ADOBE:
            progressState.update(
                {"status": f"Loading depth model: {self.depth_method}..."}
            )

        from ..dpt_v2 import DepthAnythingV2

        match self.depth_method:
            case "small_v2":
                method = "vits"
            case _:
                raise ValueError(f"Unsupported CUDA depth method: {self.depth_method}")

        self.filename = modelsMap(
            model=self.depth_method, modelType="pth", half=self.half
        )
        modelPath = resolveWeightPath(
            self.depth_method,
            self.filename,
            half=self.half,
            modelType="pth",
        )

        modelConfigs = {
            "vits": {
                "encoder": "vits",
                "features": 64,
                "out_channels": [48, 96, 192, 384],
            },
        }

        self.model = DepthAnythingV2(**modelConfigs[method])

        stateDict = torch.load(modelPath, map_location="cpu")
        self.model.load_state_dict(stateDict)
        del stateDict

        if self.half and checker.cudaAvailable:
            self.model = self.model.half()
        self.model = self.model.to(checker.device).eval()
        torch.cuda.empty_cache()

        self.newHeight, self.newWidth = aspectKeepingInputSize(
            self.width, self.height, self.depthQuality
        )

        if self.compileMode != "default":
            try:
                if self.compileMode == "max":
                    self.model.compile(mode="max-autotune-no-cudagraphs")
                elif self.compileMode == "max-graphs":
                    self.model.compile(
                        mode="max-autotune-no-cudagraphs", fullgraph=True
                    )
            except Exception as e:
                logging.error(
                    f"Error compiling model {self.depth_method} with mode {self.compileMode}: {e}"
                )
                logAndPrint(
                    f"Error compiling model {self.depth_method} with mode {self.compileMode}: {e}",
                    "red",
                )

            self.compileMode = "default"

        self.normStream = torch.cuda.Stream()
        self.outputNormStream = torch.cuda.Stream()
        self.stream = torch.cuda.Stream()

    @torch.inference_mode()
    def normFrame(self, frame):
        if self.half and checker.cudaAvailable:
            if frame.dtype != torch.float16:
                frame = frame.half()
            return (frame - MEANTENSOR_HALF) / STDTENSOR_HALF
        return (frame - MEANTENSOR) / STDTENSOR

    @torch.inference_mode()
    def _normalizeDepth(self, depth):
        """Normalize a single already-upscaled depth frame [1, C, H, W].
        Per-frame min-max so a batched forward
        stays bit-identical to one-frame-at-a-time.

        The result goes straight to the writer, whose writer thread copies on
        its own private stream and never waits on ours, so the normalize must be
        synchronized here. Without it the writer can read the block before the
        kernels land and, because the allocator recycles the tensor freed two
        frames earlier, emit a byte-exact duplicate of frame N-2."""
        waitForCaller(self.outputNormStream)
        with torch.cuda.stream(self.outputNormStream):
            # clamp_min guards the divide only when the map is perfectly
            # flat; any real frame has a range far above it and comes out
            # bit-identical
            minVal = depth.min()
            depth = (depth - minVal) / (depth.max() - minVal).clamp_min(1e-6)
        self.outputNormStream.synchronize()
        return depth

    @torch.inference_mode()
    def processBatch(self, frames):
        try:
            waitForCaller(self.stream)
            with torch.cuda.stream(self.stream):
                # Produce the batch on the stream that reads it.
                batch = frames[0] if len(frames) == 1 else torch.cat(frames, dim=0)
                batch = self.normFrame(batch)
                depth = self.model(batch)
                depth = F.interpolate(
                    depth,
                    (self.height, self.width),
                    mode="bilinear",
                    align_corners=True,
                )
            self.stream.synchronize()
            # Normalize and write each frame in decode order.
            for i in range(depth.shape[0]):
                self.writeBuffer.write(self._normalizeDepth(depth[i : i + 1]))
        except Exception as e:
            self.recordFailure(e)
            logging.exception(f"Something went wrong while processing the frame, {e}")

    def process(self):
        frameCount = 0
        batchSize = self.depthBatch
        with ProgressBarLogic(self.totalFrames, title="Depth") as bar:
            while True:
                frames = []
                for _ in range(batchSize):
                    frame = self.readBuffer.read()
                    if frame is None:
                        break
                    frames.append(frame)
                if frames:
                    self.processBatch(frames)
                    frameCount += len(frames)
                    bar(len(frames))
                # A short read means the decoder's single None sentinel was just
                # consumed; another read() would block forever.
                if len(frames) < batchSize:
                    break

        logging.info(f"Processed {frameCount} frames")

        self.writeBuffer.close()


class LimboCuda(DepthCuda):
    """Limbo: an anime finetune of Depth Anything 3 small.

    Rides DepthCuda's tensor path (decode straight at the model resolution, one
    batched forward, write) rather than the numpy/PIL one the other ``*_v3``
    methods use, because limboResolution fixes the input size from the source
    and ``--depth_quality`` before the first frame: there is nothing for
    ``input_processor`` to negotiate, so the whole PIL round trip would be a
    no-op resize.
    """

    def handleModels(self):
        if ADOBE:
            progressState.update(
                {"status": f"Loading depth model: {self.depth_method}..."}
            )

        from .. import depth_anything_3 as depth_anything_3_pkg

        sys.modules.setdefault("depth_anything_3", depth_anything_3_pkg)
        MonocularDepthAnything3 = importlib.import_module(
            "depth_anything_3.mono"
        ).MonocularDepthAnything3

        base = self.depth_method.removeprefix("video_").split("-")[0]
        self.filename = modelsMap(model=base, modelType="pth", half=self.half)
        modelPath = resolveWeightPath(
            base,
            self.filename,
            half=self.half,
            modelType="pth",
        )

        self.model = MonocularDepthAnything3.from_pretrained(
            modelPath,
            model_name="da3-small",
            strict=False,
        ).to(checker.device)

        from ..fold_layerscale import fold_layerscale_
        from ..leanHead import installLeanHead

        fold_layerscale_(self.model)
        installLeanHead(self.model)

        # Left in fp32: DA3's own forward autocasts to bf16/fp16 on CUDA, which
        # is what the shipped *_v3 backends rely on too.
        self.newHeight, self.newWidth = limboResolution(
            self.width, self.height, self.depthQuality
        )

        if self.compileMode != "default":
            try:
                if self.compileMode == "max":
                    self.model.compile(mode="max-autotune-no-cudagraphs")
                elif self.compileMode == "max-graphs":
                    self.model.compile(
                        mode="max-autotune-no-cudagraphs", fullgraph=True
                    )
            except Exception as e:
                logging.error(
                    f"Error compiling model {self.depth_method} with mode {self.compileMode}: {e}"
                )
                logAndPrint(
                    f"Error compiling model {self.depth_method} with mode {self.compileMode}: {e}",
                    "red",
                )

            self.compileMode = "default"

        self.normStream = torch.cuda.Stream()
        self.outputNormStream = torch.cuda.Stream()
        self.stream = torch.cuda.Stream()

    @torch.inference_mode()
    def normFrame(self, frame):
        # fp32 regardless of --half: the model's own autocast picks the compute
        # dtype, and ImageNet normalization in fp16 costs range for nothing.
        return (frame.float() - MEANTENSOR) / STDTENSOR

    @torch.inference_mode()
    def processBatch(self, frames):
        try:
            waitForCaller(self.stream)
            with torch.cuda.stream(self.stream):
                # Produce the batch on the stream that reads it.
                batch = frames[0] if len(frames) == 1 else torch.cat(frames, dim=0)
                batch = self.normFrame(batch)
                # DA3 takes [B, views, C, H, W]; one frame is one single-view
                # scene, so the view axis is 1 and the batch axis stays
                # per-frame independent.
                depth = self.model.forward(batch.unsqueeze(1))["depth"][:, 0]
                depth = depth.unsqueeze(1)  # [B, 1, H, W]
            self.stream.synchronize()
            for i in range(depth.shape[0]):
                self.writeBuffer.write(self._toGray(depth[i : i + 1]))
        except Exception as e:
            self.recordFailure(e)
            logging.exception(f"Something went wrong while processing the frame, {e}")

    @torch.inference_mode()
    def _toGray(self, depth):
        """Raw depth at model resolution -> a [0, 1] map at output resolution.

        Normalizing before the upscale, not after, keeps the percentile pair off
        a multi-megapixel tensor (``nanquantile`` caps out around 16M elements)
        and hands the writer a frame that already matches the output, so its
        bicubic fallback -- which overshoots on a map that is already clipped to
        [0, 1] -- never runs.

        Synchronized here for the same reason DepthCuda._normalizeDepth is: the
        writer thread copies on its own private stream and never waits on ours.
        """
        waitForCaller(self.outputNormStream)
        with torch.cuda.stream(self.outputNormStream):
            gray = limboDisparity(depth)
            if gray.shape[-2:] != (self.height, self.width):
                gray = F.interpolate(
                    gray,
                    (self.height, self.width),
                    mode="bilinear",
                    align_corners=True,
                ).clamp(0.0, 1.0)
        self.outputNormStream.synchronize()
        return gray


class DepthV3Cuda(DepthRunOutcome):
    """Depth Anything 3 on CUDA, one frame per scene; base of the streaming classes."""

    def __init__(
        self,
        input,
        output,
        width,
        height,
        fps,
        half,
        inpoint=0,
        outpoint=0,
        encode_method="x264",
        depth_method="small_v3",
        benchmark=False,
        totalFrames=0,
        bitDepth: str = "16bit",
        depthQuality: str = "high",
        compileMode: str = "default",
        depth_batch: int = 1,
    ):
        self.input = input
        self.output = output
        self.width = width
        self.height = height
        self.fps = fps
        self.half = half
        self.inpoint = inpoint
        self.outpoint = outpoint
        self.encode_method = encode_method
        self.depth_method = depth_method
        self.benchmark = benchmark
        self.totalFrames = totalFrames
        self.bitDepth = bitDepth
        self.depthQuality = depthQuality
        self.compileMode = compileMode
        self.depthBatch = max(1, int(depth_batch))

        self.handleModels()

        decodeW = getattr(self, "_decodeWidth", self.width)
        decodeH = getattr(self, "_decodeHeight", self.height)
        decodeResize = getattr(self, "_decodeResize", False)
        try:
            self.readBuffer = BuildBuffer(
                videoInput=self.input,
                inpoint=self.inpoint,
                outpoint=self.outpoint,
                width=decodeW,
                height=decodeH,
                half=self.half,
                resize=decodeResize,
                toTorch=False,
                uint8Frames=True,
            )

            self.writeBuffer = createWriteBuffer(
                self.encode_method,
                input=self.input,
                output=self.output,
                width=self.width,
                height=self.height,
                fps=self.fps,
                grayscale=True,
                benchmark=self.benchmark,
                bitDepth=self.bitDepth,
            )
            with ThreadPoolExecutor(max_workers=3) as executor:
                executor.submit(self.readBuffer)
                executor.submit(self.writeBuffer)
                executor.submit(self.guardedProcess)

        except Exception as e:
            self.recordFailure(e)
            logging.exception(f"Something went wrong, {e}")

        self.reportOutcome()

    def process(self):
        frameCount = 0
        batchSize = self.depthBatch
        with ProgressBarLogic(self.totalFrames, title="Depth") as bar:
            # Each batch's post-process is queued on the GPU behind its forward
            # and written once the next batch's forward is queued too, so the
            # wait for it overlaps that forward; frames still reach the writer
            # in decode order.
            with OrderedFrameSink(
                self._writeGray, self.recordFailure, bar, maxPending=batchSize
            ) as sink:
                while True:
                    frames = []
                    for _ in range(batchSize):
                        frame = self.readBuffer.read()
                        if frame is None:
                            break
                        frames.append(frame)
                    if frames:
                        self._submitBatch(frames, sink)
                        frameCount += len(frames)
                    # A short read means the decoder's single None sentinel was
                    # just consumed; another read() would block forever.
                    if len(frames) < batchSize:
                        break

        logging.info(f"Processed {frameCount} frames")
        self.writeBuffer.close()

    @torch.inference_mode()
    def _submitBatch(self, frames, sink):
        try:
            grays = _grayMaps(self._inferBatch(frames))
        except Exception as e:
            sink.fail(e, len(frames))
            return
        sink.submit(grays)

    def handleModels(self):
        from .. import depth_anything_3 as depth_anything_3_pkg

        sys.modules.setdefault("depth_anything_3", depth_anything_3_pkg)
        MonocularDepthAnything3 = importlib.import_module(
            "depth_anything_3.mono"
        ).MonocularDepthAnything3
        toDownload = self.depth_method.removeprefix("video_")
        modelMap = {
            "small_v3": "da3-small",
            "base_v3": "da3-base",
            "large_v3": "da3mono-large",
        }
        modelName = modelMap[toDownload]

        self.filename = modelsMap(model=toDownload, modelType="pth", half=self.half)

        modelPath = resolveWeightPath(
            toDownload,
            self.filename,
            half=self.half,
            modelType="pth",
        )

        self.model = MonocularDepthAnything3.from_pretrained(
            modelPath,
            model_name=modelName,
            strict=False,
        ).to(checker.device)

        from ..fold_layerscale import fold_layerscale_
        from ..leanHead import installLeanHead

        fold_layerscale_(self.model)
        installLeanHead(self.model)

        self.processRes = calculateAspectRatio(
            self.width, self.height, self.depthQuality, True
        )
        # processRes is a longest-side target at every quality ("high" is the
        # source's own longest side), so it must bound the longest side. A
        # lower_bound_resize stretched the shortest side to that length -- 4K
        # 16:9 decoded at 6832x3850, OOMing the DPT head even on 24 GB.
        self.processResMethod = "upper_bound_resize"
        scale = self.processRes / max(self.width, self.height)
        tgt_w = max(14, (max(1, round(self.width * scale)) // 14) * 14)
        tgt_h = max(14, (max(1, round(self.height * scale)) // 14) * 14)
        self._decodeWidth = tgt_w
        self._decodeHeight = tgt_h
        self._decodeResize = True

        if self.compileMode != "default":
            try:
                if self.compileMode == "max":
                    self.model.compile(mode="max-autotune-no-cudagraphs")
                elif self.compileMode == "max-graphs":
                    self.model.compile(
                        mode="max-autotune-no-cudagraphs", fullgraph=True
                    )
            except Exception as e:
                logging.error(
                    f"Error compiling model {self.depth_method} with mode {self.compileMode}: {e}"
                )
                logAndPrint(
                    f"Error compiling model {self.depth_method} with mode {self.compileMode}: {e}",
                    "red",
                )

            self.compileMode = "default"

    def _inferBatch(self, frames):
        """DA3 mono depth for a list of frames in ONE forward, as raw fp32 depth
        maps on the model's device, resized to each frame's own resolution."""
        return [_resizeDepth(depth, h, w) for depth, (h, w) in self._inferRaw(frames)]

    def _modelInput(self, frames):
        """Decoded RGB frames -> the normalized [N, 3, H, W] fp32 model input.

        The decoder scales to the model's size, which DA3's ``input_processor``
        (a PIL round trip on the CPU) would leave as it is: its resize only
        fires off that size, and its ToTensor + ImageNet normalization is done
        here on the device instead, bit-identically (a true division by 255, as
        ToTensor's; CUDA turns a division by a Python scalar into a reciprocal
        multiply). A frame at another size -- the decoder's OpenCV fallback does
        not scale -- still goes through ``input_processor``.
        """
        device = self._modelDevice()
        size = (
            getattr(self, "_decodeHeight", None),
            getattr(self, "_decodeWidth", None),
        )
        if all(tuple(frame.shape[:2]) == size for frame in frames):
            # Contiguous NCHW like ToTensor's; in place on that fresh tensor,
            # since a 32-view 1080p chunk is ~0.8 GB per fp32 copy.
            rgb = stackFrames(frames, device).permute(0, 3, 1, 2).contiguous()
            rgb = rgb.float().div_(torch.tensor(255.0, device=device))
            return rgb.sub_(MEANTENSOR.to(device)).div_(STDTENSOR.to(device))
        host = [
            frame.cpu().numpy() if isinstance(frame, torch.Tensor) else frame
            for frame in frames
        ]
        images = [
            self.model.input_processor(
                [frame], None, None, self.processRes, self.processResMethod
            )[0]
            for frame in host
        ]  # [1, C, H, W] each
        return torch.cat(images).to(device, non_blocking=True).float()

    def _modelDevice(self):
        return self.model._get_model_device()

    @torch.inference_mode()
    def _inferRaw(self, frames):
        """DA3 mono depth for a list of frames in ONE forward. Each frame is a
        separate scene with a single view ([B, 1, C, H, W]); the batch dim is
        per-frame independent (no cross-view attention couples them, verified
        empirically incl. sky-mask models). Returns (depth at model resolution
        on the model's device, the frame's (H, W)) per frame; ``_resizeDepth``
        takes it from there."""
        batch = self._modelInput(frames).unsqueeze(1)
        # output_processor's depth, without its copy of every output to numpy.
        depth = self.model.forward(batch)["depth"].squeeze(0).squeeze(-1)
        return [
            (depth[i].squeeze(), tuple(frame.shape[:2]))
            for i, frame in enumerate(frames)
        ]

    @torch.inference_mode()
    def processBatch(self, frames):
        """One batch, inferred, post-processed and written on this thread."""
        try:
            grays = _grayMaps(self._inferBatch(frames))
        except Exception as e:
            self.recordFailure(e)
            logging.exception(f"Something went wrong while processing the frame, {e}")
            return

        # The writer copies on its own stream and never waits on ours.
        if grays and grays[0].is_cuda:
            torch.cuda.current_stream(grays[0].device).synchronize()
        for gray in grays:
            try:
                self._writeGray(gray)
            except Exception as e:
                self.recordFailure(e)
                logging.exception(
                    f"Something went wrong while processing the frame, {e}"
                )

    def _writeGray(self, gray):
        """[H, W] float in [0, 1] -> the [1, 1, H, W] frame the writer expects.

        Handed over on its device, at the model's size: the writer resizes it
        to the output on its own CUDA stream. The caller has finished the work
        that produced it (``OrderedFrameSink`` waits for its event)."""
        if not isinstance(gray, torch.Tensor):
            gray = torch.from_numpy(np.ascontiguousarray(gray, dtype=np.float32))
        self.writeBuffer.write(gray.float()[None, None])
