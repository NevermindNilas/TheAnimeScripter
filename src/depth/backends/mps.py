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
from src.depth.backends._batch import iterBatches
from src.depth.backends._shared import (
    DepthRunOutcome,
    aspectKeepingInputSize,
    calculateAspectRatio,
    limboDisparity,
    limboResolution,
    longestSideShape,
)
from src.infra.logAndPrint import logAndPrint
from src.infra.progressBarLogic import ProgressBarLogic
from src.io.ffmpegSettings import BuildBuffer, createWriteBuffer
from src.model.download import resolveWeightPath
from src.model.registry import modelsMap

if ADOBE:
    from src.server.aeComms import progressState


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def ignoreMpsCompileMode(driver):
    """Nothing is compiled on MPS: say so once and fall back to the default mode."""
    if driver.compileMode == "default":
        return
    logAndPrint(
        f"MPS depth cannot use compileMode '{driver.compileMode}'; running with 'default'.",
        "yellow",
    )
    driver.compileMode = "default"


def _requireMps():
    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "This depth method needs MPS, and torch reports MPS as unavailable here"
        )


def _frameFailed(driver, error):
    """Record a swallowed frame/batch error; call from inside the ``except``."""
    driver.recordFailure(error)
    logging.exception(f"Something went wrong while processing the frame, {error}")


def _stackTensors(frames):
    return frames[0] if len(frames) == 1 else torch.cat(frames, dim=0)


def _disparityGray(depth):
    """One raw DA3 depth map -> [0, 1] disparity with a 2/98-percentile stretch.

    Two scalar percentile calls on purpose: asking for both quantiles at once
    interpolates in another precision and moves the upper bound by ~1e-6.
    """
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    valid = depth > 0
    if valid.sum() <= 10:
        return np.zeros(depth.shape, dtype=np.float32)
    disparity = np.zeros_like(depth, dtype=np.float32)
    disparity[valid] = 1.0 / depth[valid]
    samples = disparity[valid]
    low = np.percentile(samples, 2)
    high = np.percentile(samples, 98)
    if low == high:
        low -= 1e-6
        high += 1e-6
    return ((disparity - low) / (high - low)).clip(0, 1)


class _MpsDepthRun(DepthRunOutcome):
    """What the MPS depth classes share: arguments, load status, the frame loop."""

    def _storeArguments(
        self,
        input,
        output,
        width,
        height,
        fps,
        half,
        inpoint,
        outpoint,
        encode_method,
        depth_method,
        benchmark,
        totalFrames,
        bitDepth,
        depthQuality,
        compileMode,
        depth_batch,
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
        self.device = torch.device("mps")
        self.baseMethod = depth_method.replace("-mps", "")

    def _announceLoad(self):
        if ADOBE:
            progressState.update(
                {"status": f"Loading depth model: {self.depth_method}..."}
            )

    def process(self):
        frameCount = 0
        with ProgressBarLogic(self.totalFrames, title="Depth") as bar:
            for frames in iterBatches(self.readBuffer.read, self.depthBatch):
                self.processBatch(frames)
                frameCount += len(frames)
                bar(len(frames))

        logging.info(f"Processed {frameCount} frames")

        self.writeBuffer.close()


class DepthMPS(_MpsDepthRun):
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
        depth_method="small_v2-mps",
        benchmark=False,
        totalFrames=0,
        bitDepth: str = "16bit",
        depthQuality: str = "high",
        compileMode: str = "default",
        depth_batch: int = 1,
    ):
        _requireMps()
        self._storeArguments(
            input,
            output,
            width,
            height,
            fps,
            half,
            inpoint,
            outpoint,
            encode_method,
            depth_method,
            benchmark,
            totalFrames,
            bitDepth,
            depthQuality,
            compileMode,
            depth_batch,
        )

        self.handleModels()

        # After handleModels: a subclass loader decides dtype and half.
        self.meanTensor = torch.tensor(IMAGENET_MEAN, device=self.device).view(
            1, 3, 1, 1
        )
        self.stdTensor = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        if self.dtype == torch.float16:
            self.meanTensor = self.meanTensor.half()
            self.stdTensor = self.stdTensor.half()

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
        self._announceLoad()

        from ..dpt_v2 import DepthAnythingV2

        if self.baseMethod != "small_v2":
            raise ValueError(f"{self.depth_method} has no MPS depth implementation")

        self.filename = modelsMap(
            model=self.baseMethod, modelType="pth", half=self.half
        )
        modelPath = resolveWeightPath(
            self.baseMethod,
            self.filename,
            half=self.half,
            modelType="pth",
        )

        self.model = DepthAnythingV2(
            encoder="vits", features=64, out_channels=[48, 96, 192, 384]
        )
        stateDict = torch.load(modelPath, map_location="cpu")
        self.model.load_state_dict(stateDict)
        del stateDict

        self.newHeight, self.newWidth = aspectKeepingInputSize(
            self.width, self.height, self.depthQuality
        )

        self._settlePrecision()
        self.model = self.model.to(self.device)

        ignoreMpsCompileMode(self)
        self._warmup()

    def _settlePrecision(self):
        """fp16 when asked for and the module converts, fp32 otherwise."""
        self.model = self.model.eval()
        if self.half:
            try:
                self.model = self.model.half()
                self.dtype = torch.float16
                return
            except Exception as e:
                logging.warning(
                    f"MPS depth model failed in fp16 ({e}); continuing in fp32"
                )
                self.half = False
        self.model = self.model.float()
        self.dtype = torch.float32

    @torch.inference_mode()
    def _warmup(self):
        # Straight through the model: normFrame's tensors do not exist yet.
        dummy = torch.zeros(
            (1, 3, self.newHeight, self.newWidth),
            device=self.device,
            dtype=self.dtype,
        )
        for _ in range(2):
            self.model(dummy)
        torch.mps.synchronize()

    @torch.inference_mode()
    def normFrame(self, frame):
        frame = frame.to(device=self.device, dtype=self.dtype)
        return (frame - self.meanTensor) / self.stdTensor

    @torch.inference_mode()
    def processBatch(self, frames):
        try:
            depth = self.model(self.normFrame(_stackTensors(frames)))
            depth = F.interpolate(
                depth,
                (self.height, self.width),
                mode="bilinear",
                align_corners=True,
            ).cpu()
            torch.mps.synchronize()
            # Per-frame min-max, so batching never changes a frame's output.
            for i in range(depth.shape[0]):
                frame = depth[i : i + 1]
                low = frame.min()
                self.writeBuffer.write((frame - low) / (frame.max() - low))
        except Exception as e:
            _frameFailed(self, e)


class LimboMPS(DepthMPS):
    """Limbo (anime-finetuned Depth Anything 3 small) on Apple Silicon.

    Same tensor path as LimboCuda: limboResolution fixes the input size before
    the first frame, so the frames are decoded straight at it.
    """

    def handleModels(self):
        self._announceLoad()

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
        )

        from ..fold_layerscale import fold_layerscale_

        fold_layerscale_(self.model)

        self.newHeight, self.newWidth = limboResolution(
            self.width, self.height, self.depthQuality
        )

        # fp32 whatever --precision says: DA3's forward only autocasts on CUDA, so a
        # halved model would run the whole DINOv2 stack in pure fp16 here.
        self.model = self.model.float().eval().to(self.device)
        self.dtype = torch.float32
        self.half = False

        ignoreMpsCompileMode(self)
        self._warmup()

    @torch.inference_mode()
    def _warmup(self):
        # DA3 layout: [batch, views, C, H, W].
        dummy = torch.zeros(
            (1, 1, 3, self.newHeight, self.newWidth),
            device=self.device,
            dtype=self.dtype,
        )
        for _ in range(2):
            self.model.forward(dummy)
        torch.mps.synchronize()

    @torch.inference_mode()
    def processBatch(self, frames):
        try:
            batch = self.normFrame(_stackTensors(frames))
            # One frame is one single-view scene: [B, 1, C, H, W] in.
            depth = self.model.forward(batch.unsqueeze(1))["depth"][:, 0]
            depth = depth.unsqueeze(1).cpu()  # [B, 1, h, w]
            torch.mps.synchronize()
            size = (self.height, self.width)
            for i in range(depth.shape[0]):
                gray = limboDisparity(depth[i : i + 1])
                if tuple(gray.shape[-2:]) != size:
                    gray = F.interpolate(
                        gray, size, mode="bilinear", align_corners=True
                    ).clamp(0.0, 1.0)
                self.writeBuffer.write(gray)
        except Exception as e:
            _frameFailed(self, e)


class DepthV3MPS(_MpsDepthRun):
    """Depth Anything 3 on MPS, one frame per scene."""

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
        depth_method="small_v3-mps",
        benchmark=False,
        totalFrames=0,
        bitDepth: str = "16bit",
        depthQuality: str = "high",
        compileMode: str = "default",
        depth_batch: int = 1,
    ):
        _requireMps()
        self._storeArguments(
            input,
            output,
            width,
            height,
            fps,
            half,
            inpoint,
            outpoint,
            encode_method,
            depth_method,
            benchmark,
            totalFrames,
            bitDepth,
            depthQuality,
            compileMode,
            depth_batch,
        )

        self.handleModels()

        decodeWidth = getattr(self, "_decodeWidth", self.width)
        decodeHeight = getattr(self, "_decodeHeight", self.height)
        decodeResize = getattr(self, "_decodeResize", False)
        try:
            self.readBuffer = BuildBuffer(
                videoInput=self.input,
                inpoint=self.inpoint,
                outpoint=self.outpoint,
                width=decodeWidth,
                height=decodeHeight,
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

    def handleModels(self):
        self._announceLoad()

        from .. import depth_anything_3 as depth_anything_3_pkg

        sys.modules.setdefault("depth_anything_3", depth_anything_3_pkg)
        MonocularDepthAnything3 = importlib.import_module(
            "depth_anything_3.mono"
        ).MonocularDepthAnything3

        modelName = {
            "small_v3": "da3-small",
            "base_v3": "da3-base",
            "large_v3": "da3mono-large",
        }[self.baseMethod]

        self.filename = modelsMap(
            model=self.baseMethod, modelType="pth", half=self.half
        )
        modelPath = resolveWeightPath(
            self.baseMethod,
            self.filename,
            half=self.half,
            modelType="pth",
        )

        self.model = MonocularDepthAnything3.from_pretrained(
            modelPath,
            model_name=modelName,
            strict=False,
        ).to(self.device)

        from ..fold_layerscale import fold_layerscale_

        fold_layerscale_(self.model)

        self.processRes = calculateAspectRatio(
            self.width, self.height, self.depthQuality, True
        )
        # processRes bounds the LONGEST side; a lower-bound resize once blew a
        # 4K frame up to 6832x3850.
        self.processResMethod = "upper_bound_resize"
        self._decodeHeight, self._decodeWidth = longestSideShape(
            self.width, self.height, self.processRes
        )
        self._decodeResize = True

        ignoreMpsCompileMode(self)

    @torch.inference_mode()
    def _inferBatch(self, frames):
        """Raw fp32 depth per uint8 [H, W, 3] frame, at the frame's own size."""
        import cv2

        images = []
        sizes = []
        for frame in frames:
            processed = self.model.input_processor(
                [frame], None, None, self.processRes, self.processResMethod
            )
            images.append(processed[0])  # [1, C, h, w]
            sizes.append(tuple(frame.shape[:2]))

        # [N, 1, C, h, w]: N single-view scenes in one forward.
        batch = torch.stack(images).to(self.device).float()
        prediction = self.model.output_processor(self.model.forward(batch))
        torch.mps.synchronize()

        depths = []
        for i, (height, width) in enumerate(sizes):
            depth = np.squeeze(np.asarray(prediction.depth[i]))
            if depth.shape != (height, width):
                depth = cv2.resize(
                    depth, (width, height), interpolation=cv2.INTER_LINEAR
                )
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            depths.append(depth.astype(np.float32))
        return depths

    @torch.inference_mode()
    def processBatch(self, frames):
        try:
            depths = self._inferBatch(frames)
        except Exception as e:
            _frameFailed(self, e)
            return

        for depth in depths:
            try:
                self._writeGray(_disparityGray(depth))
            except Exception as e:
                _frameFailed(self, e)

    def _writeGray(self, gray):
        """[H, W] float in [0, 1] -> the [1, 1, H, W] CPU frame the writer expects."""
        gray = torch.from_numpy(np.ascontiguousarray(gray, dtype=np.float32))
        self.writeBuffer.write(gray[None, None])
