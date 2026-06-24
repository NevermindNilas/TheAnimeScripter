import os
import sys

os.environ.setdefault("DA3_LOG_LEVEL", "ERROR")

import importlib
import logging
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.constants import ADOBE
from src.infra.isCudaInit import CudaChecker
from src.infra.logAndPrint import logAndPrint
from src.infra.progressBarLogic import ProgressBarLogic
from src.io.ffmpegSettings import (
    BuildBuffer,
    WriteBuffer,
)
from src.model.downloadModels import (
    modelsMap,
    resolveWeightPath,
)

if ADOBE:
    from src.server.aeComms import progressState

import statistics
from collections import deque

checker = CudaChecker()


class SlidingWindowNormalizer:
    """
    Causal window-median percentile deflicker for depth streams.

    Smooths only the normalization range (low/high percentiles) across a
    sliding window of past frames, then rescales each frame independently.
    No pixel-level temporal blending, so no motion ghosting. Resets the
    window when per-frame percentiles diverge sharply from the running
    median (scene cut / abrupt exposure change).
    """

    def __init__(
        self,
        low_pct: float = 2.0,
        high_pct: float = 98.0,
        window_size: int = 7,
        scene_cut_ratio: float = 2.5,
        scene_cut_shift: float = 1.0,
    ):
        self.low_q = low_pct / 100.0
        self.high_q = high_pct / 100.0
        self.window_size = window_size
        self.scene_cut_ratio = scene_cut_ratio
        self.scene_cut_shift = scene_cut_shift
        self.lows: deque = deque(maxlen=window_size)
        self.highs: deque = deque(maxlen=window_size)

    def normalize(self, depth, mask=None):
        is_numpy = isinstance(depth, np.ndarray)
        if is_numpy:
            t = torch.from_numpy(depth).float()
        else:
            t = depth if depth.dtype.is_floating_point else depth.float()

        if mask is not None:
            sample = t[mask]
            if sample.numel() == 0:
                sample = t.flatten()
        else:
            sample = t.flatten()

        if sample.dtype not in (torch.float32, torch.float64):
            sample = sample.float()

        cur_low = torch.quantile(sample, self.low_q).item()
        cur_high = torch.quantile(sample, self.high_q).item()

        if len(self.lows) >= 3:
            med_low = statistics.median(self.lows)
            med_high = statistics.median(self.highs)
            med_range = max(med_high - med_low, 1e-6)
            cur_range = max(cur_high - cur_low, 1e-6)
            ratio = cur_range / med_range
            shift = abs(cur_low - med_low) / med_range
            if (
                ratio > self.scene_cut_ratio
                or ratio < 1.0 / self.scene_cut_ratio
                or shift > self.scene_cut_shift
            ):
                self.lows.clear()
                self.highs.clear()

        self.lows.append(cur_low)
        self.highs.append(cur_high)

        target_low = statistics.median(self.lows)
        target_high = statistics.median(self.highs)
        denom = max(target_high - target_low, 1e-6)

        normalized = ((t - target_low) / denom).clamp(0.0, 1.0)

        if is_numpy:
            return normalized.cpu().numpy()
        return normalized


MEANTENSOR = (
    torch.tensor([0.485, 0.456, 0.406]).contiguous().view(3, 1, 1).to(checker.device)
)
STDTENSOR = (
    torch.tensor([0.229, 0.224, 0.225]).contiguous().view(3, 1, 1).to(checker.device)
)
MEANTENSOR_HALF = MEANTENSOR.half() if checker.cudaAvailable else MEANTENSOR
STDTENSOR_HALF = STDTENSOR.half() if checker.cudaAvailable else STDTENSOR


def calculateAspectRatio(width, height, depthQuality="high", isV3=False):
    if isV3:
        if depthQuality == "high":
            return ((max(width, height) + 13) // 14) * 14
        if depthQuality == "medium":
            return 700
        return 518

    if depthQuality == "high":
        # Whilst this doesn't necessarily allign with the model, it produces
        # sharper results at the cost of performance and some accuracy loss.
        newHeight = ((height + 13) // 14) * 14
        newWidth = ((width + 13) // 14) * 14
    else:
        # I'd suggest 700px as a good middle ground for resizing
        size = 700 if depthQuality == "medium" else 518
        newHeight = size
        newWidth = size

    logging.info(f"Depth Padding: {newWidth}x{newHeight}")
    return newHeight, newWidth


class DepthCuda:
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
        custom_encoder="",
        benchmark=False,
        totalFrames=0,
        bitDepth: str = "16bit",
        depthQuality: str = "high",
        compileMode: str = "default",
        depthNorm: bool = False,
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
        self.custom_encoder = custom_encoder
        self.benchmark = benchmark
        self.totalFrames = totalFrames
        self.bitDepth = bitDepth
        self.depthQuality = depthQuality
        self.compileMode = compileMode
        self.normalizer = SlidingWindowNormalizer() if depthNorm else None

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

            self.writeBuffer = WriteBuffer(
                self.input,
                self.output,
                self.encode_method,
                self.custom_encoder,
                self.width,
                self.height,
                self.fps,
                grayscale=True,
                benchmark=self.benchmark,
                bitDepth=self.bitDepth,
            )

            with ThreadPoolExecutor(max_workers=3) as executor:
                executor.submit(self.writeBuffer)
                executor.submit(self.readBuffer)
                executor.submit(self.process)

        except Exception as e:
            logging.exception(f"Something went wrong, {e}")

    def handleModels(self):
        if ADOBE:
            progressState.update(
                {"status": f"Loading depth model: {self.depth_method}..."}
            )

        from ..dpt_v2 import DepthAnythingV2

        match self.depth_method:
            case "small_v2" | "distill_small_v2":
                method = "vits"
            case "base_v2" | "distill_base_v2":
                method = "vitb"
            case "large_v2" | "distill_large_v2":
                method = "vitl"
            case "giant_v2":
                method = "vitg"

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
            "vitb": {
                "encoder": "vitb",
                "features": 128,
                "out_channels": [96, 192, 384, 768],
            },
            "vitl": {
                "encoder": "vitl",
                "features": 256,
                "out_channels": [256, 512, 1024, 1024],
            },
            "vitg": {
                "encoder": "vitg",
                "features": 384,
                "out_channels": [1536, 1536, 1536, 1536],
            }
            if "distill" not in self.depth_method
            else {
                "encoder": "vitl",
                "features": 256,
                "out_channels": [256, 512, 1024, 1024],
                "use_bn": False,
                "use_clstoken": False,
                "max_depth": 150.0,
                "mode": "disparity",
                "pretrain_type": "dinov2",
                "del_mask_token": False,
            },
        }

        if "distill" in self.depth_method and "large" in self.depth_method:
            from src.depth.distillanydepth.modeling.archs.dam.dam import DepthAnything

            self.model = DepthAnything(**modelConfigs[method])
        else:
            self.model = DepthAnythingV2(**modelConfigs[method])

        if "distill" in self.depth_method:
            from safetensors.torch import load_file

            modelWeights = load_file(modelPath)
            self.model.load_state_dict(modelWeights)
            del modelWeights
        else:
            stateDict = torch.load(modelPath, map_location="cpu")
            self.model.load_state_dict(stateDict)
            del stateDict

        if self.half and checker.cudaAvailable:
            self.model = self.model.half()
        self.model = self.model.to(checker.device).eval()
        torch.cuda.empty_cache()

        self.newHeight, self.newWidth = calculateAspectRatio(
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
    def outputFrameNorm(self, depth):
        depth = F.interpolate(
            depth,
            (self.height, self.width),
            mode="bilinear",
            align_corners=True,
        )
        if self.normalizer is not None:
            return self.normalizer.normalize(depth)
        return (depth - depth.min()) / (depth.max() - depth.min())

    @torch.inference_mode()
    def processFrame(self, frame):
        try:
            with torch.cuda.stream(self.stream):
                frame = self.normFrame(frame)
                depth = self.model(frame)
                depth = self.outputFrameNorm(depth)
            self.stream.synchronize()
            self.writeBuffer.write(depth)

        except Exception as e:
            logging.exception(f"Something went wrong while processing the frame, {e}")

    def process(self):
        frameCount = 0
        currentFrame = self.readBuffer.read()
        nextFrame = self.readBuffer.read() if currentFrame is not None else None
        with ProgressBarLogic(self.totalFrames) as bar:
            while currentFrame is not None:
                self.processFrame(currentFrame)
                frameCount += 1
                bar(1)
                currentFrame = nextFrame
                if currentFrame is not None:
                    nextFrame = self.readBuffer.read()

        logging.info(f"Processed {frameCount} frames")

        self.writeBuffer.close()


class OGDepthV2CUDA:
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
        custom_encoder="",
        benchmark=False,
        totalFrames=0,
        bitDepth: str = "16bit",
        depthQuality: str = "high",
        compileMode: str = "default",
        depthNorm: bool = False,
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
        self.custom_encoder = custom_encoder
        self.benchmark = benchmark
        self.totalFrames = totalFrames
        self.bitDepth = bitDepth
        self.depthQuality = depthQuality
        self.compileMode = compileMode
        self.normalizer = SlidingWindowNormalizer() if depthNorm else None
        self.encodeBuffer = Queue(maxsize=10)

        self.handleModels()

        if not hasattr(self, "newHeight") or not hasattr(self, "newWidth"):
            self.newHeight, self.newWidth = calculateAspectRatio(
                self.width, self.height, self.depthQuality
            )
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
            )

            self.output = cv2.VideoWriter(
                self.output,
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps,
                (self.width, self.height),
            )
            with ThreadPoolExecutor(max_workers=3) as executor:
                executor.submit(self.readBuffer)
                executor.submit(self.encodeThread)
                executor.submit(self.process)

        except Exception as e:
            logging.exception(f"Something went wrong, {e}")

    def handleModels(self):
        from ..og_dpt_v2 import DepthAnythingV2

        match self.depth_method:
            case "og_small_v2" | "og_distill_small_v2":
                method = "vits"
                toDownload = "small_v2"
            case "og_base_v2" | "og_distill_base_v2":
                method = "vitb"
                toDownload = "base_v2"
            case "og_large_v2" | "og_distill_large_v2":
                method = "vitl"
                toDownload = "large_v2"
            case "og_giant_v2":
                method = "vitg"
                toDownload = "giant_v2"

        modelType = "pth"
        self.filename = modelsMap(model=toDownload, modelType=modelType, half=self.half)

        modelPath = resolveWeightPath(
            toDownload,
            self.filename,
            half=self.half,
            modelType=modelType,
        )

        modelConfigs = {
            "vits": {
                "encoder": "vits",
                "features": 64,
                "out_channels": [48, 96, 192, 384],
            },
            "vitb": {
                "encoder": "vitb",
                "features": 128,
                "out_channels": [96, 192, 384, 768],
            },
            "vitl": {
                "encoder": "vitl",
                "features": 256,
                "out_channels": [256, 512, 1024, 1024],
            },
            "vitg": {
                "encoder": "vitg",
                "features": 384,
                "out_channels": [1536, 1536, 1536, 1536],
            }
            if "distill" not in self.depth_method
            else {
                "encoder": "vitl",
                "features": 256,
                "out_channels": [256, 512, 1024, 1024],
                "use_bn": False,
                "use_clstoken": False,
                "max_depth": 150.0,
                "mode": "disparity",
                "pretrain_type": "dinov2",
                "del_mask_token": False,
            },
        }

        if "distill" in self.depth_method and "large" in self.depth_method:
            from src.depth.distillanydepth.modeling.archs.dam.dam import DepthAnything

            self.model = DepthAnything(**modelConfigs[method])
        else:
            self.model = DepthAnythingV2(**modelConfigs[method])

        self.model.load_state_dict(torch.load(modelPath, map_location="cpu"))

        self.model = self.model.to(checker.device).eval()

        self.newHeight, self.newWidth = calculateAspectRatio(
            self.width, self.height, self.depthQuality
        )

        if self.half and checker.cudaAvailable:
            self.model = self.model.half()
        else:
            self.model = self.model.float()

        self.normStream = torch.cuda.Stream()
        self.stream = torch.cuda.Stream()

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
                    f"Error compiling model {self.interpolateMethod} with mode {self.compileMode}: {e}"
                )
                logAndPrint(
                    f"Error compiling model {self.interpolateMethod} with mode {self.compileMode}: {e}",
                    "red",
                )

            self.compileMode = "default"

    @torch.inference_mode()
    def processFrame(self, frame):
        try:
            if self.normalizer is not None:
                image, (h, w) = self.model.image2tensor(frame, self.newHeight)
                image = image.half() if self.half else image.float()
                depth = self.model.forward(image)
                depth = depth.unsqueeze(1)
                depth = F.interpolate(
                    depth, (h, w), mode="bilinear", align_corners=True
                )
                depth = self.normalizer.normalize(depth[0, 0])
                depth = (depth * 255.0).byte()
                depth = depth.repeat(3, 1, 1).permute(1, 2, 0).cpu().numpy()
            else:
                depth = self.model.infer_image(frame, self.newHeight, self.half)
            self.encodeBuffer.put(depth)
        except Exception as e:
            logging.exception(f"Something went wrong while processing the frame, {e}")

    def process(self):
        frameCount = 0

        currentFrame = self.readBuffer.read()
        nextFrame = self.readBuffer.read() if currentFrame is not None else None
        with ProgressBarLogic(self.totalFrames) as bar:
            while currentFrame is not None:
                self.processFrame(currentFrame)
                frameCount += 1
                bar(1)
                currentFrame = nextFrame
                if currentFrame is not None:
                    nextFrame = self.readBuffer.read()

        logging.info(f"Processed {frameCount} frames")
        self.encodeBuffer.put(None)

    def encodeThread(self):
        while True:
            frame = self.encodeBuffer.get()
            if frame is None:
                break
            self.output.write(frame)

        self.output.release()


class OGDepthV3Cuda(OGDepthV2CUDA):
    def handleModels(self):
        from .. import depth_anything_3 as depth_anything_3_pkg

        sys.modules.setdefault("depth_anything_3", depth_anything_3_pkg)
        MonocularDepthAnything3 = importlib.import_module(
            "depth_anything_3.mono"
        ).MonocularDepthAnything3
        toDownload = self.depth_method
        modelMap = {
            "small_v3": "da3-small",
            "base_v3": "da3-base",
            "og_large_v3": "da3metric-large",
        }
        modelName = modelMap[self.depth_method]

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

        fold_layerscale_(self.model)

        self.newHeight, self.newWidth = calculateAspectRatio(
            self.width, self.height, self.depthQuality
        )
        self.processRes = calculateAspectRatio(
            self.width, self.height, self.depthQuality, True
        )
        self.processResMethod = (
            "lower_bound_resize"
            if self.depthQuality == "high"
            else "upper_bound_resize"
        )

        if self.processResMethod == "upper_bound_resize":
            scale = self.processRes / max(self.width, self.height)
        else:
            scale = self.processRes / min(self.width, self.height)
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

    @torch.inference_mode()
    def processFrame(self, frame):
        try:
            depth = self.model.infer_image(
                frame,
                process_res=self.processRes,
                process_res_method=self.processResMethod,
            )
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            validMask = depth > 0

            if validMask.sum() <= 10:
                gray = np.zeros(depth.shape, dtype=np.uint8)
                self.encodeBuffer.put(np.stack([gray] * 3, axis=-1))
                return

            disparity = np.zeros_like(depth, dtype=np.float32)
            disparity[validMask] = 1.0 / depth[validMask]

            if self.normalizer is not None:
                gray = self.normalizer.normalize(disparity)
            else:
                disp_min = np.percentile(disparity[validMask], 2)
                disp_max = np.percentile(disparity[validMask], 98)
                if disp_min == disp_max:
                    disp_min -= 1e-6
                    disp_max += 1e-6
                gray = ((disparity - disp_min) / (disp_max - disp_min)).clip(0, 1)
            gray = (gray * 255.0).astype(np.uint8)
            self.encodeBuffer.put(np.stack([gray] * 3, axis=-1))
        except Exception as e:
            logging.exception(f"Something went wrong while processing the frame, {e}")

    def encodeThread(self):
        while True:
            frame = self.encodeBuffer.get()
            if frame is None:
                break
            if frame.ndim == 2:
                frame = np.stack([frame] * 3, axis=-1)
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(
                    frame,
                    (self.width, self.height),
                    interpolation=cv2.INTER_LINEAR,
                )
            self.output.write(frame)

        self.output.release()
