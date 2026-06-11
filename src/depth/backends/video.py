import os
os.environ.setdefault("DA3_LOG_LEVEL", "ERROR")

import torch
import logging
import numpy as np
import cv2

from concurrent.futures import ThreadPoolExecutor
from src.io.ffmpegSettings import (
    BuildBuffer,
)
from src.model.download import resolveWeightPath
from src.model.registry import modelsMap
from src.infra.progressBarLogic import ProgressBarLogic
from src.infra.isCudaInit import CudaChecker
from queue import Queue
from src.constants import ADOBE

if ADOBE:
    pass

from collections import deque
import statistics

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

class VideoDepthAnythingCUDA:
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
        depth_method="og_video_small_v2",
        custom_encoder="",
        benchmark=False,
        totalFrames=0,
        bitDepth: str = "16bit",
        depthQuality: str = "high",
        compileMode: str = "default",
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
        self.encodeBuffer = Queue(maxsize=10)

        self.handleModels()
        try:
            self.readBuffer = BuildBuffer(
                videoInput=self.input,
                inpoint=self.inpoint,
                outpoint=self.outpoint,
                width=self.width,
                height=self.height,
                half=self.half,
                resize=False,
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
                executor.submit(self.process_nelux)

        except Exception as e:
            logging.exception(f"Something went wrong, {e}")

    def handleModels(self):
        from ..video_depth_anything.video_depth_stream import VideoDepthAnything

        self.filename = modelsMap(
            model=self.depth_method, modelType="pth", half=self.half
        )

        modelPath = resolveWeightPath(
            self.depth_method,
            self.filename,
            half=self.half,
            modelType="pth",
        )

        model_configs = {
            "vits": {
                "encoder": "vits",
                "features": 64,
                "out_channels": [48, 96, 192, 384],
            },
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            "vitl": {
                "encoder": "vitl",
                "features": 256,
                "out_channels": [256, 512, 1024, 1024],
            },
        }

        if self.depth_method == "og_video_small_v2":
            encoder = "vits"
        elif self.depth_method == "og_video_base_v2":
            encoder = "vitb"
        elif self.depth_method == "og_video_large_v2":
            encoder = "vitl"
            
        self.model = VideoDepthAnything(**model_configs[encoder])

        self.model.load_state_dict(
            torch.load(modelPath, map_location="cpu"), strict=True
        )


        self.model = self.model.to(checker.device).eval()
        #self.model.half() if self.half else self.model.float()
        self.device = "cuda" if checker.cudaAvailable else "cpu"

    def _resetVideoDepthState(self):
        self.model.transform = None
        self.model.frame_id_list = []
        self.model.frame_cache_list = []
        self.model.id = -1

    def processFrame(self, frame):
        try:
            depth = self.model.infer_video_depth_one(frame, 518, self.device, True)
            min_val, max_val = depth.min(), depth.max()
            depth = ((depth - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            depth = np.stack([depth] * 3, axis=-1)
            self.encodeBuffer.put(depth)
        except Exception as e:
            self._resetVideoDepthState()
            logging.exception(f"Something went wrong while processing the frame, {e}")

    def process_nelux(self):
        """Process using Nelux-backed BuildBuffer decoding."""
        frameCount = 0
        self._resetVideoDepthState()
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


class VideoDepthAnythingTorch:
    """Video Depth pipeline using Nelux-backed decoding and PyTorch for processing."""
    
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
        depth_method="video_small_v2",
        custom_encoder="",
        benchmark=False,
        totalFrames=0,
        bitDepth: str = "16bit",
        depthQuality: str = "high",
        compileMode: str = "default",
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
        self.encodeBuffer = Queue(maxsize=10)

        self.handleModels()
        try:
            self.readBuffer = BuildBuffer(
                videoInput=self.input,
                inpoint=self.inpoint,
                outpoint=self.outpoint,
                width=self.width,
                height=self.height,
                half=self.half,
                resize=False,
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
                executor.submit(self.process_nelux)

        except Exception as e:
            logging.exception(f"Something went wrong, {e}")

    def handleModels(self):
        from ..video_depth_anything.video_depth_stream import VideoDepthAnything

        # Map video_small_v2 -> og_video_small_v2 weights
        weights_model = self.depth_method.replace("video_", "og_video_")
        self.filename = modelsMap(
            model=weights_model, modelType="pth", half=self.half
        )

        modelPath = resolveWeightPath(
            weights_model,
            self.filename,
            half=self.half,
            modelType="pth",
        )

        model_configs = {
            "vits": {
                "encoder": "vits",
                "features": 64,
                "out_channels": [48, 96, 192, 384],
            },
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            "vitl": {
                "encoder": "vitl",
                "features": 256,
                "out_channels": [256, 512, 1024, 1024],
            },
        }

        if "small" in self.depth_method:
            encoder = "vits"
        elif "large" in self.depth_method:
            encoder = "vitl"
        else:
            encoder = "vits"
            
        self.model = VideoDepthAnything(**model_configs[encoder])

        self.model.load_state_dict(
            torch.load(modelPath, map_location="cpu"), strict=True
        )

        self.model = self.model.to(checker.device).eval()
        self.device = "cuda" if checker.cudaAvailable else "cpu"

    def _resetVideoDepthState(self):
        self.model.transform = None
        self.model.frame_id_list = []
        self.model.frame_cache_list = []
        self.model.id = -1

    @torch.inference_mode()
    def processFrame(self, frame):
        """Process a single frame - frame is a PyTorch tensor (C, H, W) or numpy array."""
        try:
            # If tensor, convert to numpy for the model (it expects numpy RGB)
            if isinstance(frame, torch.Tensor):
                if frame.dim() == 3:  # C, H, W
                    frame = frame.permute(1, 2, 0).cpu().numpy()
                elif frame.dim() == 4:  # B, C, H, W
                    frame = frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
            
            # Ensure RGB format and uint8
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            
            depth = self.model.infer_video_depth_one(frame, 518, self.device, True)
            min_val, max_val = depth.min(), depth.max()
            depth = ((depth - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            depth = np.stack([depth] * 3, axis=-1)
            self.encodeBuffer.put(depth)
        except Exception as e:
            self._resetVideoDepthState()
            logging.exception(f"Something went wrong while processing the frame, {e}")

    def process_nelux(self):
        """Process using Nelux-backed BuildBuffer decoding."""
        frameCount = 0

        self._resetVideoDepthState()
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
