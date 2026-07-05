import os

os.environ.setdefault("DA3_LOG_LEVEL", "ERROR")

import logging
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import cv2
import numpy as np
import torch

from src.infra.isCudaInit import CudaChecker
from src.infra.progressBarLogic import ProgressBarLogic
from src.io.ffmpegSettings import (
    BuildBuffer,
)
from src.model.download import resolveWeightPath
from src.model.registry import modelsMap

checker = CudaChecker()


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
        # self.model.half() if self.half else self.model.float()
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
        self.filename = modelsMap(model=weights_model, modelType="pth", half=self.half)

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
