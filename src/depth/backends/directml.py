import os

os.environ.setdefault("DA3_LOG_LEVEL", "ERROR")

import logging
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.constants import ADOBE
from src.utils.downloadModels import (
    modelsMap,
    resolveWeightPath,
)
from src.utils.ffmpegSettings import (
    BuildBuffer,
    WriteBuffer,
)
from src.utils.isCudaInit import CudaChecker
from src.utils.logAndPrint import logAndPrint
from src.utils.progressBarLogic import ProgressBarLogic

if ADOBE:
    from src.utils.aeComms import progressState

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


class DepthDirectMLV2:
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
        depthNorm: bool = False,
    ):
        import onnxruntime as ort

        self.ort = ort

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
        self.normalizer = SlidingWindowNormalizer() if depthNorm else None

        if "openvino" in depth_method:
            logAndPrint(
                "OpenVINO backend is an experimental feature, please report any issues you encounter.",
                "yellow",
            )
            import openvino  # noqa: F401

        self.handleModels()

        try:
            self.readBuffer = BuildBuffer(
                videoInput=self.input,
                inpoint=self.inpoint,
                outpoint=self.outpoint,
                resize=False,
                width=self.width,
                height=self.height,
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

        method = self.depth_method
        if "openvino" in self.depth_method:
            method = method.replace("openvino", "directml")

        self.filename = modelsMap(model=method, modelType="onnx", half=self.half)

        if "directml" in self.depth_method:
            folderName = self.depth_method.replace("-directml", "-onnx")
        elif "openvino" in self.depth_method:
            folderName = self.depth_method.replace("-openvino", "-onnx")

        modelPath = resolveWeightPath(
            folderName,
            self.filename,
            downloadModel=method,
            half=self.half,
            modelType="onnx",
        )

        providers = self.ort.get_available_providers()

        if (
            "DmlExecutionProvider" in providers
            or "OpenVINOExecutionProvider" in providers
        ):
            if "directml" in self.depth_method:
                logging.info("DirectML provider available. Defaulting to DirectML")
                self.model = self.ort.InferenceSession(
                    modelPath, providers=["DmlExecutionProvider"]
                )
            elif "openvino" in self.depth_method:
                logging.info("Using OpenVINO model")
                self.model = self.ort.InferenceSession(
                    modelPath, providers=["OpenVINOExecutionProvider"]
                )
        else:
            logging.info(
                "DirectML provider not available, falling back to CPU, expect significantly worse performance, ensure that your drivers are up to date and your GPU supports DirectX 12"
            )
            self.model = self.ort.InferenceSession(
                modelPath, providers=["CPUExecutionProvider"]
            )
        # Bind on CPU memory; ORT will handle copies for DML provider
        self.deviceType = "cpu"
        self.device = torch.device(self.deviceType)

        # Calculate padded model resolution (height, width)
        self.newHeight, self.newWidth = calculateAspectRatio(
            self.width, self.height, self.depthQuality
        )

        # Discover actual I/O names and dtypes from the ONNX model
        onnxInputs = self.model.get_inputs()
        onnxOutputs = self.model.get_outputs()
        self.inputName = onnxInputs[0].name
        self.outputName = onnxOutputs[0].name

        def onnxTypeToNumpy(typ: str):
            return np.float16 if "float16" in typ else np.float32

        def onnxTypeToTorch(typ: str):
            return torch.float16 if "float16" in typ else torch.float32

        self.numpyInDType = onnxTypeToNumpy(onnxInputs[0].type)
        self.torchInDType = onnxTypeToTorch(onnxInputs[0].type)
        self.numpyOutDType = onnxTypeToNumpy(onnxOutputs[0].type)
        self.torchOutDType = onnxTypeToTorch(onnxOutputs[0].type)

        # Allocate input/output buffers with correct shapes and dtypes
        self.IoBinding = self.model.io_binding()
        self.dummyInput = torch.zeros(
            (1, 3, self.newHeight, self.newWidth),
            device=self.deviceType,
            dtype=self.torchInDType,
        ).contiguous()

        out_rank = len(onnxOutputs[0].shape) if hasattr(onnxOutputs[0], "shape") else 4
        if out_rank == 3:
            out_shape = (1, self.newHeight, self.newWidth)
        else:
            # Default to NCHW with C=1 when unknown or 4D
            out_shape = (1, 1, self.newHeight, self.newWidth)

        self.dummyOutput = torch.zeros(
            out_shape,
            device=self.deviceType,
            dtype=self.torchOutDType,
        ).contiguous()

        self.IoBinding.bind_output(
            name=self.outputName,
            device_type=self.deviceType,
            device_id=0,
            element_type=self.numpyOutDType,
            shape=self.dummyOutput.shape,
            buffer_ptr=self.dummyOutput.data_ptr(),
        )

        self.usingCpuFallback = False
        self.modelPath = modelPath

    def _fallbackToCpu(self):
        """Reinitialize model and CPU accelerate it after DirectML or OpenVINO fails."""
        logAndPrint(
            "DirectML/OpenVINO encountered an error, falling back to CPU. Performance will be slower.",
            "yellow",
        )

        self.model = self.ort.InferenceSession(
            self.modelPath, providers=["CPUExecutionProvider"]
        )

        self.IoBinding = self.model.io_binding()
        self.IoBinding.bind_output(
            name=self.outputName,
            device_type=self.deviceType,
            device_id=0,
            element_type=self.numpyOutDType,
            shape=self.dummyOutput.shape,
            buffer_ptr=self.dummyOutput.data_ptr(),
        )

        self.usingCpuFallback = True

    @torch.inference_mode()
    def processFrame(self, frame):
        try:
            frame = frame.to(self.device)

            frame = F.interpolate(
                frame,
                size=(self.newHeight, self.newWidth),
                mode="bilinear",
                align_corners=True,
            )

            frame = frame.to(dtype=self.torchInDType)

            self.dummyInput.copy_(frame)
            self.IoBinding.bind_input(
                name=self.inputName,
                device_type=self.deviceType,
                device_id=0,
                element_type=self.numpyInDType,
                shape=self.dummyInput.shape,
                buffer_ptr=self.dummyInput.data_ptr(),
            )

            self.model.run_with_iobinding(self.IoBinding)

            out_tensor = self.dummyOutput.float()
            if out_tensor.ndim == 3:
                out_tensor = out_tensor.unsqueeze(0)
            depth = F.interpolate(
                out_tensor,
                size=(self.height, self.width),
                mode="bilinear",
                align_corners=True,
            )

            if self.normalizer is not None:
                depth = self.normalizer.normalize(depth)
            else:
                depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
            self.writeBuffer.write(depth)

        except UnicodeDecodeError as e:
            if not self.usingCpuFallback:
                logging.warning(f"DirectML/OpenVINO UnicodeDecodeError: {e}")
                self._fallbackToCpu()
                self.processFrame(frame)
            else:
                logging.exception(
                    f"Something went wrong while processing the frame, {e}"
                )

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


class OGDepthV2DirectML:
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
        depth_method="og_small_v2-directml",
        custom_encoder="",
        benchmark=False,
        totalFrames=0,
        bitDepth: str = "16bit",
        depthQuality: str = "high",
        depthNorm: bool = False,
    ):
        import onnxruntime as ort

        self.ort = ort

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
        self.normalizer = SlidingWindowNormalizer() if depthNorm else None
        self.encodeBuffer = Queue(maxsize=10)

        if "openvino" in depth_method:
            logAndPrint(
                "OpenVINO backend is an experimental feature, please report any issues you encounter.",
                "yellow",
            )
            import openvino  # noqa: F401

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

            self.outputWriter = cv2.VideoWriter(
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
        depth_method = self.depth_method
        if "openvino" in depth_method:
            depth_method = depth_method.replace("openvino", "directml")

        if "og_" in depth_method:
            depth_method = depth_method.replace("og_", "")

        self.filename = modelsMap(model=depth_method, modelType="onnx", half=self.half)

        folderName = depth_method.replace("-directml", "-onnx")
        modelPath = resolveWeightPath(
            folderName,
            self.filename,
            downloadModel=depth_method,
            half=self.half,
            modelType="onnx",
        )

        providers = self.ort.get_available_providers()

        if (
            "DmlExecutionProvider" in providers
            or "OpenVINOExecutionProvider" in providers
        ):
            if "directml" in self.depth_method:
                logging.info("DirectML provider available. Defaulting to DirectML")
                self.model = self.ort.InferenceSession(
                    modelPath, providers=["DmlExecutionProvider"]
                )
            elif "openvino" in self.depth_method:
                logging.info("Using OpenVINO model")
                self.model = self.ort.InferenceSession(
                    modelPath, providers=["OpenVINOExecutionProvider"]
                )
        else:
            logging.info("DirectML provider not available, falling back to CPU")
            self.model = self.ort.InferenceSession(
                modelPath, providers=["CPUExecutionProvider"]
            )

        self.deviceType = "cpu"
        self.device = torch.device(self.deviceType)

        self.newHeight, self.newWidth = calculateAspectRatio(
            self.width, self.height, self.depthQuality
        )

        onnxInputs = self.model.get_inputs()
        onnxOutputs = self.model.get_outputs()
        self.inputName = onnxInputs[0].name
        self.outputName = onnxOutputs[0].name

        def onnxTypeToNumpy(typ: str):
            return np.float16 if "float16" in typ else np.float32

        def onnxTypeToTorch(typ: str):
            return torch.float16 if "float16" in typ else torch.float32

        self.numpyInDType = onnxTypeToNumpy(onnxInputs[0].type)
        self.torchInDType = onnxTypeToTorch(onnxInputs[0].type)
        self.numpyOutDType = onnxTypeToNumpy(onnxOutputs[0].type)
        self.torchOutDType = onnxTypeToTorch(onnxOutputs[0].type)

        self.IoBinding = self.model.io_binding()
        self.dummyInput = torch.zeros(
            (1, 3, self.newHeight, self.newWidth),
            device=self.deviceType,
            dtype=self.torchInDType,
        ).contiguous()

        out_rank = len(onnxOutputs[0].shape) if hasattr(onnxOutputs[0], "shape") else 4
        if out_rank == 3:
            out_shape = (1, self.newHeight, self.newWidth)
        else:
            out_shape = (1, 1, self.newHeight, self.newWidth)

        self.dummyOutput = torch.zeros(
            out_shape,
            device=self.deviceType,
            dtype=self.torchOutDType,
        ).contiguous()

        self.IoBinding.bind_output(
            name=self.outputName,
            device_type=self.deviceType,
            device_id=0,
            element_type=self.numpyOutDType,
            shape=self.dummyOutput.shape,
            buffer_ptr=self.dummyOutput.data_ptr(),
        )

    @torch.inference_mode()
    def processFrame(self, frame):
        try:
            frame = torch.from_numpy(frame).to(self.device)
            frame = frame.permute(2, 0, 1).unsqueeze(0)

            frame = F.interpolate(
                frame.float() / 255.0,
                size=(self.newHeight, self.newWidth),
                mode="bilinear",
                align_corners=True,
            )

            frame = (frame - MEANTENSOR.cpu()) / STDTENSOR.cpu()
            frame = frame.to(dtype=self.torchInDType)

            self.dummyInput.copy_(frame)
            self.IoBinding.bind_input(
                name=self.inputName,
                device_type=self.deviceType,
                device_id=0,
                element_type=self.numpyInDType,
                shape=self.dummyInput.shape,
                buffer_ptr=self.dummyInput.data_ptr(),
            )

            self.model.run_with_iobinding(self.IoBinding)

            out_tensor = self.dummyOutput.float()
            if out_tensor.ndim == 4:
                out_tensor = out_tensor.squeeze(1)

            depth = out_tensor[0].cpu().numpy()
            if self.normalizer is not None:
                depth = self.normalizer.normalize(depth)
                depth = (depth * 255.0).astype(np.uint8)
            else:
                depth = (
                    (depth - depth.min()) / (depth.max() - depth.min() + 1e-6) * 255.0
                )
                depth = depth.astype(np.uint8)

            self.encodeBuffer.put(depth)

        except Exception as e:
            logging.exception(f"Something went wrong while processing the frame, {e}")

    def process(self):
        frameCount = 0

        with ProgressBarLogic(self.totalFrames) as bar:
            for _ in range(self.totalFrames):
                frame = self.readBuffer.read()
                if frame is None:
                    break
                self.processFrame(frame)
                frameCount += 1
                bar(1)
                if self.readBuffer.isReadFinished():
                    if self.readBuffer.isQueueEmpty():
                        break

        logging.info(f"Processed {frameCount} frames")
        self.encodeBuffer.put(None)

    def encodeThread(self):
        while True:
            frame = self.encodeBuffer.get()
            if frame is None:
                break
            if frame.ndim == 2:
                frame = np.stack([frame] * 3, axis=-1)
            frame = cv2.resize(
                frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR
            )
            self.outputWriter.write(frame)
        self.outputWriter.release()
