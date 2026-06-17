import os

os.environ.setdefault("DA3_LOG_LEVEL", "ERROR")

import logging
from concurrent.futures import ThreadPoolExecutor

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


class DepthTensorRTV2:
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

        import tensorrt as trt

        from src.utils.trtHandler import (
            tensorRTEngineCreator,
            tensorRTEngineLoader,
            tensorRTEngineNameHandler,
        )

        self.trt = trt
        self.tensorRTEngineCreator = tensorRTEngineCreator
        self.tensorRTEngineLoader = tensorRTEngineLoader
        self.tensorRTEngineNameHandler = tensorRTEngineNameHandler

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
                {"status": f"Loading TensorRT depth model: {self.depth_method}..."}
            )

        self.filename = modelsMap(
            model=self.depth_method, modelType="onnx", half=self.half
        )

        folderName = self.depth_method.replace("-tensorrt", "-onnx")
        self.modelPath = resolveWeightPath(
            folderName,
            self.filename,
            downloadModel=self.depth_method,
            half=self.half,
            modelType="onnx",
        )

        self.newHeight, self.newWidth = calculateAspectRatio(
            self.width, self.height, self.depthQuality
        )

        enginePath = self.tensorRTEngineNameHandler(
            modelPath=self.modelPath,
            fp16=self.half,
            optInputShape=[1, 3, self.newHeight, self.newWidth],
        )

        self.engine, self.context = self.tensorRTEngineLoader(enginePath)
        if (
            self.engine is None
            or self.context is None
            or not os.path.exists(enginePath)
        ):
            inputName = "image" if "distill" not in self.depth_method else "input"
            self.engine, self.context = self.tensorRTEngineCreator(
                modelPath=self.modelPath,
                enginePath=enginePath,
                fp16=self.half,
                inputsMin=[1, 3, self.newHeight, self.newWidth],
                inputsOpt=[1, 3, self.newHeight, self.newWidth],
                inputsMax=[1, 3, self.newHeight, self.newWidth],
                inputName=[inputName],
            )

        self.stream = torch.cuda.Stream()
        self.dummyInput = torch.zeros(
            (1, 3, self.newHeight, self.newWidth),
            device=checker.device,
            dtype=torch.float16 if self.half else torch.float32,
        )

        self.dummyOutput = torch.zeros(
            (1, 1, self.newHeight, self.newWidth),
            device=checker.device,
            dtype=torch.float16 if self.half else torch.float32,
        )

        if "distill" in self.depth_method:
            self.dummyConst = torch.zeros(
                (1, 1, self.newHeight, self.newWidth),
                device=checker.device,
                dtype=torch.float16 if self.half else torch.float32,
            )

        self.bindings = [self.dummyInput.data_ptr(), self.dummyOutput.data_ptr()]

        if "distill" in self.depth_method:
            self.bindings.append(self.dummyConst.data_ptr())

        for i in range(self.engine.num_io_tensors):
            self.context.set_tensor_address(
                self.engine.get_tensor_name(i), self.bindings[i]
            )
            tensor_name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(tensor_name) == self.trt.TensorIOMode.INPUT:
                self.context.set_input_shape(tensor_name, self.dummyInput.shape)

        self.normStream = torch.cuda.Stream()
        self.outputNormStream = torch.cuda.Stream()
        self.cudaGraph = torch.cuda.CUDAGraph()
        self.initTorchCudaGraph()

    @torch.inference_mode()
    def initTorchCudaGraph(self):
        with torch.cuda.graph(self.cudaGraph, stream=self.stream):
            self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        self.stream.synchronize()

    @torch.inference_mode()
    def normFrame(self, frame):
        with torch.cuda.stream(self.normStream):
            frame = F.interpolate(
                frame.float(),
                (self.newHeight, self.newWidth),
                mode="bilinear",
                align_corners=True,
            )
            frame = (frame - MEANTENSOR) / STDTENSOR
            if self.half:
                frame = frame.half()
            self.dummyInput.copy_(frame, non_blocking=True)
        self.normStream.synchronize()

    @torch.inference_mode()
    def normOutputFrame(self):
        with torch.cuda.stream(self.outputNormStream):
            depth = F.interpolate(
                self.dummyOutput,
                size=[self.height, self.width],
                mode="bilinear",
                align_corners=True,
            )
            if self.normalizer is not None:
                depth = self.normalizer.normalize(depth)
            else:
                depth = (depth - depth.min()) / (depth.max() - depth.min())
        self.outputNormStream.synchronize()
        return depth

    @torch.inference_mode()
    def processFrame(self, frame):
        try:
            self.normFrame(frame)
            with torch.cuda.stream(self.stream):
                self.cudaGraph.replay()
            self.stream.synchronize()
            depth = self.normOutputFrame()

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


class OGDepthV2TensorRT:
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
        depth_method="og_small_v2",
        custom_encoder="",
        benchmark=False,
        totalFrames=0,
        bitDepth: str = "16bit",
        depthQuality: str = "high",
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
        self.normalizer = SlidingWindowNormalizer() if depthNorm else None

        import tensorrt as trt

        from src.utils.trtHandler import (
            tensorRTEngineCreator,
            tensorRTEngineLoader,
            tensorRTEngineNameHandler,
        )

        self.trt = trt
        self.tensorRTEngineCreator = tensorRTEngineCreator
        self.tensorRTEngineLoader = tensorRTEngineLoader
        self.tensorRTEngineNameHandler = tensorRTEngineNameHandler

        self.handleModels()

        self.newHeight, self.newWidth = calculateAspectRatio(
            self.width, self.height, self.depthQuality
        )
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
        if "og_" in self.depth_method:
            self.depth_method = self.depth_method.replace("og_", "")

        self.filename = modelsMap(
            model=self.depth_method, modelType="onnx", half=self.half
        )

        folderName = self.depth_method.replace("-tensorrt", "-onnx")
        self.modelPath = resolveWeightPath(
            folderName,
            self.filename,
            downloadModel=self.depth_method,
            half=self.half,
            modelType="onnx",
        )

        self.newHeight, self.newWidth = calculateAspectRatio(
            self.width, self.height, self.depthQuality
        )

        self.isVideoDepthTensorRT = "video_small_v2" in self.depth_method
        self.temporalWindowSize = 32

        inputShape = [1, 3, self.newHeight, self.newWidth]
        if self.isVideoDepthTensorRT:
            inputShape = [1, self.temporalWindowSize, 3, self.newHeight, self.newWidth]

        enginePath = self.tensorRTEngineNameHandler(
            modelPath=self.modelPath,
            fp16=self.half,
            optInputShape=inputShape,
        )

        if os.path.exists(enginePath) and os.path.getmtime(
            enginePath
        ) < os.path.getmtime(self.modelPath):
            try:
                os.remove(enginePath)
            except Exception:
                pass

        self.engine, self.context = self.tensorRTEngineLoader(enginePath)
        if (
            self.engine is None
            or self.context is None
            or not os.path.exists(enginePath)
        ):
            inputName = "image"
            if "distill" in self.depth_method or self.isVideoDepthTensorRT:
                inputName = "input"
            maxWorkspaceSize = (4 << 30) if self.isVideoDepthTensorRT else (1 << 30)
            self.engine, self.context = self.tensorRTEngineCreator(
                modelPath=self.modelPath,
                enginePath=enginePath,
                fp16=self.half,
                inputsMin=inputShape,
                inputsOpt=inputShape,
                inputsMax=inputShape,
                inputName=[inputName],
                maxWorkspaceSize=maxWorkspaceSize,
            )

        if self.engine is None or self.context is None:
            raise RuntimeError(
                f"Failed to initialize TensorRT engine for {self.depth_method} from {self.modelPath}"
            )

        self.stream = torch.cuda.Stream()
        inputTensorShape = (1, 3, self.newHeight, self.newWidth)
        outputTensorShape = (1, 1, self.newHeight, self.newWidth)
        if self.isVideoDepthTensorRT:
            inputTensorShape = (
                1,
                self.temporalWindowSize,
                3,
                self.newHeight,
                self.newWidth,
            )
            outputTensorShape = (
                1,
                self.temporalWindowSize,
                self.newHeight,
                self.newWidth,
            )

        self.dummyInput = torch.zeros(
            inputTensorShape,
            device=checker.device,
            dtype=torch.float16 if self.half else torch.float32,
        )

        self.dummyOutput = torch.zeros(
            outputTensorShape,
            device=checker.device,
            dtype=torch.float16 if self.half else torch.float32,
        )

        if "distill" in self.depth_method:
            self.dummyConst = torch.zeros(
                (1, 1, self.newHeight, self.newWidth),
                device=checker.device,
                dtype=torch.float16 if self.half else torch.float32,
            )

        self.bindings = [self.dummyInput.data_ptr(), self.dummyOutput.data_ptr()]

        if "distill" in self.depth_method:
            self.bindings.append(self.dummyConst.data_ptr())

        for i in range(self.engine.num_io_tensors):
            self.context.set_tensor_address(
                self.engine.get_tensor_name(i), self.bindings[i]
            )
            tensor_name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(tensor_name) == self.trt.TensorIOMode.INPUT:
                self.context.set_input_shape(tensor_name, self.dummyInput.shape)

        self.normStream = torch.cuda.Stream()
        self.outputNormStream = torch.cuda.Stream()
        self.cudaGraph = torch.cuda.CUDAGraph()
        self.initTorchCudaGraph()

    @torch.inference_mode()
    def initTorchCudaGraph(self):
        with torch.cuda.graph(self.cudaGraph, stream=self.stream):
            self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        self.stream.synchronize()

    @torch.inference_mode()
    def normFrame(self, frame):
        with torch.cuda.stream(self.normStream):
            frame = torch.from_numpy(frame).to(checker.device)
            frame = frame.permute(2, 0, 1).unsqueeze(0)
            if self.half:
                frame = frame.half()
            else:
                frame = frame.float()

            frame = frame.mul(1 / 255)
            frame = F.interpolate(
                frame.float(),
                (self.newHeight, self.newWidth),
                mode="bilinear",
                align_corners=True,
            )
            frame = (frame - MEANTENSOR) / STDTENSOR
            if self.half:
                frame = frame.half()
            if self.isVideoDepthTensorRT:
                frame = frame.squeeze(0)
                if not hasattr(self, "frameWindow"):
                    self.frameWindow = frame.unsqueeze(0).repeat(
                        self.temporalWindowSize, 1, 1, 1
                    )
                else:
                    self.frameWindow[:-1].copy_(self.frameWindow[1:])
                    self.frameWindow[-1].copy_(frame)
                self.dummyInput.copy_(self.frameWindow.unsqueeze(0), non_blocking=True)
            else:
                self.dummyInput.copy_(frame, non_blocking=True)
        self.normStream.synchronize()

    @torch.inference_mode()
    def normOutputFrame(self):
        if self.isVideoDepthTensorRT:
            depthTensor = self.dummyOutput[0, -1].float()
            depthTensor = torch.nan_to_num(depthTensor, nan=0.0, posinf=0.0, neginf=0.0)
            flatTensor = depthTensor.flatten()
            lowerBound = torch.quantile(flatTensor, 0.01)
            upperBound = torch.quantile(flatTensor, 0.99)
            denom = (upperBound - lowerBound).clamp_min(1e-6)
            depthTensor = ((depthTensor - lowerBound) / denom).clamp(0.0, 1.0)
            depth = (depthTensor * 255.0).byte().cpu().numpy()
        else:
            depth = self.dummyOutput.cpu().numpy()
            depth = np.reshape(depth, (self.newHeight, self.newWidth))
            if self.normalizer is not None:
                depth = self.normalizer.normalize(depth)
                depth = (depth * 255.0).astype(np.uint8)
            else:
                depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
                depth = depth.astype(np.uint8)

        return depth

    @torch.inference_mode()
    def processFrame(self, frame):
        try:
            self.normFrame(frame)
            with torch.cuda.stream(self.stream):
                self.cudaGraph.replay()
            self.stream.synchronize()
            depth = self.normOutputFrame()
            depthTensor = (
                torch.from_numpy(depth)
                .to(checker.device, dtype=torch.float32)
                .unsqueeze(0)
                .unsqueeze(0)
                .mul(1 / 255)
            )
            self.writeBuffer.write(depthTensor)
        except Exception as e:
            logging.exception(f"Something went wrong while processing the frame, {e}")

    def process(self):
        frameCount = 0
        with ProgressBarLogic(self.totalFrames) as bar:
            currentFrame = self.readBuffer.read()
            nextFrame = self.readBuffer.read() if currentFrame is not None else None
            while currentFrame is not None:
                self.processFrame(currentFrame)
                frameCount += 1
                bar(1)
                currentFrame = nextFrame
                if currentFrame is not None:
                    nextFrame = self.readBuffer.read()

        logging.info(f"Processed {frameCount} frames")
        self.writeBuffer.close()
