import os

os.environ.setdefault("DA3_LOG_LEVEL", "ERROR")

import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F

from src.constants import ADOBE
from src.depth.backends._shared import (
    MEANTENSOR,
    STDTENSOR,
    SlidingWindowNormalizer,
    VideoRangeNormalizer,
    calculateAspectRatio,
)
from src.infra.isCudaInit import CudaChecker
from src.infra.progressBarLogic import ProgressBarLogic
from src.io.ffmpegSettings import (
    BuildBuffer,
    WriteBuffer,
)
from src.model.download import resolveWeightPath
from src.model.registry import modelsMap

if ADOBE:
    from src.server.aeComms import progressState

checker = CudaChecker()


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
        self.custom_encoder = custom_encoder
        self.benchmark = benchmark
        self.totalFrames = totalFrames
        self.bitDepth = bitDepth
        self.depthQuality = depthQuality
        self.normalizer = SlidingWindowNormalizer() if depthNorm else None
        self._reqBatch = max(1, int(depth_batch))

        from src.model.trtHandler import (
            tensorRTEngineCreator,
            tensorRTEngineLoader,
            tensorRTEngineNameHandler,
        )
        from src.utils.tensorrt_import import trt

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

        # distill is multi-input (needs a paired const), so batching stays at 1;
        # the single-input image engines batch the frame axis.
        B = 1 if "distill" in self.depth_method else self._reqBatch
        self._batch = B

        enginePath = self.tensorRTEngineNameHandler(
            modelPath=self.modelPath,
            fp16=self.half,
            optInputShape=[B, 3, self.newHeight, self.newWidth],
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
                inputsOpt=[B, 3, self.newHeight, self.newWidth],
                inputsMax=[B, 3, self.newHeight, self.newWidth],
                inputName=[inputName],
            )

        self.stream = torch.cuda.Stream()
        self.dummyInput = torch.zeros(
            (B, 3, self.newHeight, self.newWidth),
            device=checker.device,
            dtype=torch.float16 if self.half else torch.float32,
        )

        self.dummyOutput = torch.zeros(
            (B, 1, self.newHeight, self.newWidth),
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
        # A dynamic-batch engine lazily allocates device memory on its first
        # execute; that cudaMalloc must happen BEFORE graph capture or the
        # capture is invalidated. Warm up once, then capture.
        with torch.cuda.stream(self.stream):
            self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        self.stream.synchronize()
        with torch.cuda.graph(self.cudaGraph, stream=self.stream):
            self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        self.stream.synchronize()

    @torch.inference_mode()
    def normFrame(self, frames):
        # frames: list of B decoded tensors [1, 3, H, W]; build the [B,3,H,W] input
        with torch.cuda.stream(self.normStream):
            batch = frames[0] if len(frames) == 1 else torch.cat(frames, dim=0)
            batch = F.interpolate(
                batch.float(),
                (self.newHeight, self.newWidth),
                mode="bilinear",
                align_corners=True,
            )
            batch = (batch - MEANTENSOR) / STDTENSOR
            if self.half:
                batch = batch.half()
            self.dummyInput.copy_(batch, non_blocking=True)
        self.normStream.synchronize()

    @torch.inference_mode()
    def normOutputFrame(self, i):
        # per-frame min-max (or normalizer) on slice i so a batched forward is
        # bit-equivalent to one-frame-at-a-time
        with torch.cuda.stream(self.outputNormStream):
            depth = F.interpolate(
                self.dummyOutput[i : i + 1],
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
    def processBatch(self, frames):
        try:
            real = len(frames)
            # a static-shape CUDA graph runs exactly B frames; pad a short final
            # batch with the last frame and drop the padded outputs
            if real < self._batch:
                frames = frames + [frames[-1]] * (self._batch - real)
            self.normFrame(frames)
            with torch.cuda.stream(self.stream):
                self.cudaGraph.replay()
            self.stream.synchronize()
            for i in range(real):
                self.writeBuffer.write(self.normOutputFrame(i))
        except Exception as e:
            logging.exception(f"Something went wrong while processing the frame, {e}")

    def process(self):
        frameCount = 0
        with ProgressBarLogic(self.totalFrames) as bar:
            while True:
                frames = []
                for _ in range(self._batch):
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
                if len(frames) < self._batch:
                    break

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
        self.custom_encoder = custom_encoder
        self.benchmark = benchmark
        self.totalFrames = totalFrames
        self.bitDepth = bitDepth
        self.depthQuality = depthQuality
        self.normalizer = SlidingWindowNormalizer() if depthNorm else None
        self._reqBatch = max(1, int(depth_batch))

        from src.model.trtHandler import (
            tensorRTEngineCreator,
            tensorRTEngineLoader,
            tensorRTEngineNameHandler,
        )
        from src.utils.tensorrt_import import trt

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
        if self.isVideoDepthTensorRT and self.normalizer is not None:
            self.normalizer = VideoRangeNormalizer()
        self.temporalWindowSize = 32

        # distill (multi-input) and the temporal video engine can't batch the
        # frame axis; only the single-input image engines do.
        self._batch = (
            1
            if (self.isVideoDepthTensorRT or "distill" in self.depth_method)
            else self._reqBatch
        )
        B = self._batch

        inputShape = [B, 3, self.newHeight, self.newWidth]
        inputsMinShape = [1, 3, self.newHeight, self.newWidth]
        if self.isVideoDepthTensorRT:
            inputShape = [1, self.temporalWindowSize, 3, self.newHeight, self.newWidth]
            inputsMinShape = inputShape

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
                inputsMin=inputsMinShape,
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
        inputTensorShape = (B, 3, self.newHeight, self.newWidth)
        outputTensorShape = (B, 1, self.newHeight, self.newWidth)
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
        # warm up once so a dynamic-batch engine's lazy device-memory alloc
        # happens before capture (a cudaMalloc during capture invalidates it)
        with torch.cuda.stream(self.stream):
            self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        self.stream.synchronize()
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
                    # shift the window left by one; clone the source slice first
                    # because frameWindow[:-1] and frameWindow[1:] alias overlapping
                    # memory (copy_ rejects a self-overlapping src/dst).
                    self.frameWindow[:-1].copy_(self.frameWindow[1:].clone())
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
            if self.normalizer is not None:
                depthTensor = self.normalizer.normalize(depthTensor)
            else:
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

    @torch.inference_mode()
    def normFrameBatch(self, frames):
        # frames: list of B decoded numpy HWC frames -> [B,3,newH,newW] input.
        # image path only (video/distill run one-at-a-time via processFrame).
        with torch.cuda.stream(self.normStream):
            tensors = []
            for f in frames:
                t = torch.from_numpy(f).to(checker.device).permute(2, 0, 1).unsqueeze(0)
                t = (t.half() if self.half else t.float()).mul(1 / 255)
                tensors.append(t)
            batch = tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=0)
            batch = F.interpolate(
                batch.float(),
                (self.newHeight, self.newWidth),
                mode="bilinear",
                align_corners=True,
            )
            batch = (batch - MEANTENSOR) / STDTENSOR
            if self.half:
                batch = batch.half()
            self.dummyInput.copy_(batch, non_blocking=True)
        self.normStream.synchronize()

    @torch.inference_mode()
    def normOutputFrameAt(self, i):
        depth = self.dummyOutput[i].cpu().numpy()
        depth = np.reshape(depth, (self.newHeight, self.newWidth))
        if self.normalizer is not None:
            depth = self.normalizer.normalize(depth)
            depth = (depth * 255.0).astype(np.uint8)
        else:
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth.astype(np.uint8)
        return depth

    @torch.inference_mode()
    def processBatch(self, frames):
        try:
            real = len(frames)
            if real < self._batch:
                frames = frames + [frames[-1]] * (self._batch - real)
            self.normFrameBatch(frames)
            with torch.cuda.stream(self.stream):
                self.cudaGraph.replay()
            self.stream.synchronize()
            for i in range(real):
                depth = self.normOutputFrameAt(i)
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
            while True:
                frames = []
                for _ in range(self._batch):
                    frame = self.readBuffer.read()
                    if frame is None:
                        break
                    frames.append(frame)
                if frames:
                    # B==1 keeps the exact single-frame path (handles video/distill);
                    # B>1 is the batched single-input image path
                    if self._batch == 1:
                        self.processFrame(frames[0])
                    else:
                        self.processBatch(frames)
                    frameCount += len(frames)
                    bar(len(frames))
                # A short read means the decoder's single None sentinel was just
                # consumed; another read() would block forever.
                if len(frames) < self._batch:
                    break

        logging.info(f"Processed {frameCount} frames")
        self.writeBuffer.close()
