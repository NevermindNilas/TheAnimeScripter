import torch
import os
import logging
import cv2

from torch.nn import functional as F
from src.utils.downloadModels import downloadModels, weightsDir, modelsMap


class SceneChange:
    def __init__(
        self,
        half,
        sceneChangeThreshold,
    ):
        self.half = half

        import onnxruntime as ort
        import numpy as np

        self.ort = ort
        self.np = np
        self.sceneChangeThreshold = sceneChangeThreshold

        self.loadModel()

    def loadModel(self):
        filename = modelsMap(
            "maxxvit-directml",
            half=self.half,
        )

        if not os.path.exists(os.path.join(weightsDir, "maxxvit-directml", filename)):
            modelPath = downloadModels(
                "maxxvit-directml",
                half=self.half,
            )

        else:
            modelPath = os.path.join(weightsDir, "maxxvit-directml", filename)

        providers = self.ort.get_available_providers()
        if "DmlExecutionProvider" in providers:
            logging.info(
                "DirectML provider available for scenechange detection. Defaulting to DirectML"
            )
            self.model = self.ort.InferenceSession(
                modelPath, providers=["DmlExecutionProvider"]
            )
        else:
            logging.info(
                "DirectML provider not available for scenechange detection, falling back to CPU, expect significantly worse performance, ensure that your drivers are up to date and your GPU supports DirectX 12"
            )
            self.model = self.ort.InferenceSession(
                modelPath, providers=["CPUExecutionProvider"]
            )

        self.I0 = None
        self.I1 = None

    @torch.inference_mode()
    def processFrame(self, frame):
        frame = frame.cpu().numpy()
        frame = self.np.resize(frame, (224, 224, 3))
        frame = (
            frame.astype(self.np.float16)
            if self.half
            else frame.astype(self.np.float32)
        )
        frame = frame.transpose((2, 0, 1))
        return frame

    @torch.inference_mode()
    def __call__(self, frame):
        if self.I0 is None:
            self.I0 = self.processFrame(frame)
            return False

        self.I1 = self.processFrame(frame)
        inputs = self.np.concatenate((self.I0, self.I1), 0)

        self.I0 = self.I1

        result = self.model.run(None, {"input": inputs})[0][0][0] * 255

        return result > self.sceneChangeThreshold


class SceneChangeTensorRT:
    def __init__(self, half, sceneChangeThreshold=0.85, sceneChangeMethod=""):
        self.half = half
        self.sceneChangeThreshold = sceneChangeThreshold
        self.sceneChangeMethod = sceneChangeMethod

        import tensorrt as trt
        from .utils.trtHandler import (
            tensorRTEngineCreator,
            tensorRTEngineLoader,
            tensorRTEngineNameHandler,
        )

        self.trt = trt
        self.tensorRTEngineCreator = tensorRTEngineCreator
        self.tensorRTEngineLoader = tensorRTEngineLoader
        self.tensorRTEngineNameHandler = tensorRTEngineNameHandler

        self.handleModel()

    def handleModel(self):
        filename = modelsMap(
            self.sceneChangeMethod,
            half=self.half,
            modelType="onnx",
        )

        folderName = self.sceneChangeMethod.replace("-tensorrt", "-onnx")
        if not os.path.exists(os.path.join(weightsDir, folderName, filename)):
            self.modelPath = downloadModels(
                model=self.sceneChangeMethod,
                half=self.half,
                modelType="onnx",
            )

        else:
            self.modelPath = os.path.join(weightsDir, folderName, filename)

        if self.sceneChangeMethod == "maxxvit-tensorrt":
            self.height = self.width = 224
        elif self.sceneChangeMethod == "shift_lpips-tensorrt":
            self.height = self.width = 256

        enginePath = self.tensorRTEngineNameHandler(
            modelPath=self.modelPath,
            fp16=self.half,
            optInputShape=[0, 6, self.height, self.width],
        )

        self.engine, self.context = self.tensorRTEngineLoader(enginePath)
        if (
            self.engine is None
            or self.context is None
            or not os.path.exists(enginePath)
        ):
            self.engine, self.context = self.tensorRTEngineCreator(
                modelPath=self.modelPath,
                enginePath=enginePath,
                fp16=self.half,
                inputsMin=[6, self.height, self.width],
                inputsOpt=[6, self.height, self.width],
                inputsMax=[6, self.height, self.width],
            )

        self.dType = torch.float16 if self.half else torch.float32
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stream = torch.cuda.Stream()
        self.dummyInput = torch.zeros(
            (6, self.height, self.width),
            device=self.device,
            dtype=self.dType,
        )
        self.dummyOutput = torch.zeros(
            (1, 2),
            device=self.device,
            dtype=self.dType,
        )

        self.bindings = [self.dummyInput.data_ptr(), self.dummyOutput.data_ptr()]

        for i in range(self.engine.num_io_tensors):
            self.context.set_tensor_address(
                self.engine.get_tensor_name(i), self.bindings[i]
            )
            tensor_name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(tensor_name) == self.trt.TensorIOMode.INPUT:
                self.context.set_input_shape(tensor_name, self.dummyInput.shape)

        with torch.cuda.stream(self.stream):
            for _ in range(10):
                self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
                self.stream.synchronize()

        self.I0 = None
        self.I1 = torch.zeros(
            (3, self.height, self.width), device=self.device, dtype=self.dType
        )

        self.normStream = torch.cuda.Stream()
        self.cudaGraph = torch.cuda.CUDAGraph()
        self.initTorchCudaGraph()

    @torch.inference_mode()
    def initTorchCudaGraph(self):
        with torch.cuda.graph(self.cudaGraph, stream=self.stream):
            self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        self.stream.synchronize()

    @torch.inference_mode()
    def processFrame(self, frame):
        return F.interpolate(
            frame.to(dtype=self.dType),
            size=(self.height, self.width),
            mode="bilinear",
        ).squeeze(0)

    @torch.inference_mode()
    def __call__(self, frame):
        with torch.cuda.stream(self.stream):
            if self.I0 is None:
                self.I0 = self.processFrame(frame)
                return False

            self.I1 = self.processFrame(frame)

            self.dummyInput.copy_(
                torch.cat([self.I0, self.I1], dim=0),
                non_blocking=True,
            )

            self.cudaGraph.replay()
            self.I0.copy_(self.I1, non_blocking=True)
            result = self.dummyOutput[0][0].item()
        self.stream.synchronize()
        return result > self.sceneChangeThreshold


class SceneChangeCPU:
    def __init__(self, sceneChangeThreshold):
        import numpy as np

        self.np = np
        self.sceneChangeThreshold = sceneChangeThreshold
        self.I0 = None

    def processFrame(self, frame):
        frame = cv2.resize(frame.cpu().numpy(), (224, 224))
        frame = cv2.cvtColor(frame, self.cv2.COLOR_BGR2GRAY)
        frame = frame.astype(self.np.float16)
        return frame

    def __call__(self, frame):
        if self.I0 is None:
            self.I0 = self.processFrame(frame)
            return False

        self.I1 = self.processFrame(frame)
        result = self.np.clip((self.np.mean((self.I0 - self.I1) ** 2) * 10), 0, 1)

        self.I0 = self.I1
        return result > self.sceneChangeThreshold


class SceneChangeCuda:
    def __init__(self, sceneChangeThreshold):
        self.sceneChangeThreshold = sceneChangeThreshold
        self.I0 = None
        self.stream = torch.cuda.Stream()
        self.device = torch.device("cuda")

    def processFrame(self, frame):
        with torch.cuda.stream(self.stream):
            frame = frame.to(self.device, non_blocking=True)
            frame = F.interpolate(
                frame.unsqueeze(0).float(),
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            )
            frame = torch.mean(frame, dim=0, keepdim=True)
            frame = frame.half()
        return frame

    def __call__(self, frame):
        with torch.cuda.stream(self.stream):
            if self.I0 is None:
                self.I0 = self.processFrame(frame)
                return False

            self.I1 = self.processFrame(frame)
            mse = (torch.mean((self.I0 - self.I1) ** 2) * 10).clamp(0, 1)

            self.I0.copy_(self.I1, non_blocking=True)
            torch.cuda.synchronize()
            return mse.item() > self.sceneChangeThreshold


class DifferentialTensorRT:
    def __init__(
        self,
        scenechangeThreshold,
        height=1080,
        width=1920,
    ):
        import tensorrt as trt
        from .utils.trtHandler import (
            tensorRTEngineCreator,
            tensorRTEngineLoader,
            tensorRTEngineNameHandler,
        )

        self.trt = trt
        self.tensorRTEngineCreator = tensorRTEngineCreator
        self.tensorRTEngineLoader = tensorRTEngineLoader
        self.tensorRTEngineNameHandler = tensorRTEngineNameHandler

        self.scenechangeThreshold = scenechangeThreshold
        self.height = height
        self.width = width

        self.handleModel()

    def handleModel(self):
        filename = modelsMap(
            "differential-tensorrt",
            half=True,
        )

        if not os.path.exists(
            os.path.join(weightsDir, "differential-tensorrt", filename)
        ):
            self.modelPath = downloadModels(
                "differential-tensorrt",
                half=True,
            )
        else:
            self.modelPath = os.path.join(weightsDir, "differential-tensorrt", filename)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stream = torch.cuda.Stream()

        inputsMin = [1, 6, self.height, self.width]
        inputsOpt = [1, 6, self.height, self.width]
        inputsMax = [1, 6, self.height, self.width]

        enginePath = self.tensorRTEngineNameHandler(
            modelPath=self.modelPath,
            fp16=True,
            optInputShape=inputsOpt,
        )

        self.engine, self.context = self.tensorRTEngineLoader(enginePath)
        if (
            self.engine is None
            or self.context is None
            or not os.path.exists(enginePath)
        ):
            self.engine, self.context = self.tensorRTEngineCreator(
                modelPath=self.modelPath,
                enginePath=enginePath,
                fp16=True,
                inputsMin=inputsMin,
                inputsOpt=inputsOpt,
                inputsMax=inputsMax,
            )

        self.dType = torch.float16

        self.dummyInput = torch.zeros(
            (1, 6, self.height, self.width),
            device=self.device,
            dtype=self.dType,
        )

        self.dummyOutput = torch.zeros(
            (1,),
            device=self.device,
            dtype=self.dType,
        )

        self.bindings = [self.dummyInput.data_ptr(), self.dummyOutput.data_ptr()]

        for i in range(self.engine.num_io_tensors):
            self.context.set_tensor_address(
                self.engine.get_tensor_name(i), self.bindings[i]
            )
            tensor_name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(tensor_name) == self.trt.TensorIOMode.INPUT:
                self.context.set_input_shape(tensor_name, self.dummyInput.shape)

        with torch.cuda.stream(self.stream):
            for _ in range(10):
                self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
                self.stream.synchronize()

        self.I0 = None
        self.I1 = torch.zeros(
            (1, 6, self.height, self.width), device=self.device, dtype=self.dType
        )

    @torch.inference_mode()
    def processFrame(self, frame):
        return (
            frame.to(self.device, non_blocking=True, dtype=self.dType)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .contiguous()
        )

    @torch.inference_mode()
    def __call__(self, frame):
        with torch.cuda.stream(self.stream):
            if self.I0 is None:
                self.I0 = self.processFrame(frame)
                return False

            self.I1 = self.processFrame(frame)

            self.dummyInput.copy_(
                torch.cat([self.I0, self.I1], dim=1),
                non_blocking=True,
            )

            self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
            self.I0.copy_(self.I1, non_blocking=True)
            self.stream.synchronize()

            return self.dummyOutput.item() > self.scenechangeThreshold
