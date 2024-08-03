import torch
import os
import logging

from torch.nn import functional as F
from .downloadModels import downloadModels, weightsDir, modelsMap

torch.set_float32_matmul_precision("medium")


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
            "scenechange",
            half=self.half,
        )

        if not os.path.exists(os.path.join(weightsDir, "scenechange", filename)):
            modelPath = downloadModels(
                "scenechange",
                half=self.half,
            )

        else:
            modelPath = os.path.join(weightsDir, "scenechange", filename)

        logging.info(f"Loading scenechange detection model from {modelPath}")

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
        frame = frame.astype(self.np.float16) if self.half else frame.astype(self.np.float32)
        frame = frame / 255.0
        frame = frame.transpose((2, 0, 1))
        return frame
    
    @torch.inference_mode()
    def run(self, frame):
        if self.I0 is None:
            self.I0 = self.processFrame(frame)
            return False
        
        self.I1 = self.processFrame(frame)
        inputs = self.np.concatenate(
            (self.I0, self.I1), 0
        )

        self.I0 = self.I1
        return (
            self.model.run(None, {"input": inputs})[0][0][0] > self.sceneChangeThreshold
        )

class SceneChangeTensorRT:
    def __init__(self, half, sceneChangeThreshold=0.85):
        self.half = half
        self.sceneChangeThreshold = sceneChangeThreshold

        import tensorrt as trt
        from .utils.trtHandler import TensorRTEngineCreator, TensorRTEngineLoader, TensorRTEngineNameHandler

        self.trt = trt
        self.TensorRTEngineCreator = TensorRTEngineCreator
        self.TensorRTEngineLoader = TensorRTEngineLoader
        self.TensorRTEngineNameHandler = TensorRTEngineNameHandler

        self.handleModel()

    def handleModel(self):
        filename = modelsMap(
            "scenechange",
            half=self.half,
        )

        if not os.path.exists(os.path.join(weightsDir, "scenechange", filename)):
            self.modelPath = downloadModels(
                "scenechange",
                half=self.half,
            )

        else:
            self.modelPath = os.path.join(weightsDir, "scenechange", filename)

        enginePath = self.TensorRTEngineNameHandler(
            modelPath=self.modelPath, fp16=self.half, optInputShape=[0, 6, 224, 224]
        )

        if not os.path.exists(enginePath):
            self.engine, self.context = self.TensorRTEngineCreator(
                modelPath=self.modelPath,
                enginePath=enginePath,
                fp16=self.half,
                inputsMin=[6, 224, 224],
                inputsOpt=[6, 224, 224],
                inputsMax=[6, 224, 224],
            )
        else:
            self.engine, self.context = self.TensorRTEngineLoader(enginePath)

        self.dType = torch.float16 if self.half else torch.float32
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stream = torch.cuda.Stream()
        self.dummyInput = torch.zeros(
            (6, 224, 224),
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
        self.I1 = torch.zeros((3, 224, 224), device=self.device, dtype=self.dType)

    @torch.inference_mode()
    def processFrame(self, frame):
        frame = frame.to(self.device, non_blocking=True, dtype=self.dType).permute(2, 0, 1).unsqueeze(0).mul(1.0 / 255.0)
        frame = F.interpolate(frame, size=(224, 224), mode="bilinear")
        return frame.contiguous().squeeze(0)
    
    @torch.inference_mode()
    def run(self, frame):
        with torch.cuda.stream(self.stream):
            if self.I0 is None:
                self.I0 = self.processFrame(frame)
                return False
            
            self.I1 = self.processFrame(frame)

            self.dummyInput.copy_(
                torch.cat([self.I0, self.I1], dim=0),
                non_blocking=True,
            )

            self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
            self.I0.copy_(self.I1, non_blocking=True)
            self.stream.synchronize()

            return self.dummyOutput[0][0].item() > self.sceneChangeThreshold
