import torch
import os
import logging

from torch.nn import functional as F
from .downloadModels import downloadModels, weightsDir, modelsMap
from .coloredPrints import yellow

torch.set_float32_matmul_precision("medium")

class SceneChange:
    def __init__(
        self,
        half,
        sceneChangeThreshold,
    ):
        self.half = half

        import onnxruntime as ort
        import cv2
        import numpy as np

        self.ort = ort
        self.cv2 = cv2
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
        frame = self.cv2.resize(frame, (224, 224))
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

        from polygraphy.backend.trt import (
            TrtRunner,
            engine_from_network,
            network_from_onnx_path,
            CreateConfig,
            Profile,
            EngineFromBytes,
            SaveEngine,
        )
        from polygraphy.backend.common import BytesFromPath

        import tensorrt as trt


        self.TrtRunner = TrtRunner
        self.engine_from_network = engine_from_network
        self.network_from_onnx_path = network_from_onnx_path
        self.CreateConfig = CreateConfig
        self.Profile = Profile
        self.EngineFromBytes = EngineFromBytes
        self.SaveEngine = SaveEngine
        self.BytesFromPath = BytesFromPath

        self.trt = trt

        self.handleModel()

    def handleModel(self):
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

        if self.half:
            trtEngineModelPath = modelPath.replace(".onnx", "_fp16.engine")
        else:
            trtEngineModelPath = modelPath.replace(".onnx", "_fp32.engine")

        if not os.path.exists(trtEngineModelPath):
            toPrint = f"Engine not found, creating dynamic engine for model: {modelPath}, this may take a while, but it is worth the wait..."
            print(yellow(toPrint))
            logging.info(toPrint)

            profile = [
                self.Profile().add(
                    "input",
                    min=(6, 224, 224),
                    opt=(6, 224, 224),
                    max=(6, 224, 224),
                )
            ]

            self.config = self.CreateConfig(
                fp16=self.half,
                profiles=profile,
                preview_features=[],
            )

            self.engine = self.engine_from_network(
                self.network_from_onnx_path(modelPath),
                config=self.config,
            )
            self.engine = self.SaveEngine(self.engine, trtEngineModelPath)
            self.engine.__call__()

        with open(trtEngineModelPath, "rb") as f, self.trt.Runtime(
            self.trt.Logger(self.trt.Logger.INFO)
        ) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()

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
