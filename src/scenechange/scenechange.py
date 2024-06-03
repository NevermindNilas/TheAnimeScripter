import torch
import onnxruntime as ort
import os
import logging
import numpy as np

from torch.nn import functional as F
from src.downloadModels import weightsDir, downloadModels, modelsMap
from src.coloredPrints import yellow


class SceneChange:
    def __init__(
        self,
        half,
    ):
        self.half = half
        self.loadModel()

    def loadModel(self):
        filename = modelsMap(
            "scenechange",
            self.half,
        )

        if not os.path.exists(os.path.join(weightsDir, "scenechange", filename)):
            modelPath = downloadModels(
                "scenechange",
                self.half,
            )

        else:
            modelPath = os.path.join(weightsDir, "scenechange", filename)

        providers = ort.get_available_providers()
        if "DmlExecutionProvider" in providers:
            logging.info("DirectML provider available. Defaulting to DirectML")
            self.model = ort.InferenceSession(
                modelPath, providers=["DmlExecutionProvider"]
            )
        else:
            logging.info(
                "DirectML provider not available, falling back to CPU, expect significantly worse performance, ensure that your drivers are up to date and your GPU supports DirectX 12"
            )
            self.model = ort.InferenceSession(
                modelPath, providers=["CPUExecutionProvider"]
            )

        self.deviceType = "cpu"
        self.device = torch.device(self.deviceType)
        self.firstRun = True

    @torch.inference_mode()
    def processFrame(self, frame):
        frame = frame.to(self.device)
        frame = frame.unsqueeze(0)
        frame = frame.permute(0, 3, 1, 2)
        frame = F.interpolate(frame, size=(224, 224), mode="bilinear")
        frame = frame
        frame = frame.to(self.device).squeeze(0)
        frame = frame.half() if self.half else frame.float()
        return frame.numpy()

    @torch.inference_mode()
    def run(self, frame):
        if self.firstRun:
            self.I0 = self.processFrame(frame)
            self.firstRun = False

        self.I1 = self.processFrame(frame)

        inputs = np.ascontiguousarray(np.concatenate((self.I0, self.I1), 0))
        result = self.model.run(None, {"input": inputs})[0][0][0] * 255

        print(yellow(f"SceneChange: {result}"))

        self.I0 = self.I1

        return result > 0.93  # hardcoded for testing purposes
