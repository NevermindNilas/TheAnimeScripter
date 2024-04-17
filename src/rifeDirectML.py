import os
import torch
import numpy as np
import logging
import onnxruntime as ort

from .downloadModels import downloadModels, weightsDir, modelsMap

class RifeDirectML:
    def __init__(
            self,
            interpolateMethod: str = "rife415",
            half = True,
            ensemble: bool = False,
            nt: int = 1,
    ):
        """
        Interpolates frames using DirectML
        
        Args:
            interpolateMethod (str, optional): Interpolation method. Defaults to "rife415".
            half (bool, optional): Half resolution. Defaults to True.
            ensemble (bool, optional): Ensemble. Defaults to False.
            nt (int, optional): Number of threads. Defaults to 1.
        """
        self.interpolateMethod = interpolateMethod
        self.half = half
        self.ensemble = ensemble
        self.nt = nt
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.modelPath = os.path.join(weightsDir, modelsMap[self.interpolateMethod])
        
        self.handleModel()

    def handleModel(self):
        """
        Load the model
        """
        
        self.filename = modelsMap(
            model=self.interpolateMethod, modelType="onnx"
        )        

        if not os.path.exists(self.modelPath):
            os.path.join(weightsDir, self.interpolateMethod, self.filename)
        
            modelPath = downloadModels(
                model=self.interpolateMethod,
                modelType="onnx",
                half=self.half,
            )
        else:
            modelPath = os.path.join(weightsDir, self.interpolateMethod, self.filename)

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

        
        if self.half:
            self.numpyDType = np.float16
            self.torchDType = torch.float16
        else:
            self.numpyDType = np.float32
            self.torchDType = torch.float32
        
