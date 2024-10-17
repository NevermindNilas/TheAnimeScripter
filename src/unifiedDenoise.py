import os
import torch
import logging

from spandrel import ModelLoader
from .utils.downloadModels import downloadModels, weightsDir, modelsMap


class UnifiedDenoise:
    def __init__(
        self,
        model: str = "scunet",
        half: bool = True,
    ):
        """
        Initialize the denoiser with the desired model

        Args:
            model (str): The model to use for denoising
            width (int): The width of the input frame
            height (int): The height of the input frame
            half (bool): Whether to use half precision
            customModel (str): The path to a custom model file
        """

        self.model = model
        self.half = half
        self.handleModel()

    def handleModel(self):
        """
        Load the Model
        """
        if self.model in ["nafnet"]:
            self.half = False
            print("NAFNet does not support half precision, using float32 instead")

        self.filename = modelsMap(self.model)
        if not os.path.exists(os.path.join(weightsDir, self.model, self.filename)):
            modelPath = downloadModels(model=self.model)
        else:
            modelPath = os.path.join(weightsDir, self.model, self.filename)

        try:
            self.model = ModelLoader().load_from_file(path=modelPath)

        except Exception as e:
            logging.error(f"Error loading model: {e}")

        self.isCudaAvailable = torch.cuda.is_available()
        self.model = (
            self.model.eval().cuda() if self.isCudaAvailable else self.model.eval()
        )

        if self.half:
            self.model.model.half()
            self.dType = torch.float16
        else:
            self.dType = torch.float32

        self.device = torch.device("cuda" if self.isCudaAvailable else "cpu")
        self.stream = torch.cuda.Stream()

        self.model.model.to(memory_format=torch.channels_last)

    @torch.inference_mode()
    def __call__(self, frame: torch.tensor) -> torch.tensor:
        with torch.cuda.stream(self.stream):
            frame = (
                frame.to(self.device, non_blocking=True, dtype=self.dType)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(memory_format=torch.channels_last)
                .mul(1 / 255)
            )

            frame = self.model(frame).squeeze_(0).permute(1, 2, 0).mul(255)
            self.stream.synchronize()
            return frame
