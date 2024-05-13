import os
import torch
import logging
import numpy as np

from spandrel import ModelLoader, ImageModelDescriptor
from .downloadModels import downloadModels, weightsDir, modelsMap

# Apparently this can improve performance slightly
torch.set_float32_matmul_precision("medium")


class UnifiedDenoise:
    def __init__(
        self,
        model: str = "scunet",
        width: int = 1920,
        height: int = 1080,
        half: bool = False,
        customModel: str = None,
        nt: int = 1,
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
        self.width = width
        self.height = height
        self.half = half
        self.customModel = customModel
        self.nt = nt

        self.handleModel()

    def handleModel(self):
        """
        Load the Model
        """
        if not self.customModel:
            if self.model == "span":
                # This is so that span ( for upscaling ) doesn't overlap with span-denoise,
                # Hackintoshy solution until I think of a better way
                self.model = "span-denoise"

            if self.half:
                match self.model:
                    case "dpir" | "nafnet":
                        self.precision = "bfloat16"
                    case "span-denoise" | "scunet":
                        self.precision = "fp16"
            else:
                self.precision = "fp32"

            self.filename = modelsMap(self.model)

            if not os.path.exists(os.path.join(weightsDir, self.model, self.filename)):
                modelPath = downloadModels(model=self.model)

            else:
                modelPath = os.path.join(weightsDir, self.model, self.filename)

        else:
            if os.path.isfile(self.customModel) and self.customModel.endswith(".pth"):
                modelPath = self.customModel
            else:
                raise FileNotFoundError(f"Model file {self.customModel} not found or not a .pth file")

        try:
            self.model = ModelLoader().load_from_file(path=modelPath)

        except Exception as e:
            logging.error(f"Error loading model: {e}")

        if self.customModel:
            assert isinstance(self.model, ImageModelDescriptor)

        self.isCudaAvailable = torch.cuda.is_available()
        self.model = (
            self.model.eval().cuda() if self.isCudaAvailable else self.model.eval()
        )

        self.device = torch.device("cuda" if self.isCudaAvailable else "cpu")

        if self.isCudaAvailable:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            if self.half:
                if self.precision == "fp16":
                    torch.set_default_dtype(torch.float16)
                    self.model.half()
                elif self.precision == "bfloat16":
                    torch.set_default_dtype(torch.bfloat16)
                    self.model.bfloat16()

    @torch.inference_mode()
    def run(self, frame: np.ndarray) -> np.ndarray:
        """
        Upscale a frame using a desired model, and return the upscaled frame
        Expects a numpy array of shape (height, width, 3) and dtype uint8
        """
        if self.isCudaAvailable:
            if self.precision == "bfloat16":
                frame = frame.bfloat16()

        return self.model(frame)

