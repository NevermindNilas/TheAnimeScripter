import os
import torch
import numpy as np
import logging

# will be on wait for the next release of spandrel
from spandrel import ImageModelDescriptor, ModelLoader
from .downloadModels import downloadModels, weightsDir, modelsMap

# Apparently this can improve performance slightly
torch.set_float32_matmul_precision("medium")

class Upscaler:
    def __init__(
        self,
        upscaleMethod: str = "shufflecugan",
        upscaleFactor: int = 2,
        cuganKind: str = "conservative",
        half: bool = False,
        width: int = 1920,
        height: int = 1080,
        customModel: str = None,
        nt: int = 1,
    ):
        """
        Initialize the upscaler with the desired model

        Args:
            upscaleMethod (str): The method to use for upscaling
            upscaleFactor (int): The factor to upscale by
            cuganKind (str): The kind of cugan to use
            half (bool): Whether to use half precision
            width (int): The width of the input frame
            height (int): The height of the input frame
            customModel (str): The path to a custom model file
            nt (int): The number of threads to use
            trt (bool): Whether to use tensorRT
        """
        self.upscaleMethod = upscaleMethod
        self.upscaleFactor = upscaleFactor
        self.cuganKind = cuganKind
        self.half = half
        self.width = width
        self.height = height
        self.customModel = customModel
        self.nt = nt

        self.handleModel()

    def handleModel(self):
        """
        Load the desired model
        """

        if not self.customModel:
            self.filename = modelsMap(self.upscaleMethod, self.upscaleFactor, self.cuganKind)
            if not os.path.exists(
                os.path.join(weightsDir, self.upscaleMethod, self.filename)
            ):
                modelPath = downloadModels(
                    model=self.upscaleMethod,
                    cuganKind=self.cuganKind,
                    upscaleFactor=self.upscaleFactor,
                )

            else:
                modelPath = os.path.join(weightsDir, self.upscaleMethod, self.filename)

        else:
            if os.path.isfile(self.customModel):
                modelPath = self.customModel

            else:
                raise FileNotFoundError(
                    f"Custom model file {self.customModel} not found"
                )

        try:
            self.model = ModelLoader().load_from_file(modelPath)
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
            self.stream = [torch.cuda.Stream() for _ in range(self.nt)]
            self.currentStream = 0
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            if self.half:
                torch.set_default_dtype(torch.float16)
                self.model.half()


    @torch.inference_mode()
    def run(self, frame: np.ndarray) -> np.ndarray:
        """
        Upscale a frame using a desired model, and return the upscaled frame
        Expects a numpy array of shape (height, width, 3) and dtype uint8
        """
        with torch.no_grad():
            frame = (
                torch.from_numpy(frame)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                .mul_(1 / 255)
            )

            frame = frame.contiguous(memory_format=torch.channels_last)

            if self.isCudaAvailable:
                torch.cuda.set_stream(self.stream[self.currentStream])
                if self.half:
                    frame = frame.cuda(non_blocking=True).half()
                else:
                    frame = frame.cuda(non_blocking=True)
            else:
                frame = frame.cpu()

            frame = self.model(frame)
            frame = frame.squeeze(0).permute(1, 2, 0).mul_(255).clamp_(0, 255).byte()

            if self.isCudaAvailable:
                torch.cuda.synchronize(self.stream[self.currentStream])
                self.currentStream = (self.currentStream + 1) % len(self.stream)

            return frame.cpu().numpy()
