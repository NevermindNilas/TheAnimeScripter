import os
import torch
import logging
import numpy as np
import torch.nn.functional as F

from spandrel import ModelLoader, ImageModelDescriptor
from .downloadModels import downloadModels, weightsDir

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

            filenameMap = {
                "dpir": "placeholder",
                "scunet": "scunet_color_real_psnr.pth",
                "nafnet": "NAFNet-GoPro-width64.pth",
                "span-denoise": "1x_span_anime_pretrain.pth"
            }

            self.filename = filenameMap.get(self.model)
            if not os.path.exists(os.path.join(weightsDir, self.model, self.filename)):
                modelPath = downloadModels(model=self.model)

            else:
                modelPath = os.path.join(weightsDir, self.model, self.filename)

        else:
            if os.path.isfile(self.customModel):
                modelPath = self.customModel

            else:
                raise FileNotFoundError(f"Model file {self.customModel} not found")

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
            self.stream = [torch.cuda.Stream() for _ in range(self.nt)]
            self.currentStream = 0
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            if self.half:
                torch.set_default_dtype(torch.float16)
                self.model.half()

        # Hardcoded these for the moment being. I will need to check what padding is needed for each model
        self.padWidth = 0 if self.width % 8 == 0 else 8 - (self.width % 8)
        self.padHeight = 0 if self.height % 8 == 0 else 8 - (self.height % 8)

    def pad_frame(self, frame):
        frame = F.pad(frame, [0, self.padWidth, 0, self.padHeight])
        return frame

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
                    frame = frame.cuda().half()
                else:
                    frame = frame.cuda()
            else:
                frame = frame.cpu()

            if self.padWidth != 0 or self.padHeight != 0:
                frame = self.pad_frame(frame)

            frame = self.model(frame)
            frame = frame[:, :, : self.height, : self.width]
            frame = frame.squeeze(0).permute(1, 2, 0).mul_(255).clamp_(0, 255).byte()

            if self.isCudaAvailable:
                torch.cuda.synchronize(self.stream[self.currentStream])
                self.currentStream = (self.currentStream + 1) % len(self.stream)

            return frame.cpu().numpy()
