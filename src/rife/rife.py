import os
import torch

from src.downloadModels import downloadModels, weightsDir, modelsMap
from torch.nn import functional as F

# Apparently this can improve performance slightly
torch.set_float32_matmul_precision("medium")


class Rife:
    def __init__(
        self,
        interpolation_factor,
        half,
        width,
        height,
        UHD,
        interpolate_method,
        ensemble=False,
        nt=1,
    ):
        """
        Initialize the RIFE model

        Args:
            interpolation_factor (int): The factor to interpolate by.
            half (bool): Whether to use half precision.
            width (int): The width of the input frame.
            height (int): The height of the input frame.
            UHD (bool): Whether to use UHD mode.
            interpolate_method (str): The method to use for interpolation.
            ensemble (bool): Whether to use ensemble mode.
            nt (int): The number of streams to use, not available for now.
        """
        self.interpolation_factor = interpolation_factor
        self.half = half
        self.UHD = UHD
        self.scale = 1.0
        self.width = width
        self.height = height
        self.interpolate_method = interpolate_method
        self.ensemble = ensemble
        self.nt = nt

        if self.UHD:
            self.scale = 0.5

        self.handle_model()

    def handle_model(self):
        """
        Load the desired model
        """

        self.filename = modelsMap(self.interpolate_method)
        if not os.path.exists(os.path.join(weightsDir, "rife", self.filename)):
            modelPath = downloadModels(model=self.interpolate_method)
        else:
            modelPath = os.path.join(weightsDir, "rife", self.filename)

        match self.interpolate_method:
            case "rife" | "rife4.15":
                from .arches.IFNet_rife415 import IFNet
            case "rife4.14":
                from .arches.IFNet_rife414 import IFNet
            case "rife4.13-lite":
                from .arches.IFNet_rife413lite import IFNet
            case "rife4.6":
                from .arches.IFNet_rife46 import IFNet

        self.model = IFNet()
        self.isCudaAvailable = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.isCudaAvailable else "cpu")

        torch.set_grad_enabled(False)
        if self.isCudaAvailable:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            if self.half:
                torch.set_default_dtype(torch.float16)

        if self.isCudaAvailable and self.half:
            self.model.half()

        self.model.load_state_dict(torch.load(modelPath))
        self.model.eval().cuda() if self.isCudaAvailable else self.model.eval()
        self.model.to(self.device)

        ph = ((self.height - 1) // 32 + 1) * 32
        pw = ((self.width - 1) // 32 + 1) * 32
        self.padding = (0, pw - self.width, 0, ph - self.height)

        self.I0 = None
        self.scaleList = [
            8 / self.scale,
            4 / self.scale,
            2 / self.scale,
            1 / self.scale,
        ]

    @torch.inference_mode()
    def make_inference(self, timestep):
        output = self.model(self.I0, self.I1, timestep, self.scaleList, self.ensemble)
        output = output[:, :, : self.height, : self.width]
        output = (output[0] * 255.0).byte().cpu().numpy().transpose(1, 2, 0)

        return output

    def cacheFrame(self):
        self.I0 = self.I1.clone()

    @torch.inference_mode()
    def processFrame(self, frame):
        frame = (
            torch.from_numpy(frame)
            .to(self.device, non_blocking=True)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .mul_(1 / 255)
            .half()
            if self.isCudaAvailable and self.half
            else frame
        )

        if self.padding != (0, 0, 0, 0):
            frame = F.pad(frame, [0, self.padding[1], 0, self.padding[3]])

        return frame.contiguous(memory_format=torch.channels_last)

    @torch.inference_mode()
    def run(self, I1):
        if self.I0 is None:
            self.I0 = self.processFrame(I1)
            return False

        self.I1 = self.processFrame(I1)
        return True
