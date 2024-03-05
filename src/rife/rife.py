import os
import torch

from src.downloadModels import downloadModels, weightsDir
from torch.nn import functional as F


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
        self.interpolation_factor = interpolation_factor
        self.half = half
        self.UHD = UHD
        self.scale = 1.0
        self.width = width
        self.height = height
        self.interpolate_method = interpolate_method
        self.ensemble = ensemble
        self.nt = nt

        self.handle_model()

    def handle_model(self):
        match self.interpolate_method:
            case "rife" | "rife4.14":
                from .rife414.RIFE_HDv3 import Model
                self.filename = "rife414.pkl"

            case "rife4.14-lite":
                from .rife414lite.RIFE_HDv3 import Model
                self.filename = "rife414lite.pkl"

            case "rife4.13-lite":
                from .rife413lite.RIFE_HDv3 import Model
                self.filename = "rife413lite.pkl"

            case "rife4.6":
                from .rife46.RIFE_HDv3 import Model
                self.filename = "rife46.pkl"
        
        filenameWithoutExtension = os.path.splitext(self.filename)[0]
        if not os.path.exists(os.path.join(weightsDir, "rife", filenameWithoutExtension, "flownet.pkl")):
            modelDir = os.path.dirname(downloadModels(self.interpolate_method))
        else:
            modelDir = os.path.dirname(os.path.join(weightsDir, "rife", filenameWithoutExtension, "flownet.pkl"))

        # Apparently this can improve performance slightly
        torch.set_float32_matmul_precision("medium")

        if self.UHD:
            self.scale = 0.5

        ph = ((self.height - 1) // 64 + 1) * 64
        pw = ((self.width - 1) // 64 + 1) * 64
        self.padding = (0, pw - self.width, 0, ph - self.height)

        self.cuda_available = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda_available else "cpu")

        torch.set_grad_enabled(False)
        if self.cuda_available:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            if self.half:
                torch.set_default_dtype(torch.float16)

        self.model = Model()
        self.model.load_model(modelDir, -1)
        self.model.eval()
        

        if self.cuda_available and self.half:
            self.model.half()

        self.model.device()

        self.I0 = None
             
    @torch.inference_mode()
    def make_inference(self, n):
        output = self.model.inference(self.I0, self.I1, n, self.scale, self.ensemble)
        output = output[:, :, : self.height, : self.width]
        output = (output[0] * 255.0).byte().cpu().numpy().transpose(1, 2, 0)

        return output

    def cacheFrame(self):
        self.I0 = self.I1.clone()
        
    @torch.inference_mode()
    def run(self, I1):
        if self.I0 is None:
            self.I0 = (
                torch.from_numpy(I1)
                .to(self.device, non_blocking=True)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                / 255.0
            )
            if self.cuda_available and self.half:
                self.I0 = self.I0.half()
                
            if self.padding != (0, 0, 0, 0):
                self.I0 = F.pad(self.I0, [0, self.padding[1], 0, self.padding[3]])
                
            return False
        
        self.I1 = (
            torch.from_numpy(I1)
            .to(self.device, non_blocking=True)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            / 255.0
        )
        
        if self.cuda_available and self.half:
            self.I1 = self.I1.half()

        if self.padding != (0, 0, 0, 0):
            self.I1 = F.pad(self.I1, [0, self.padding[1], 0, self.padding[3]])
            
        return True
