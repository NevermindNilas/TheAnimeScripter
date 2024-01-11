import os
import torch
import numpy as np

from torch.nn import functional as F


class Rife:
    def __init__(self, interpolation_factor, half, width, height, UHD):
        self.interpolation_factor = interpolation_factor
        self.half = half
        self.UHD = UHD
        self.scale = 1.0
        self.width = int(width)
        self.height = int(height)
        self.modelDir = os.path.dirname(os.path.realpath(__file__))

        self.handle_model()

    def handle_model(self):
        if self.UHD == True:
            self.scale = 0.5

        ph = ((self.height - 1) // 64 + 1) * 64
        pw = ((self.width - 1) // 64 + 1) * 64
        self.padding = (0, pw - self.width, 0, ph - self.height)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        
        # Doing a torch cuda check is rather expensive on start-up times so I just decided to keep it simple
        self.cuda_available = False
        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            self.cuda_available = True
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            if self.half:
                torch.set_default_tensor_type(torch.cuda.HalfTensor)
      
        from .RIFE_HDv3 import Model
        self.model = Model()
        self.model.load_model(self.modelDir, -1)
        self.model.eval()
        
        if self.cuda_available == True:
            if self.half:
                self.model.half()
                
        self.model.device()

    def make_inference(self, I0, I1, n):
        res = []
        for i in range(n):
            res.append(self.model.inference(
                I0, I1, (i + 1) * 1. / (n + 1), self.scale))

        return res

    def pad_image(self, img):
        img = F.pad(img, self.padding)
        if self.cuda_available and self.half:
            img = img.half()
        return img

    def run(self, I0, I1):
        buffer = []
        I0 = torch.from_numpy(np.transpose(I0, (2, 0, 1))).to(
            self.device, non_blocking=True).unsqueeze(0).float() / 255.
        I1 = torch.from_numpy(np.transpose(I1, (2, 0, 1))).to(
            self.device, non_blocking=True).unsqueeze(0).float() / 255.
        
        I0 = self.pad_image(I0)
        I1 = self.pad_image(I1)
        
        output = self.make_inference(I0, I1, self.interpolation_factor - 1)

        for mid in output:
            mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
            buffer.append(mid[:self.height, :self.width, :])

        return buffer