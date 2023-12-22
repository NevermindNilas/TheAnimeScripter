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
        self.width = width
        self.height = height
        self.modelDir = os.path.dirname(os.path.realpath(__file__))

        self.handle_model()

    def handle_model(self):
        if self.UHD == True:
            self.scale = 0.5

        tmp = max(128, int(128 / self.scale))
        ph = ((self.height - 1) // tmp + 1) * tmp
        pw = ((self.width - 1) // tmp + 1) * tmp
        self.padding = (0, pw - self.width, 0, ph - self.height)


        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            if self.half:
                torch.set_default_tensor_type(torch.cuda.HalfTensor)
        try:
            from .RIFE_HDv3 import Model
        except:
            raise Exception(
                "Cannot load RIFE model, please check your weights")

        self.model = Model()
        self.model.load_model(self.modelDir, -1)
        self.model.eval()
        self.model.device()

    def make_inference(self, I0, I1, n):
        res = []
        for i in range(n):
            res.append(self.model.inference(
                I0, I1, (i + 1) * 1. / (n + 1), self.scale))

        return res

    def pad_image(self, img):
        if self.half:
            return F.pad(img, self.padding).half()
        else:
            return F.pad(img, self.padding)

    def run(self, I0, I1, n):
        buffer = []
        I0 = torch.from_numpy(np.transpose(I0, (2, 0, 1))).to(
            self.device, non_blocking=True).unsqueeze(0).float() / 255.
        I1 = torch.from_numpy(np.transpose(I1, (2, 0, 1))).to(
            self.device, non_blocking=True).unsqueeze(0).float() / 255.
        
        I0 = self.pad_image(I0)
        I1 = self.pad_image(I1)
        
        output = self.make_inference(I0, I1, n - 1)

        for mid in output:
            mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
            buffer.append(mid[:self.height, :self.width])

        return buffer
