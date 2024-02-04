import os
import wget
import torch
import numpy as np
import logging

from torch.nn import functional as F
from .swinir_arch import SwinIR as SwinIR_arch

# Apparently this can improve performance slightly
torch.set_float32_matmul_precision("medium")
class Swinir:
    def __init__(self, upscale_factor, half, width, height, custom_model):
        self.upscale_factor = upscale_factor
        self.half = half
        self.width = width
        self.height = height
        self.custom_model = custom_model

        self.handle_models()

    def handle_models(self):
        # Apparently this can improve performance slightly
        torch.set_float32_matmul_precision("medium")

        # There's like a billion arches for SwinIR, for now this will do until I add them all
        self.model = SwinIR_arch(
            upscale=2,
            img_size=64,
            window_size=8,
            depths=[6, 6, 6, 6],
            embed_dim=60,
            num_heads=[6, 6, 6, 6],
            mlp_ratio=2,
            upsampler="pixelshuffledirect",
            resi_connection="1conv",
        )
        
        if self.custom_model == "":
            self.filename = "2xHFA2kSwinIR-S.pth"

            dir_name = os.path.dirname(os.path.abspath(__file__))
            weights_dir = os.path.join(dir_name, "weights")
            if not os.path.exists(weights_dir):
                os.makedirs(weights_dir)

            if not os.path.exists(os.path.join(weights_dir, self.filename)):
                print(f"Downloading SWINIR model...")
                url = "https://github.com/NevermindNilas/TAS-Modes-Host/releases/download/main/2xHFA2kSwinIR-S.pth"
                wget.download(url, out=os.path.join(weights_dir, self.filename))

            model_path = os.path.join(weights_dir, self.filename)

        else:
            logging.info(f"Using custom model: {self.custom_model}")
            model_path = self.custom_model

        self.cuda_available = torch.cuda.is_available()

        if model_path.endswith('.pth'):
            state_dict = torch.load(model_path, map_location="cpu")
            if "params" in state_dict:
                self.model.load_state_dict(state_dict["params"])
            else:
                self.model.load_state_dict(state_dict)
        elif model_path.endswith('.onnx'):
            self.model = torch.onnx.load(model_path)

        self.model = self.model.eval().cuda() if self.cuda_available else self.model.eval()
        self.device = torch.device("cuda" if self.cuda_available else "cpu")

        if self.cuda_available:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            if self.half:
                torch.set_default_dtype(torch.float16)
                self.model.half()

        self.pad_width = 0 if self.width % 8 == 0 else 8 - (self.width % 8)
        self.pad_height = 0 if self.height % 8 == 0 else 8 - (self.height % 8)

        self.upscaled_height = self.height * self.upscale_factor
        self.upscaled_width = self.width * self.upscale_factor

    def pad_frame(self, frame):
        frame = F.pad(frame, [0, self.pad_width, 0, self.pad_height])
        return frame

    @torch.inference_mode
    def run(self, frame):
        with torch.no_grad():
            frame = torch.from_numpy(frame.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).div_(255)
            frame = frame.to(self.device)

            if self.pad_width != 0 or self.pad_height != 0:
                frame = self.pad_frame(frame)

            frame = self.model(frame)
            frame = frame[:, :, :self.upscaled_height, :self.upscaled_width]
            frame = frame.squeeze(0).permute(1, 2, 0).mul_(255).clamp_(0, 255).byte()

            return frame.cpu().numpy()