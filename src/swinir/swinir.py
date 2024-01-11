import os
import requests
import torch
import numpy as np

from torch.nn import functional as F
from .swinir_arch import SwinIR as SwinIR_arch

# Apparently this can improve performance slightly
torch.set_float32_matmul_precision("medium")
class Swinir():
    def __init__(self, upscale_factor, half, width, height):
        self.upscale_factor = upscale_factor
        self.half = half
        self.width = width
        self.height = height
        self.weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
        self.filename = "2x_Bubble_AnimeScale_SwinIR_Small_v1.pth"
        self.model_path = os.path.join(self.weights_dir, self.filename)
        self.url = "https://github.com/Bubblemint864/AI-Models/releases/download/2x_Bubble_AnimeScale_SwinIR_Small_v1/2x_Bubble_AnimeScale_SwinIR_Small_v1.pth"
        self.h_pad = (height // 8 + 1) * 8 - height
        self.w_pad = (width // 8 + 1) * 8 - width

        self.handle_models()

    def handle_models(self):
        self.create_weights_dir()
        self.download_model()
        self.load_model()

    def create_weights_dir(self):
        if not os.path.exists(self.weights_dir):
            os.makedirs(self.weights_dir)

    def download_model(self):
        if not os.path.exists(self.model_path):
            print(f"Downloading SWINIR model...")
            response = requests.get(self.url)
            if response.status_code == 200:
                with open(self.model_path, "wb") as file:
                    file.write(response.content)

    def load_model(self):
        pretrained_model = torch.load(self.model_path, map_location="cpu")
        pretrained_model = pretrained_model["params"]

        self.model = SwinIR_arch(
            upscale=2,
            in_chans=3,
            img_size=32,
            window_size=8,
            img_range=1.,
            depths=[6, 6, 6, 6],
            embed_dim=60,
            num_heads=[6, 6, 6, 6],
            mlp_ratio=2,
            upsampler="pixelshuffle",
            resi_connection="1conv",
        )

        self.model.load_state_dict(pretrained_model)
        self.model.eval().cuda() if torch.cuda.is_available() else self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            if self.half:
                torch.set_default_tensor_type(torch.cuda.HalfTensor)
                self.model.half()

    def inference(self, frame):
        if self.half:
            frame = frame.half()
        with torch.no_grad():
            return self.model(frame)

    def pad_frame(self, frame):
        frame = F.pad(frame, (0, self.w_pad, 0, self.h_pad), mode='reflect')
        return frame

    def transform_frame(self, frame):
        frame = torch.from_numpy(frame.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).div_(255)
        frame = frame.to(self.device)
        if self.width % 8 != 0 or self.height % 8 != 0:
            frame = self.pad_frame(frame)
        return frame
    
    @torch.inference_mode
    def run(self, frame):
        frame = self.transform_frame(frame)
        frame = self.inference(frame)
        frame = frame[:, :, :self.height * self.upscale_factor, :self.width * self.upscale_factor]
        frame = frame.squeeze(0).permute(1, 2, 0).mul_(255).clamp_(0, 255).byte()
        return frame.cpu().numpy()