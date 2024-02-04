import wget
import os
import torch
import torch.nn.functional as F
import logging

from .omnisr_arch import omnisr

class OmniSR:
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

        self.model = omnisr(
            upsampling=self.upscale_factor,
            window_size=8,
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64 # Might have to look into this
        )
        
        if self.custom_model == "":
            self.filename = "2xHFA2kOmniSR.pth"

            dir_name = os.path.dirname(os.path.abspath(__file__))
            weights_dir = os.path.join(dir_name, "weights")
            if not os.path.exists(weights_dir):
                os.makedirs(weights_dir)

            if not os.path.exists(os.path.join(weights_dir, self.filename)):
                print(f"Downloading OmniSR model...")
                url = f"https://github.com/NevermindNilas/TAS-Modes-Host/releases/download/main/{self.filename}"
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
            frame = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().mul_(1/255)
            frame = frame.to(self.device)

            if self.pad_width != 0 or self.pad_height != 0:
                frame = self.pad_frame(frame)

            frame = self.model(frame)
            frame = frame[:, :, :self.upscaled_height, :self.upscaled_width]
            frame = frame.squeeze(0).permute(1, 2, 0).mul_(255).clamp_(0, 255).byte()

            return frame.cpu().numpy()