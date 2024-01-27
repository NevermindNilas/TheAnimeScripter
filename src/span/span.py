from .span_arch import SPAN

import wget
import os
import torch
import torch.nn.functional as F


class SpanSR:
    def __init__(self, upscale_factor, half, width, height):
        self.upscale_factor = upscale_factor
        self.half = half
        self.width = width
        self.height = height

        self.handle_models()

    def handle_models(self):
        # Apparently this can improve performance slightly
        torch.set_float32_matmul_precision("medium")

        dir_name = os.path.dirname(os.path.abspath(__file__))
        weights_dir = os.path.join(dir_name, "weights")
        if not os.path.exists(weights_dir):
            os.makedirs(weights_dir)
        
        self.filename = "2xHFA2kSPAN_27k.pth"
        if not os.path.exists(os.path.join(weights_dir, self.filename)):
            print(f"Downloading Span model...")
            url = f"https://github.com/NevermindNilas/TAS-Modes-Host/releases/download/main/{self.filename}"
            wget.download(url, out=os.path.join(weights_dir, self.filename))

        model_path = os.path.join(weights_dir, self.filename)
        self.model = SPAN(3, 3, upscale=2, feature_channels=48)

        self.cuda_available = torch.cuda.is_available()
        
        self.model.load_state_dict(torch.load(
            model_path, map_location="cpu")["params"])

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
            frame = torch.from_numpy(frame).permute(
                2, 0, 1).unsqueeze(0).float().mul_(1/255)

            if self.cuda_available:
                if self.half:
                    frame = frame.cuda().half()
                else:
                    frame = frame.cuda()
            else:
                frame = frame.cpu()

            if self.pad_width != 0 or self.pad_height != 0:
                frame = self.pad_frame(frame)

            frame = self.model(frame)
            frame = frame[:, :, :self.upscaled_height, :self.upscaled_width]
            frame = frame.squeeze(0).permute(
                1, 2, 0).mul_(255).clamp_(0, 255).byte()

            return frame.cpu().numpy()
