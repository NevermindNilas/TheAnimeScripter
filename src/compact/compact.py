import os
import torch
import wget

from torch.nn import functional as F
from .srvgg_arch import SRVGGNetCompact


class Compact():
    def __init__(self, upscale_method, half, width, height):
        self.upscale_method = upscale_method
        self.half = half
        self.width = width
        self.height = height

        self.pad_width = 0 if self.width % 8 == 0 else 8 - (self.width % 8)
        self.pad_height = 0 if self.height % 8 == 0 else 8 - (self.height % 8)

        self.handle_models()

    def handle_models(self):

        # Apparently this can improve performance slightly
        torch.set_float32_matmul_precision("medium")

        if self.upscale_method == "compact":
            filename = "2x_AnimeJaNai_V3_SmoothRC21_Compact_50k_fp32.pth"
        elif self.upscale_method == "ultracompact":
            filename = "2x_AnimeJaNai_V2_UltraCompact_30k.pth"
        elif self.upscale_method == "superultracompact":
            filename = "2x_AnimeJaNai_V2_SuperUltraCompact_100k.pth"

        dir_name = os.path.dirname(os.path.abspath(__file__))
        weights_dir = os.path.join(dir_name, "weights")

        if not os.path.exists(weights_dir):
            os.makedirs(weights_dir)

        if not os.path.exists(os.path.join(weights_dir, filename)):
            print(f"Downloading {self.upscale_method.upper()} model...")
            url = f"https://github.com/NevermindNilas/TAS-Modes-Host/releases/download/main/{filename}"
            wget.download(url, out=os.path.join(weights_dir, filename))

        model_path = os.path.join(weights_dir, filename)

        num_conv_map = {
            "compact": [64, 16],
            "ultracompact": [64, 8],
            "superultracompact": [24, 8]
        }

        num_feat, num_conv = num_conv_map[self.upscale_method]

        self.model = SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=num_feat,
            num_conv=num_conv,
            upscale=2,
            act_type="prelu",
        )

        self.model.load_state_dict(torch.load(
            model_path, map_location="cpu")["params"])

        self.model.eval().cuda() if torch.cuda.is_available() else self.model.eval()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            if self.half:
                torch.set_default_dtype(torch.float16)
                torch.set_default_device("cuda")
                self.model.half()

        self.upscaled_height = self.height * 2
        self.upscaled_width = self.width * 2

    def pad_frame(self, frame):
        frame = F.pad(frame, [0, self.pad_width, 0, self.pad_height])
        return frame

    @torch.inference_mode
    def run(self, frame):
        frame = torch.from_numpy(frame).permute(
            2, 0, 1).unsqueeze(0).float().mul_(1/255)
        try:
            if self.half:
                frame = frame.cuda().half()
            else:
                frame = frame.cuda()
        except:
            frame = frame.cpu()
            
        if self.pad_width != 0 or self.pad_height != 0:
            frame = self.pad_frame(frame)
            
        frame = self.model(frame)
        frame = frame[:, :, :self.upscaled_height, :self.upscaled_width]
        frame = frame.squeeze(0).permute(
            1, 2, 0).mul_(255).clamp_(0, 255).byte()
        return frame.cpu().numpy()
