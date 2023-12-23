import os
import torch
import requests
import numpy as np

from .srvgg_arch import SRVGGNetCompact


class Compact():
    def __init__(self, upscale_method, half):
        self.upscale_method = upscale_method
        self.half = half

        self.handle_models()

    def handle_models(self):
        if self.upscale_method == "compact":
            filename = "2x_AnimeJaNai_V2_Compact_36k.pth"
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
            url = f"https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/{filename}"
            response = requests.get(url)
            if response.status_code == 200:
                with open(os.path.join(weights_dir, filename), "wb") as file:
                    file.write(response.content)

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
        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
        if self.half:
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
            self.model.half()

    @torch.inference_mode
    def run(self, frame):
        with torch.no_grad():
            frame = torch.from_numpy(frame).permute(
                2, 0, 1).unsqueeze(0).float().mul_(1/255)
            if self.half:
                frame = frame.cuda().half()
            else:
                try:
                    frame = frame.cuda()
                except:
                    frame = frame.cpu()

            frame = self.model(frame)
            frame = frame.squeeze(0).permute(
                1, 2, 0).mul_(255).clamp_(0, 255).byte()
            return frame.cpu().numpy()

