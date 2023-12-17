from .cugan_arch import UpCunet2x, UpCunet3x, UpCunet4x, UpCunet2x_fast

import os
import requests
import torch


class Cugan:
    def __init__(self, upscale_method, upscale_factor, cugan_kind, half):
        self.upscale_method = upscale_method
        self.upscale_factor = upscale_factor
        self.cugan_kind = cugan_kind
        self.half = half

        self.handle_models()

    def handle_models(self):
        if self.upscale_method == "shufflecgan":
            self.model = UpCunet2x_fast(in_channels=3, out_channels=3)
            self.filename = "sudo_shuffle_cugan_9.584.969.pth"
        else:
            model_path_prefix = "cugan"
            model_path_suffix = "-latest"
            model_path_middle = f"up{self.upscale_factor}x"
            model_map = {2: UpCunet2x, 3: UpCunet3x, 4: UpCunet4x}
            self.model = model_map[self.upscale_factor](
                in_channels=3, out_channels=3)
            self.filename = f"{model_path_prefix}_{model_path_middle}{model_path_suffix}-{self.cugan_kind}.pth"

        dir_name = os.path.dirname(os.path.abspath(__file__))
        weights_dir = os.path.join(dir_name, "weights")
        if not os.path.exists(weights_dir):
            os.makedirs(weights_dir)

        if not os.path.exists(os.path.join(weights_dir, self.filename)):
            print(f"Downloading {self.model_type.upper()}  model...")
            url = f"https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/{self.filename}"
            response = requests.get(url)
            if response.status_code == 200:
                with open(os.path.join(weights_dir, self.filename), "wb") as file:
                    file.write(response.content)

        model_path = os.path.abspath(os.path.join(weights_dir, self.filename))

        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval().cuda() if torch.cuda.is_available() else self.model.eval()

        if self.half:
            self.model.half()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
        if self.half:
            torch.set_default_tensor_type(torch.cuda.HalfTensor)

    @torch.inference_mode
    def run(self, frame):
        with torch.no_grad():
            frame = torch.from_numpy(frame).permute(
                2, 0, 1).unsqueeze(0).float().div_(255)
            if self.half:
                frame = frame.half()
            frame = self.model(frame)
            frame = frame.squeeze(0).permute(
                1, 2, 0).mul_(255).clamp_(0, 255).byte()
            return frame.cpu().numpy()
