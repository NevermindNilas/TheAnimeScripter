import wget
import os
import torch
import torch.nn.functional as F
import logging

from realcugan_ncnn_py import Realcugan
from .cugan_arch import UpCunet2x, UpCunet3x, UpCunet4x, UpCunet2x_fast


class Cugan:
    def __init__(self, upscale_method, upscale_factor, cugan_kind, half, width, height, custom_model):
        self.upscale_method = upscale_method
        self.upscale_factor = upscale_factor
        self.cugan_kind = cugan_kind
        self.half = half
        self.width = width
        self.height = height
        self.custom_model = custom_model

        self.handle_models()

    def handle_models(self):
        # Apparently this can improve performance slightly
        torch.set_float32_matmul_precision("medium")
        model_map = {2: UpCunet2x, 3: UpCunet3x, 4: UpCunet4x}

        if self.custom_model == "":
            if self.upscale_method == "shufflecugan":
                self.model = UpCunet2x_fast(in_channels=3, out_channels=3)
                self.filename = "sudo_shuffle_cugan_9.584.969.pth"
            elif self.upscale_method == "cugan":
                model_path_prefix = "cugan"
                model_path_suffix = "-latest"
                model_path_middle = f"up{self.upscale_factor}x"
                self.model = model_map[self.upscale_factor](
                    in_channels=3, out_channels=3)
                self.filename = f"{model_path_prefix}_{model_path_middle}{
                    model_path_suffix}-{self.cugan_kind}.pth"

            dir_name = os.path.dirname(os.path.abspath(__file__))
            weights_dir = os.path.join(dir_name, "weights")
            if not os.path.exists(weights_dir):
                os.makedirs(weights_dir)

            if not os.path.exists(os.path.join(weights_dir, self.filename)):
                print(f"Downloading {self.upscale_method.upper()} model...")
                url = f"https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/{
                    self.filename}"
                wget.download(url, out=os.path.join(
                    weights_dir, self.filename))

            model_path = os.path.join(weights_dir, self.filename)

        else:
            self.model = model_map[self.upscale_factor](
                in_channels=3, out_channels=3)

            logging.info(
                f"Using custom model: {self.custom_model}")

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


class CuganNCNN():
    def __init__(self, num_threads, upscale_factor):
        """
        Barebones for now
        """
        self.num_threads = num_threads
        self.upscale_factor = upscale_factor

        self.realcugan = Realcugan(
            num_threads=self.num_threads, gpuid=0, tta_mode=False, scale=self.upscale_factor)

    def run(self, frame):
        frame = self.realcugan.process_cv2(frame)
        return frame
