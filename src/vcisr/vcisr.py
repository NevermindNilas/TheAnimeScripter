
import wget
import os
import torch
import torch.nn.functional as F

#from .arch.rrdb import RRDBNet
#from .arch.swinir import SwinIR
from .arch.grl import GRL


class VCISR:
    def __init__(self, upscale_factor, half, width, height):
        self.upscale_factor = 4  # the only models available for now are 4x
        self.half = 0 # FP16 doesn't work just yet, same for BF16
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

        self.filename = "4x_VCISR_generator.pth"
        if not os.path.exists(os.path.join(weights_dir, self.filename)):
            print(f"Downloading VCISR model...")
            url = f"https://github.com/NevermindNilas/TAS-Modes-Host/releases/download/main/{self.filename}"
            wget.download(url, out=os.path.join(weights_dir, self.filename))

        model_path = os.path.join(weights_dir, self.filename)

        model = torch.load(model_path, map_location="cpu")

        if 'model_state_dict' in model:
            weight = model['model_state_dict']

            self.model = GRL(
                upscale=4,
                img_size=144,
                window_size=8,
                depths=[4, 4, 4, 4],
                embed_dim=128,
                num_heads_window=[2, 2, 2, 2],
                num_heads_stripe=[2, 2, 2, 2],
                mlp_ratio=2,
                qkv_proj_type="linear",
                anchor_proj_type="avgpool",
                anchor_window_down_factor=2,
                out_proj_type="linear",
                conv_type="1conv",
                upsampler="pixelshuffle",
            )

            old_keys = [key for key in weight]
            for old_key in old_keys:
                if old_key[:10] == "_orig_mod.":
                    new_key = old_key[10:]
                    weight[new_key] = weight[old_key]
                    del weight[old_key]

            self.model.load_state_dict(weight)

        self.cuda_available = torch.cuda.is_available()
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
