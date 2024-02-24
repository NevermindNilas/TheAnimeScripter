import realesrgan_ncnn_py as renp
import torch
import os
import logging

from torch.nn import functional as F
from src.downloadModels import downloadModels, weightsDir
from .rrdbnet_arch import RRDBNet as esrgan


class RealEsrgan:
    def __init__(
        self,
        upscaleFactor: int = 2,
        width: int = 0,
        height: int = 0,
        customModel: str = "",
        nt: int = 1,
        half: bool = True,
    ):
        self.upscaleFactor = upscaleFactor
        self.width = width
        self.height = height
        self.customModel = customModel
        self.nt = nt
        self.half = half

        self.handleModels()

    def handleModels(self):
        if self.upscaleFactor in [3, 4]:
            raise NotImplementedError("factors other than 2x upscale is not supported for now")

        torch.set_float32_matmul_precision("medium")

        # Needs more testing
        self.model = esrgan(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=self.upscaleFactor
            )

        if self.customModel == "":
            upscaleFactorMap = {
                2: "2xHFA2kShallowESRGAN.pth",
                3: "N/A",
                4: "N/A",
            }
            self.filename = upscaleFactorMap.get(self.upscaleFactor)
            if not os.path.exists(
                os.path.join(weightsDir, "realesrgan", self.filename)
            ):
                modelPath = downloadModels(
                    model="realesrgan",
                    upscaleFactor=self.upscaleFactor,
                )
            else:
                modelPath = os.path.join(weightsDir, "realesrgan", self.filename)
        else:
            logging.info(f"Using custom model: {self.customModel}")
            modelPath = self.customModel
        
        self.cuda_available = torch.cuda.is_available()
        if modelPath.endswith(".pth"):
            state_dict = torch.load(modelPath, map_location="cpu")
            if "params" in state_dict:
                self.model.load_state_dict(state_dict["params"])
            elif "params_ema" in state_dict:
                self.model.load_state_dict(state_dict["params_ema"])
            else:
                self.model.load_state_dict(state_dict)
        elif modelPath.endswith(".onnx"):
            self.model = torch.onnx.load(modelPath)

        self.model = (
            self.model.eval().cuda() if self.cuda_available else self.model.eval()
        )
        
        self.device = torch.device("cuda" if self.cuda_available else "cpu")

        if self.cuda_available:
            self.stream = [torch.cuda.Stream() for _ in range(self.nt)]
            self.current_stream = 0
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            if self.half:
                torch.set_default_dtype(torch.float16)
                self.model.half()

        self.pad_width = 0 if self.width % 8 == 0 else 8 - (self.width % 8)
        self.pad_height = 0 if self.height % 8 == 0 else 8 - (self.height % 8)

        self.upscaled_height = self.height * self.upscaleFactor
        self.upscaled_width = self.width * self.upscaleFactor

    def pad_frame(self, frame):
        frame = F.pad(frame, [0, self.pad_width, 0, self.pad_height])
        return frame

    @torch.inference_mode()
    def run(self, frame):
        with torch.no_grad():
            frame = (
                torch.from_numpy(frame)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                .mul_(1 / 255)
            )

            if self.cuda_available:
                torch.cuda.set_stream(self.stream[self.current_stream])
                if self.half:
                    frame = frame.cuda().half()
                else:
                    frame = frame.cuda()
            else:
                frame = frame.cpu()

            if self.pad_width != 0 or self.pad_height != 0:
                frame = self.pad_frame(frame)

            frame = self.model(frame)
            frame = frame[:, :, : self.upscaled_height, : self.upscaled_width]
            frame = frame.squeeze(0).permute(1, 2, 0).mul_(255).clamp_(0, 255).byte()

            if self.cuda_available:
                torch.cuda.synchronize(self.stream[self.current_stream])
                self.current_stream = (self.current_stream + 1) % len(self.stream)

            return frame.cpu().numpy()


class RealEsrganNCNN:
    def __init__(
        self,
        upscaleFactor: int = 2,
    ):
        self.upscaleFactor = upscaleFactor
        """
        # 0: {"param": "realesr-animevideov3-x2.param", "bin": "realesr-animevideov3-x2.bin", "scale": 2},
        # 1: {"param": "realesr-animevideov3-x3.param", "bin": "realesr-animevideov3-x3.bin", "scale": 3},
        # 2: {"param": "realesr-animevideov3-x4.param", "bin": "realesr-animevideov3-x4.bin", "scale": 4},
        # 3: {"param": "realesrgan-x4plus-anime.param", "bin": "realesrgan-x4plus-anime.bin", "scale": 4},
        # 4: {"param": "realesrgan-x4plus.param", "bin": "realesrgan-x4plus.bin", "scale": 4}
        """
        modelChooserMap = {
            2: 0,
            3: 1,
            4: 2,
        }

        modelChooser = modelChooserMap.get(self.upscaleFactor)
        self.model = renp.Realesrgan(
            gpuid=0,
            tta_mode=False,
            tilesize=0,
            model=modelChooser,
        )

    def run(self, frame):
        return self.model.process_cv2(frame)
