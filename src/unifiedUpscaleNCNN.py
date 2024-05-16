
"""
# model can be 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19; 0 for default
#span
0: {"param": "spanx2_ch48.param", "bin": "spanx2_ch48.bin", "scale": 2, "folder": "models/SPAN"},
1: {"param": "spanx2_ch52.param", "bin": "spanx2_ch52.bin", "scale": 2, "folder": "models/SPAN"},
2: {"param": "spanx4_ch48.param", "bin": "spanx4_ch48.bin", "scale": 4, "folder": "models/SPAN"},
3: {"param": "spanx4_ch52.param", "bin": "spanx4_ch52.bin", "scale": 4, "folder": "models/SPAN"},
#custom span
4: {"param": "2xHFA2kSPAN_27k.param", "bin": "2xHFA2kSPAN_27k.bin", "scale": 2, "folder": "models/SPAN"},
5: {"param": "4xSPANkendata.param", "bin": "4xSPANkendata.bin", "scale": 4, "folder": "models/SPAN"},
6: {"param": "ClearReality4x.param", "bin": "ClearReality4x.bin", "scale": 4, "folder": "models/SPAN"},

#esrgan
7: {"param": "realesr-animevideov3-x2.param", "bin": "realesr-animevideov3-x2.bin", "scale": 2, "folder": "models/ESRGAN"},
8: {"param": "realesr-animevideov3-x3.param", "bin": "realesr-animevideov3-x3.bin", "scale": 3, "folder": "models/ESRGAN"},
9: {"param": "realesr-animevideov3-x4.param", "bin": "realesr-animevideov3-x4.bin", "scale": 4, "folder": "models/ESRGAN"},
10: {"param": "realesrgan-x4plus-x4.param", "bin": "realesrgan-x4plus.bin", "scale": 4, "folder": "models/ESRGAN"},
11: {"param": "realesrgan-x4plus-anime.param", "bin": "realesrgan-x4plus-anime.bin", "scale": 4, "folder": "models/ESRGAN"},

#cugan-se models 
12: {"param": "up2x-conservative.param", "bin": "up2x-conservative.bin", "scale": 2, "folder": "models/CUGAN/models-se"},
13: {"param": "up2x-no-denoise.param", "bin": "up2x-no-denoise.bin", "scale": 2, "folder": "models/CUGAN/models-se"},
14: {"param": "up2x-denoise1x.param", "bin": "up2x-denoise1x.bin", "scale": 2, "folder": "models/CUGAN/models-se"},
15: {"param": "up2x-denoise2x.param", "bin": "up2x-denoise2x.bin", "scale": 2, "folder": "models/CUGAN/models-se"},
16: {"param": "up2x-denoise3x.param", "bin": "up2x-denoise3x.bin", "scale": 2, "folder": "models/CUGAN/models-se"},

17: {"param": "up3x-conservative.param", "bin": "up3x-conservative.bin", "scale": 3, "folder": "models/CUGAN/models-se"},
18: {"param": "up3x-no-denoise.param", "bin": "up3x-no-denoise.bin", "scale": 3, "folder": "models/CUGAN/models-se"},
19: {"param": "up3x-denoise3x.param", "bin": "up3x-denoise3x.bin", "scale": 3, "folder": "models/CUGAN/models-se"},

20: {"param": "up4x-conservative.param", "bin": "up4x-conservative.bin", "scale": 4, "folder": "models/CUGAN/models-se"},
21: {"param": "up4x-no-denoise.param", "bin": "up4x-no-denoise.bin", "scale": 4, "folder": "models/CUGAN/models-se"},
22: {"param": "up4x-denoise3x.param", "bin": "up3x-denoise3x.bin", "scale": 4, "folder": "models/CUGAN/models-se"},

#cugan-pro models
23: {"param": "up2x-denoise3x.param", "bin": "up2x-denoise3x.bin", "scale": 2, "folder": "models/CUGAN/models-pro"},
24: {"param": "up2x-conservative.param", "bin": "up2x-conservative.bin", "scale": 2, "folder": "models/CUGAN/models-pro"},
25: {"param": "up2x-no-denoise.param", "bin": "up2x-no-denoise.bin", "scale": 2, "folder": "models/CUGAN/models-pro"},

26: {"param": "up3x-denoise3x", "bin": "denoise3x-up3x", "scale": 3, "folder": "models/CUGAN/models-pro"},
27: {"param": "up3x-conservative", "bin": "up3x-conservative.bin", "scale": 3, "folder": "models/CUGAN/models-pro"},
28: {"param": "up3x-no-denoise.param", "bin": "up3x-no-denoise.bin", "scale": 3, "folder": "models/CUGAN/models-pro"},

#shufflecugan
29: {"param": "sudo_shuffle_cugan-x2.param", "bin": "sudo_shuffle_cugan-x2.bin", "scale": 2, "folder": "models/SHUFFLECUGAN"},
"""
import torch
from upscale_ncnn_py import UPSCALE

class UniversalNCNN:
    def __init__(
        self,
        upscaleMethod,
        upscaleFactor,
        nt,
    ):
        self.upscaleMethod = upscaleMethod
        self.upscaleFactor = upscaleFactor
        self.nt = nt

        match (self.upscaleMethod, self.upscaleFactor):
            case ("span-ncnn", 2):
                self.modelId = 4
            case ("span-ncnn", 4):
                self.modelId = 5
            case ("realesrgan-ncnn", 2):
                self.modelId = 7
            case ("realesrgan-ncnn", 3):
                self.modelId = 8
            case ("realesrgan-ncnn", 4):
                self.modelId = 9
            case ("cugan-ncnn", 2, "conservative"):
                self.modelId = 23
            case ("cugan-ncnn", 3, "conservative"):
                self.modelId = 27
            case ("cugan-ncnn", 4, "conservative"):
                self.modelId = 20
            case ("shufflecugan-ncnn", 2):
                self.modelId = 29
            case _:
                raise ValueError(
                    f"Invalid upscale method {self.upscaleMethod} with factor {self.upscaleFactor}"
                )

        if self.nt == 1:
            self.nt == 2

        self.model = UPSCALE(
            gpuid=0,
            tta_mode=False,
            tilesize=0,
            model=self.modelId,
            num_threads=self.nt,
        )

    def run(self, frame):
        frame = self.model.process_cv2(frame.cpu().numpy())
        return torch.from_numpy(frame).permute(2, 0, 1)
