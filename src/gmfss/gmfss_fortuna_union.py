import os
import wget
import torch
import numpy as np
import logging

from torch.nn import functional as F

# from: https://github.com/HolyWu/vs-gmfss_fortuna/blob/master/vsgmfss_fortuna/__init__.py
class GMFSS():
    def __init__(self, interpolation_factor, half, width, height, UHD):

        self.width = width
        self.height = height
        self.half = half
        self.interpolation_factor = interpolation_factor
        self.UHD = UHD

        # Yoinked from rife, needs further testing if these are the optimal
        # FLownet, from what I recall needs 32 paddings
        ph = ((self.height - 1) // 64 + 1) * 64
        pw = ((self.width - 1) // 64 + 1) * 64
        self.padding = (0, pw - self.width, 0, ph - self.height)

        if self.UHD == True:
            self.scale = 0.5
        else:
            self.scale = 1.0

        self.handle_model()

    def handle_model(self):
        # Apparently this can improve performance slightly
        torch.set_float32_matmul_precision("medium")
        
        dir_path = os.path.dirname(os.path.realpath(__file__))
        weights_dir = os.path.join(dir_path, "weights")

        if not os.path.exists(weights_dir):
            os.mkdir(weights_dir)

        url_list = ["https://github.com/NevermindNilas/TAS-Modes-Host/releases/download/main/feat_base.pkl",
                    "https://github.com/NevermindNilas/TAS-Modes-Host/releases/download/main/feat_union.pkl",
                    "https://github.com/NevermindNilas/TAS-Modes-Host/releases/download/main/flownet.pkl",
                    "https://github.com/NevermindNilas/TAS-Modes-Host/releases/download/main/fusionnet_base.pkl",
                    "https://github.com/NevermindNilas/TAS-Modes-Host/releases/download/main/fusionnet_union.pkl",
                    "https://github.com/NevermindNilas/TAS-Modes-Host/releases/download/main/metric_base.pkl",
                    "https://github.com/NevermindNilas/TAS-Modes-Host/releases/download/main/metric_union.pkl",
                    "https://github.com/NevermindNilas/TAS-Modes-Host/releases/download/main/rife.pkl",
                    ]

        download = any(not os.path.exists(os.path.join(weights_dir, url.split('/')[-1])) for url in url_list)

        if download:
            print("Downloading GMFSS models...")
            for url in url_list:
                filename = url.split('/')[-1]
                if not os.path.exists(os.path.join(weights_dir, filename)):
                    print(f"\nDownloading {filename}")
                    wget.download(url, out=weights_dir)
                else:
                    print(f"\n{filename} already exists. Skipping download.")
            print("\n")
            
        model_dir = os.path.join(dir_path, "weights")
        model_type = "union"
        
        self.cuda_available = torch.cuda.is_available()
        
        if not self.cuda_available:
            logging.info(
                "CUDA is not available, using CPU. Expect significant slowdows or no functionality at all. If you have a NVIDIA GPU, please install CUDA and make sure that CUDA_Path is in the environment variables.")
            logging.info(
                "CUDA Installation link: https://developer.nvidia.com/cuda-downloads")
            
        self.device = torch.device("cuda" if self.cuda_available else "cpu")

        torch.set_grad_enabled(False)
        if self.cuda_available:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            if self.half:
                torch.set_default_tensor_type(torch.cuda.HalfTensor)

        from .model.GMFSS import GMFSS as Model

        self.model = Model(model_dir, model_type, self.scale, ensemble=False)
        self.model.eval().to(self.device, memory_format=torch.channels_last)

        self.dtype = torch.float
        if self.cuda_available and self.half:
            self.model.half()
            self.dtype = torch.half
                
              
    @torch.inference_mode()
    def make_inference(self, n, ensemble=False):
        timestep = torch.tensor((n+1) * 1. / (self.interpolation_factor+1), dtype=self.dtype, device=self.device)
        output = self.model(self.I0, self.I1, timestep)
        output = (((output[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
        
        return output[:self.height, :self.width, :]

    @torch.inference_mode()
    def pad_image(self, img):
        img = F.pad(img, self.padding)
        return img

    @torch.inference_mode()
    def run(self, I0, I1):
        self.I0 = torch.from_numpy(np.transpose(I0, (2, 0, 1))).to(
            self.device, non_blocking=True).unsqueeze(0).float() / 255.

        self.I1 = torch.from_numpy(np.transpose(I1, (2, 0, 1))).to(
            self.device, non_blocking=True).unsqueeze(0).float() / 255.

        if self.cuda_available and self.half:
            self.I0 = self.I0.half()
            self.I1 = self.I1.half()

        if self.padding != (0, 0, 0, 0):
            self.I0 = self.pad_image(self.I0)
            self.I1 = self.pad_image(self.I1)