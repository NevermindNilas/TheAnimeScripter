import os
import wget
import torch
import numpy as np
import logging
import zipfile

from torch.nn import functional as F

# from: https://github.com/HolyWu/vs-gmfss_fortuna/blob/master/vsgmfss_fortuna/__init__.py


class GMFSS():
    def __init__(self, interpolation_factor, half, width, height, UHD, ensemble=False):

        self.width = width
        self.height = height
        self.half = half
        self.interpolation_factor = interpolation_factor
        self.UHD = UHD
        self.ensemble = ensemble

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
        torch.set_float32_matmul_precision("medium")

        dir_path = os.path.dirname(os.path.realpath(__file__))
        weights_dir = os.path.join(dir_path, "weights")
        
        if not os.path.exists(weights_dir):
            os.makedirs(weights_dir)

        files_to_check = [
            "feat_base.pkl",
            "feat_union.pkl",
            "flownet.pkl",
            "fusionnet_base.pkl",
            "fusionnet_union.pkl",
            "metric_base.pkl",
            "metric_union.pkl",
            "rife.pkl",
        ]

        all_files_exist = all(os.path.exists(os.path.join(
            weights_dir, filename)) for filename in files_to_check)

        if not all_files_exist:
            zip_url = "https://github.com/NevermindNilas/TAS-Modes-Host/releases/download/main/gmfss-fortuna-union.zip"
            zip_filename = zip_url.split('/')[-1]
            zip_filepath = os.path.join(weights_dir, zip_filename)

            print(f"\nDownloading {zip_filename}")
            logging.info(
                f"Downloading {zip_filename}")
            wget.download(zip_url, out=weights_dir)

            # Extract the zip file  
            with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
                print(f"\nExtracting {zip_filename}")
                zip_ref.extractall(weights_dir)
                
            os.remove(zip_filepath)
        else:
            logging.info("Weights already exist, skipping download")

        model_type = "union"

        self.cuda_available = torch.cuda.is_available()

        if not self.cuda_available:
            print(
                "CUDA is not available, using CPU. Expect significant slowdows or no functionality at all. If you have a NVIDIA GPU, please install CUDA and make sure that CUDA_Path is in the environment variables.")
            print(
                "CUDA Installation link: https://developer.nvidia.com/cuda-downloads")
            
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

        self.model = Model(weights_dir, model_type,
                           self.scale, ensemble=self.ensemble)
        self.model.eval().to(self.device, memory_format=torch.channels_last)

        self.dtype = torch.float
        if self.cuda_available and self.half:
            self.model.half()
            self.dtype = torch.half

    @torch.inference_mode()
    def make_inference(self, n):
        timestep = torch.tensor(
            (n+1) * 1. / (self.interpolation_factor+1), dtype=self.dtype, device=self.device)
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
