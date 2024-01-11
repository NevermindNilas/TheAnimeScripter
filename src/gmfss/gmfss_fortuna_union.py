import os
import requests
import torch
import numpy as np

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

        torch.set_float32_matmul_precision("medium")
        
        # Check if the model is already downloaded
        dir_path = os.path.dirname(os.path.realpath(__file__))

        download = False
        if not os.path.exists(os.path.join(dir_path, "weights")):
            os.mkdir(os.path.join(dir_path, "weights"))
            download = True

        if download:
            print("Downloading GMFSS models...")
            url_list = ["https://github.com/NevermindNilas/TAS-Modes-Host/releases/download/main/feat_base.pkl",
                        "https://github.com/NevermindNilas/TAS-Modes-Host/releases/download/main/feat_union.pkl",
                        "https://github.com/NevermindNilas/TAS-Modes-Host/releases/download/main/flownet.pkl",
                        "https://github.com/NevermindNilas/TAS-Modes-Host/releases/download/main/fusionnet_base.pkl",
                        "https://github.com/NevermindNilas/TAS-Modes-Host/releases/download/main/fusionnet_union.pkl",
                        "https://github.com/NevermindNilas/TAS-Modes-Host/releases/download/main/metric_base.pkl",
                        "https://github.com/NevermindNilas/TAS-Modes-Host/releases/download/main/metric_union.pkl",
                        "https://github.com/NevermindNilas/TAS-Modes-Host/releases/download/main/rife.pkl",
                        ]

            for url in url_list:
                response = requests.get(url)
                if response.status_code == 200:
                    with open(os.path.join(dir_path, "weights", url.split("/")[-1]), "wb") as file:
                        file.write(response.content)
                else:
                    print(f"Failed to download {url}")
                    return
                
        model_dir = os.path.join(dir_path, "weights")
        model_type = "union"
        
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Doing a torch cuda check is rather expensive on start-up times so I just decided to keep it simple
        self.cuda_available = False
        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            self.cuda_available = True
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            if self.half:
                torch.set_default_tensor_type(torch.cuda.HalfTensor)

        from .model.GMFSS import GMFSS as Model

        self.model = Model(model_dir, model_type, self.scale, ensemble=False)
        self.model.eval().to(self.device, memory_format=torch.channels_last)

        if self.cuda_available:
            if self.half:
                self.model.half()
                
        self.dtype = torch.half if self.half else torch.float  
              
    def pad_image(self, img):
        img = F.pad(img, self.padding)
        if self.cuda_available and self.half:
            img = img.half()
        return img

    def make_inference(self, I0, I1, n):
        res = []
        for i in range(n-1):
            timestep = torch.tensor((i+1) * 1. / (n+1), dtype=self.dtype, device=self.device)
            res.append(self.model(I0, I1, timestep))
        return res
    
    @torch.inference_mode()
    def run(self, I0, I1):
        buffer = []
        I0 = torch.from_numpy(np.transpose(I0, (2, 0, 1))).to(
            self.device, non_blocking=True).unsqueeze(0).float() / 255.
        I1 = torch.from_numpy(np.transpose(I1, (2, 0, 1))).to(
            self.device, non_blocking=True).unsqueeze(0).float() / 255.

        I0 = self.pad_image(I0)
        I1 = self.pad_image(I1)
        
        output = self.make_inference(I0, I1, self.interpolation_factor)

        for mid in output:
            mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
            buffer.append(mid[:self.height, :self.width, :])

        return buffer
