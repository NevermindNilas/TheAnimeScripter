import os
import requests
import torch
import numpy as np

from torch.nn import functional as F

# FP16 doesn't work for now, fix coming soon
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

        # Check if the model is already downloaded
        dir_path = os.path.dirname(os.path.realpath(__file__))

        download = False
        if not os.path.exists(os.path.join(dir_path, "weights")):
            os.mkdir(os.path.join(dir_path, "weights"))
            download = True

        if download:
            print("Downloading GMFSS models...")
            url_list = ["https://github.com/NevermindNilas/TAS-Modes-Host/releases/download/main/feat.pkl",
                        "https://github.com/NevermindNilas/TAS-Modes-Host/releases/download/main/flownet.pkl",
                        "https://github.com/NevermindNilas/TAS-Modes-Host/releases/download/main/fusionnet.pkl",
                        "https://github.com/NevermindNilas/TAS-Modes-Host/releases/download/main/metric.pkl",
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

        from .model.GMFSS_infer_u import Model

        self.model = Model()
        self.model.load_model(os.path.join(dir_path, "weights"), -1)
        self.model.eval()
        
        if self.cuda_available:
            if self.half:
                self.model.half()
                
        self.model.device()
        
    def pad_image(self, img):
        img = F.pad(img, self.padding)
        if self.cuda_available and self.half:
            img = img.half()
        return img

    def make_inference(self, I0, I1, reuse_things, n):
        res = []
        for i in range(n):
            res.append(self.model.inference(
                I0, I1, reuse_things, (i+1) * 1. / (n+1)))
        return res

    def run(self, I0, I1):
        buffer = []
        I0 = torch.from_numpy(np.transpose(I0, (2, 0, 1))).to(
            self.device, non_blocking=True).unsqueeze(0).float() / 255.
        I1 = torch.from_numpy(np.transpose(I1, (2, 0, 1))).to(
            self.device, non_blocking=True).unsqueeze(0).float() / 255.

        I0 = self.pad_image(I0)
        I1 = self.pad_image(I1)
        
        reuse_things = self.model.reuse(I0, I1, self.scale)
        output = self.make_inference(I0, I1, reuse_things, self.interpolation_factor-1)

        for mid in output:
            mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
            buffer.append(mid[:self.height, :self.width, :])

        return buffer
