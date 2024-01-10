import os
import requests
import subprocess
import torch

class GMFSS():
    def __init__(self, width, height, half, interpolation_factor):
        
        self.width = width
        self.height = height
        self.half = half
        self.interpolation_factor = interpolation_factor
        
        # Yoinked from rife, needs further testing if these are the optimal
        # FLownet, from what I recall needs 32 paddings
        ph = ((self.height - 1) // 64 + 1) * 64
        pw = ((self.width - 1) // 64 + 1) * 64
        self.padding = (0, pw - self.width, 0, ph - self.height)
        
        self.handle_model()
        
    def handle_model(self):
        
        # Check if the model is already downloaded
        dir_path = os.path.dirname(os.path.realpath(__file__))
        
        download = False
        if not os.path.exists(os.path.join(dir_path, "weights")):
            os.mkdir(os.path.join(dir_path, "weights"))
            download = True
            
        if download:
            print("Downloading GMFSS weights...")
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
                                        
            
                        