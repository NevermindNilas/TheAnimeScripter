import os
import requests
from tqdm import tqdm
from src.compact.compact import Compact
from src.cugan.cugan import Cugan
from src.swinir.swinir import Swin

"""
After Effects doesn't like dynamically loading models, so we need to download them before the script is ran.
"""
def handle_swinir():
    pbar = tqdm(total=5, desc="Downloading SwinIR models", unit="", colour="green")
    for scale in [2, 4]:
        model_type = {
            'small': f'002_lightweightSR_DIV2K_s64w8_SwinIR-S_x{scale}.pth',
            'medium': f'001_classicalSR_DF2K_s64w8_SwinIR-M_x{scale}.pth',
            'large': f'003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth'
        }
        if not os.path.exists("src/swinir/weights"):
            os.makedirs("src/swinir/weights")

        for kind_model in ['small', 'medium', 'large']:
            filename = model_type[kind_model]
            if not os.path.exists(os.path.join(os.path.abspath("src/swinir/weights"), filename)):
                url = f"https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/{filename}"
                response = requests.get(url)
                if response.status_code == 200:
                    with open(os.path.join("src/swinir/weights", filename), "wb") as file:
                        file.write(response.content)
            pbar.update(1)
    pbar.close()
def handle_cugan():
    pbar = tqdm(total=16, desc="Downloading Cugan models", unit="", colour="green")
    if not os.path.exists("src/cugan/weights"):
        os.makedirs("src/cugan/weights", exist_ok=True)

    for model_type in ["shufflecugan", "cugan"]:
        if model_type == "shufflecugan":
            filename = "sudo_shuffle_cugan_9.584.969.pth"
            download_cugan(filename)
            pbar.update(1)
        else:
            for scale in [2, 3, 4]:
                for kind_model in ["no-denoise", "conservative", "denoise1x", "denoise2x", "denoise3x"]:
                    for pro in [False, True]:
                        model_path_prefix = "cugan_pro" if pro else "cugan"
                        model_path_suffix = "-latest" if not pro else ""
                        model_path_middle = f"up{scale}x"
                        filename = f"{model_path_prefix}_{model_path_middle}{model_path_suffix}-{kind_model}.pth"
        
                        if not os.path.exists(os.path.join(os.path.abspath("src/cugan/weights"), filename)):
                            download_cugan(filename)
                    pbar.update(1)
    pbar.close()
def download_cugan(filename):
    if not os.path.exists(os.path.join(os.path.abspath("src/cugan/weights"), filename)):
        url = f"https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/{filename}"
        response = requests.get(url)
        if response.status_code == 200:
            with open(os.path.join("src/cugan/weights", filename), "wb") as file:
                file.write(response.content)

def handle_compact():
    pbar = tqdm(total=2, desc="Downloading Compact models", unit="", colour="green")
    if not os.path.exists("src/compact/weights"):
        os.mkdir("src/compact/weights")

    for model_type in ["compact", "ultracompact"]:
        if model_type == "compact":
            filename = "2x_Bubble_AnimeScale_Compact_v1.pth"
            if not os.path.exists(os.path.join(os.path.abspath("src/compact/weights"), filename)):
                url = f"https://github.com/Bubblemint864/AI-Models/releases/download/2x_Bubble_AnimeScale_Compact_v1/{filename}"
                response = requests.get(url)
                if response.status_code == 200:
                    with open(os.path.join("src/compact/weights", filename), "wb") as file:
                        file.write(response.content)
                        
        elif model_type == "ultracompact":
                filename = "sudo_UltraCompact_2x_1.121.175_G.pth"
                if not os.path.exists(os.path.join(os.path.abspath("src/compact/weights"), filename)):
                    url = f"https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/{filename}"
                    response = requests.get(url)
                    if response.status_code == 200:
                        with open(os.path.join("src/compact/weights", filename), "wb") as file:
                            file.write(response.content)
        pbar.update(1)
    pbar.close()
if __name__ == "__main__":
    handle_swinir()
    handle_cugan()
    handle_compact()
    

        
