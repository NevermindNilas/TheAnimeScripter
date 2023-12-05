import os
import requests
from tqdm import tqdm

"""
After Effects doesn't like dynamically loading models, so we need to download them before the script is ran.
"""
def handle_swinir():
    pbar = tqdm(total=5, desc="Downloading SwinIR models", unit="", colour="green")

    if not os.path.exists("swinir/weights"):
        os.makedirs("swinir/weights")

    for scale in [2, 4]:
        model_type = {
            'small': f'002_lightweightSR_DIV2K_s64w8_SwinIR-S_x{scale}.pth',
            'medium': f'001_classicalSR_DF2K_s64w8_SwinIR-M_x{scale}.pth',
            'large': f'003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth'
        }

        for kind_model in ['small', 'medium', 'large']:
            filename = model_type[kind_model]
            if not os.path.exists(os.path.join(os.path.abspath("swinir/weights"), filename)):
                url = f"https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/{filename}"
                response = requests.get(url)
                try:
                    if response.status_code == 200:
                        with open(os.path.join("swinir/weights", filename), "wb") as file:
                            file.write(response.content)
                except:
                    raise Exception(f"Could not download {filename}")
                
            pbar.update(1)
    pbar.close()

def handle_cugan():
    pbar = tqdm(total=16, desc="Downloading Cugan models", unit="", colour="green")
    if not os.path.exists("cugan/weights"):
        os.makedirs("cugan/weights")

    for model_type in ["shufflecugan", "cugan"]:
        if model_type == "shufflecugan":
            filename = "sudo_shuffle_cugan_9.584.969.pth"
            if not os.path.exists(os.path.join(os.path.abspath("cugan/weights"), filename)):
                url = f"https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/{filename}"
                response = requests.get(url)
                try:
                    if response.status_code == 200:
                        with open(os.path.join("cugan/weights", filename), "wb") as file:
                            file.write(response.content)
                except:
                    raise Exception(f"Could not download {filename}")
            pbar.update(1)
        else:
            # Removed pro models for now
            for scale in [2, 3, 4]:
                for kind_model in ["no-denoise", "conservative", "denoise1x", "denoise2x", "denoise3x"]:
                    model_path_prefix = "cugan"
                    model_path_suffix = "-latest"
                    model_path_middle = f"up{scale}x"
                    filename = f"{model_path_prefix}_{model_path_middle}{model_path_suffix}-{kind_model}.pth"
    
                    if not os.path.exists(os.path.join(os.path.abspath("cugan/weights"), filename)):
                        url = f"https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/{filename}"
                        response = requests.get(url)
                        try:
                            if response.status_code == 200:
                                with open(os.path.join("cugan/weights", filename), "wb") as file:
                                    file.write(response.content)
                        except:
                            raise Exception(f"Could not download {filename}")
                    pbar.update(1)
    
    pbar.close()

def handle_compact():
    pbar = tqdm(total=2, desc="Downloading Compact models", unit="", colour="green")

    if not os.path.exists("compact/weights"):
        os.makedirs("compact/weights")

    for model_type in ["compact", "ultracompact"]:
        if model_type == "compact":
            filename = "2x_Bubble_AnimeScale_Compact_v1.pth"
            if not os.path.exists(os.path.join(os.path.abspath("src/compact/weights"), filename)):
                url = f"https://github.com/Bubblemint864/AI-Models/releases/download/2x_Bubble_AnimeScale_Compact_v1/{filename}"
                response = requests.get(url)
                if response.status_code == 200:
                    with open(os.path.join("compact/weights", filename), "wb") as file:
                        file.write(response.content)
                else:
                    raise Exception(f"Could not download {filename}")
                        
        elif model_type == "ultracompact":
                filename = "sudo_UltraCompact_2x_1.121.175_G.pth"
                if not os.path.exists(os.path.join(os.path.abspath("src/compact/weights"), filename)):
                    url = f"https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/{filename}"
                    response = requests.get(url)
                    try:
                        if response.status_code == 200:
                            with open(os.path.join("compact/weights", filename), "wb") as file:
                                file.write(response.content)
                    except:
                        raise Exception(f"Could not download {filename}")
        pbar.update(1)
    pbar.close()
    
if __name__ == "__main__":
    handle_swinir()
    handle_cugan()
    handle_compact()

"""
# Another attempt at downloading models, maybe it will be used in the future, idk.
import os
import requests
from tqdm import tqdm

def download_file(url, filename, weights_dir):
    if not os.path.exists(os.path.join(os.path.abspath(weights_dir), filename)):
        response = requests.get(url)
        if response.status_code == 200:
            with open(os.path.join(weights_dir, filename), "wb") as file:
                file.write(response.content)
        else:
            raise Exception(f"Could not download {filename}")

def handle_model(model_name, total, model_files):
    pbar = tqdm(total=total, desc=f"Downloading {model_name} models", unit="", colour="green")
    weights_dir = f"src/{model_name}/weights"
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir, exist_ok=True)

    for filename, url in model_files:
        download_file(url, filename, weights_dir)
        pbar.update(1)
    pbar.close()

if __name__ == "__main__":

    swinir_files = [
        (f'002_lightweightSR_DIV2K_s64w8_SwinIR-S_x{scale}.pth', f"https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x{scale}.pth") for scale in [2, 4]
    ] + [
        (f'001_classicalSR_DF2K_s64w8_SwinIR-M_x{scale}.pth', f"https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x{scale}.pth") for scale in [2, 4]
    ] + [
        ('003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth', "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth")
    ]

    cugan_files = [
        ("sudo_shuffle_cugan_9.584.969.pth", "https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/sudo_shuffle_cugan_9.584.969.pth")
    ] + [
        (f"cugan_up{scale}x-latest-{kind_model}.pth", f"https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/cugan_up{scale}x-latest-{kind_model}.pth") for scale in [2, 3, 4] for kind_model in ["no-denoise", "conservative", "denoise1x", "denoise2x", "denoise3x"]
    ] + [
        (f"cugan_pro_up{scale}x-{kind_model}.pth", f"https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/cugan_pro_up{scale}x-{kind_model}.pth") for scale in [2, 3, 4] for kind_model in ["no-denoise", "conservative", "denoise1x", "denoise2x", "denoise3x"]
    ]

    compact_files = [
        ("2x_Bubble_AnimeScale_Compact_v1.pth", "https://github.com/Bubblemint864/AI-Models/releases/download/2x_Bubble_AnimeScale_Compact_v1/2x_Bubble_AnimeScale_Compact_v1.pth"),
        ("sudo_UltraCompact_2x_1.121.175_G.pth", "https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/sudo_UltraCompact_2x_1.121.175_G.pth")
    ]

    handle_model("swinir", len(swinir_files), swinir_files)
    handle_model("cugan", len(cugan_files), cugan_files)
    handle_model("compact", len(compact_files), compact_files)
    
"""
    

        
