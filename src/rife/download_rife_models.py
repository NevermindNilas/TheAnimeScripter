import os
import requests
from tqdm import tqdm
import argparse

# from https://github.com/HolyWu/vs-rife/blob/master/vsrife/__main__.py
# thanks to sudo for the models

def download_model(url: str) -> None:
    filename = url.split("/")[-1]
    r = requests.get(url, stream=True)
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "models", filename), "wb") as f:
        with tqdm(
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            miniters=1,
            desc=filename,
            total=int(r.headers.get("content-length", 0)),
        ) as pbar:
            for chunk in r.iter_content(chunk_size=4096):
                f.write(chunk)
                pbar.update(len(chunk))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download RIFE model")
    parser.add_argument('-model_type', required=True, type=str, help="Model type to download")
    args = parser.parse_args()

    base_url = "https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/"
    download_model(base_url + args.model_type + ".pth")