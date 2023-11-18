import argparse
import os
import sys
import subprocess
from inference import process_video_rife
import torch
from src.rife.RIFE_HDv3 import Model
import warnings


warnings.filterwarnings("ignore")

def main(scale, half, model_type):
    if "rife" in model_type:
        model, device = handle_rife_models(half)
    elif "cugan" in model_type:
        pass
    elif "dedup" in model_type:
        pass

    input_path = os.path.join(".", "input")
    output_path = os.path.join(".", "output")

    os.makedirs(input_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    video_files = [
        f
        for f in os.listdir(input_path)
        if f.endswith((".mp4", ".avi", ".mkv", ".mov"))
    ]
    video_files.sort()

    if not video_files:
        sys.exit("No videos found in the input folder")

    for i, video_file in enumerate(video_files):
        output = os.path.splitext(video_file)[0] + ".mp4"
        output_path = os.path.join(output_path, output)
        video_file = os.path.join(input_path, video_file)
        if "rife" in model_type:
            process_video_rife(video_file, output_path, model, scale, device, half)
        elif "cugan" in model_type:
            pass
        elif "dedup" in model_type:
            pass

def handle_rife_models(half):
    filename = "flownet.pkl"
    for root, dirs, files in os.walk(os.path.dirname(os.path.realpath(__file__))):
        if filename in files:
            flownet_path = root

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        if half:
            torch.set_default_tensor_type(torch.cuda.HalfTensor)

    model = Model()
    if not hasattr(model, "version"):
        model.version = 0
    model.load_model(flownet_path, -1)
    model.eval()
    model.device()

    return model, device


def handle_cugan_models(model_type, half):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Contact Sheet Generator")
    parser.add_argument(
        "-model_type",
        required=False,
        type=str,
        help="rife",
        default="rife",
        action="store",
    )
    parser.add_argument("-half", type=str, help="", default="True", action="store")
    parser.add_argument("-scale", type=int, help="", default=2, action="store")
    args = parser.parse_args()
    
    main(args.scale, args.half, args.model_type)
