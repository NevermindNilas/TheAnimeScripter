import argparse
import os
import sys
import subprocess
from inference import process_video
import torch
from .src.rife.rife import RIFE
#from .src.rife.rife_arch import IFNet

def main(scale, half, model_type, height, width):
    
    if 'rife' in model_type:
        model = handle_rife_models(model_type, scale, half)
    elif 'cugan' in model_type:
        pass
    elif 'dedup' in model_type:
        pass
    
    input_path = os.path.join('.', "input")
    output_path = os.path.join('.', "output")
    
    os.makedirs(input_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
        
    video_files = [f for f in os.listdir(input_path) if f.endswith(('.mp4', '.avi', '.mkv', '.mov'))]
    video_files.sort()

    if not video_files:
        sys.exit("No videos found in the input folder")
    
    for i,video_file in enumerate(video_files):
        output = os.path.splitext(video_file)[0] + ".mp4"
        output_path = os.path.join(output_path, output)
        video_file = os.path.join(input_path, video_file)
        process_video(video_file, output_path, model, height, width)
    
def handle_rife_models(model_type, scale, half):
    models = [
        "rife40",
        "rife41",
        "rife42",
        "rife43",
        "rife44",
        "rife45",
        "rife46",
        "rife47",
        "rife48",
        "rife49"
    ]
    if model_type not in models:
        sys.exit(f"Model type {model_type} not found. Please choose from {models}")
        
    model_file = f'./src/rife/models/{model_type}.pth'
    if not os.path.exists(model_file):
        for root, dirs, files in os.walk('.'):
            if 'download_rife_models.py' in files:
                script_path = os.path.join(root, 'download_rife_models.py')
                subprocess.run(['python', script_path, '-model_type', model_type], check=True)
                break
    
    match model_type:
        case "rife40" | "rife41":
            arch_ver = "4.0"
        case "rife42":
            arch_ver = "4.2"
        case "rife43" | "rife44":
            arch_ver = "4.3"
        case "rife45":
            arch_ver = "4.5"
        case "rife46":
            arch_ver = "4.6"
        case _:
            sys.exit(f"Model type {model_type} not found. Please choose from {models}")
    
    model = RIFE
        
    return model
    
def handle_cugan_models(model_type, half):
    pass
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Contact Sheet Generator")
    parser.add_argument('-width', type=int, help="", default=1280)
    parser.add_argument("-height", type=int, help="", default=720)
    parser.add_argument('-model_type', required=False, type=str, help="", default="rife49", action="store")
    parser.add_argument('-half', type=str, help="", default="True", action="store")
    parser.add_argument('-scale', type=int, help="", default=1, action="store")
    args = parser.parse_args()

    main(args.scale, args.half, args.model_type, args.height, args.width)