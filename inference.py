import argparse
import os
import sys
import time
from threading import Thread
import cv2
import numpy as np
import requests
from download_models import download_model
import tqdm

def main(half, model_type, height, width):
    
    download_model(model_type)
    input_path = os.path.join('.', "input")
    output_path = os.path.join('.', "output")
    os.makedirs(input_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    
    if width is None or height is None:
        sys.exit("You must specify both width and height")
    elif width % 32 != 0:
        #print("The width is not divisible by 32, rounding up to the nearest multiple:", width)
        width = (width // 32 + 1) * 32
    elif height % 32 != 0:
        height = (height // 32 + 1) * 32
        #print("The height is not divisible by 32, rounding up to the nearest multiple:", height)
        
    video_files = [f for f in os.listdir(input_path) if f.endswith(('.mp4', '.avi', '.mkv', '.mov'))]
    video_files.sort()

    if not video_files:
        sys.exit("No videos found in the input folder")

    for i,video_file in enumerate(video_files):
        output = os.path.splitext(video_file)[0] + ".mp4"
        output_path = os.path.join(output_path, output)
        video_file = os.path.join(input_path, video_file)
        
        print("\n") 
        print("===================================================================")
        print("Processing Video File:", os.path.basename(video_file))
        print("===================================================================")
        print("\n") # Force new line for each video to make it more user readable
        
        load_model(model_type, half)
        process_video(video_file, output_path, width, height, half, model_type)

def process_video(video_file, output_path, width, height, half, model_type):
    pass

def load_model(model_type, half):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Contact Sheet Generator")
    parser.add_argument('-width', type=int, help="Width of the corresponding output, must be a multiple of 32", default=1280)
    parser.add_argument("-height", type=int, help="Height of the corresponding output, must be a multiple of 32", default=736)
    parser.add_argument('-model_type', required=False, type=str, help="", default="4.9", action="store")
    parser.add_argument('-half', type=str, help="", default="True", action="store")
    args = parser.parse_args()
    
    main(args.half, args.model_type, args.height, args.width)