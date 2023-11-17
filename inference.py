import argparse
import os
import sys
import time
from threading import Thread
import cv2
import numpy as np
import requests
import tqdm

def main(half, model_type, height, width):
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
'''
class VideoDecodeStream:
    def __init__(self, video_file, width, height):
        self.vcap = cv2.VideoCapture(video_file)
        self.width = width
        self.height = height
        self.decode_buffer = []
        self.grabbed , self.frame = self.vcap.read()
        self.frame = cv2.resize(self.frame, (self.width, self.height))
        self.decode_buffer.append(self.frame)
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True 

    def start(self):
        self.stopped = False
        self.thread.start() 

    def update(self):
        while not self.stopped:
            grabbed, frame = self.vcap.read()
            if not grabbed:
                print('The video buffering has been finished')
                self.stopped = True
                break
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
            self.decode_buffer.append(frame)
        self.vcap.release()

    def read(self):
        return self.decode_buffer.pop(0) if self.decode_buffer else None

    def stop(self):
        self.stopped = True 

    def get_fps(self):
        return self.vcap.get(cv2.CAP_PROP_FPS)

    def get_frame_count(self):
        return self.vcap.get(cv2.CAP_PROP_FRAME_COUNT)

    def get_fourcc(self):
        return cv2.VideoWriter_fourcc(*'mp4v')'''

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
    parser = argparse.ArgumentParser(description="Contact Sheet Generator")
    parser.add_argument('-width', type=int, help="Width of the corresponding output, must be a multiple of 32", default=1280)
    parser.add_argument("-height", type=int, help="Height of the corresponding output, must be a multiple of 32", default=736)
    parser.add_argument('-model_type', required=False, type=str, help="", default="4.9", action="store")
    parser.add_argument('-half', type=str, help="", default="True", action="store")
    args = parser.parse_args()
    
    url = "https://github.com/HolyWu/vs-rife/releases/download/model/"
    models = [
        "flownet_v4.0",
        "flownet_v4.1",
        "flownet_v4.2",
        "flownet_v4.3",
        "flownet_v4.4",
        "flownet_v4.5",
        "flownet_v4.6",
        "flownet_v4.7",
        "flownet_v4.8",
        "flownet_v4.9",
        "flownet_v4.10",
        "flownet_v4.11",
    ]
    for model in models:
        download_model(url + model + ".pkl")
        
    main(args.half, args.model_type, args.height, args.width)