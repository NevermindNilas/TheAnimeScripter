import os
import requests
import torch
from torch.nn import functional as F
from torch import nn as nn
import _thread
from queue import Queue
import time
import cv2
import threading
import numpy as np
from tqdm import tqdm
from moviepy.editor import VideoFileClip
import sys

from train import AnimeSegmentation

@torch.inference_mode()
class Segment():
    def __init__(self, video, output, kind_model, nt, half, w, h, tot_frame):
        self.video = video
        self.output = output
        self.kind_model = kind_model
        self.nt = nt
        self.half = half
        self.w = w
        self.h = h
        self.tot_frame = tot_frame
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lock = threading.Lock()
        
        self.handle_models()
        self._initialize()
        
        threads = []
        for _ in range(self.nt):
            thread = SegmentMT(self.device, self.model, self.nt, self.half, self.read_buffer, self.write_buffer, self.lock)
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()
        
        while threading.active_count() > 1 and self.write.qsize() > 0:
            time.sleep(0.1)
        
        self.pbar.close()
        self.vid_out.release()
        self.videogen.reader.close()
        
        def handle_models(self):
            if not os.path.exists("src/segment/models"):
                os.mkdir("src/segment/models")
                

            if not os.path.exists(os.path.join(os.path.abspath("src/segment/models"), "isnetis.ckpt" )):
                print("Downloading segment checkpoint...")
                url = f"https://huggingface.co/skytnt/anime-seg/resolve/main/isnetis.ckpt?download=true"
                response = requests.get(url)
                if response.status_code == 200:
                    with open(os.path.join("src/segment/models", "isnetic.ckpt"), 'wb') as f:
                        f.write(response.content)
            
            models = [
                "isnet_is",
                "isnet",
                "u2net",
                "u2netl",
                "modnet",
                "inspyrnet_res",
                "inspyrnet_swin"
            ]
            
            if kind_model not in models:
                print("Invalid model name, please choose between:")
                print(models)
                sys.exit()
                
            if self.w >= 1024 or self.h >= 1024 and self.kind_model != "isnet_is" or "isnet":
                print("For resolutions above 1024, it is recommended to use isnet or isnet_is")
            elif 384 < self.w < 1024 or 384 < self.h < 1024 and self.kind_model != "u2net" or "u2netl" or "modnet":
                print("For resolution between 384 and 1024, it is recommended to use u2net, u2netl or modnet")
            elif self.w <= 384 or self.h <= 384 and self.kind_model != "inspyrnet_res" or "inspyrnet_swin":
                print("For resolutions below 384, it is recommended to use inspyrnet_res or inspyrnet_swin")
            
        def _initialize(self):
            
            self.model = AnimeSegmentation(self.kind_model, self.w) if self.w > self.h else AnimeSegmentation(self.kind_model, self.h)

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if torch.cuda.is_available():
                torch.backends.cudnn.enabled = True
                torch.backends.cudnn.benchmark = True
                if self.half:
                    torch.set_default_tensor_type(torch.cuda.HalfTensor)
            
            self.pbar = tqdm(total=self.tot_frame)
            self.write_buffer = Queue(maxsize=500)
            self.read_buffer = Queue(maxsize=500)
            
            self.videogen = VideoFileClip(self.video_file)
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            self.vid_out = cv2.VideoWriter(self.output, fourcc, self.videogen.fps, (w, h))
            self.frames = self.videogen.iter_frames()
            
            _thread.start_new_thread(self._build_read_buffer, ())
            _thread.start_new_thread(self._clear_write_buffer, ())
        
        def _clear_write_buffer(self):
            while True:
                frame = self.write_buffer.get()
                if frame is None:
                    break
                self.pbar.update(1)
                self.vid_out.write(frame[:, :, ::-1])
            
        def _build_read_buffer(self):
            try:
                for frame in self.frames:
                    self.read_buffer.put(frame)
            except:
                pass
            for _ in range(self.nt):
                self.read_buffer.put(None)
                
class SegmentMT(threading.Thread):
    def __init__(self, device, model, nt, half, read_buffer, write_buffer, lock):
        threading.Thread.__init__(self)
        self.device = device
        self.model = model
        self.nt = nt
        self.half = half
        self.read_buffer = read_buffer
        self.write_buffer = write_buffer
        self.lock = lock
    
    def inference(self, frame):
        if self.half:
            frame = frame.half()
        with torch.no_grad():
            frame = frame.to(self.device)
            frame = self.model(frame)
            frame = frame.float()
            frame = frame.cpu()
    
    def process_frame(self, frame):
        self.inference(frame)
        self.write_buffer.put(frame)