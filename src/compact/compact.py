import os
import requests
import numpy as np
import torch
from torch.nn import functional as F
from torch import nn as nn
import _thread
from tqdm import tqdm
from queue import Queue
import threading
import cv2
import time

from moviepy.editor import VideoFileClip
from .srvgg_arch import SRVGGNetCompact

@torch.inference_mode()
class Compact():
    def __init__(self,video_file, output, multi, half, w, h, nt, tot_frame, model_type):
        self.video_file = video_file
        self.output = output
        self.scale = multi
        self.half = half
        self.w = w
        self.h = h
        self.nt = nt
        self.tot_frame = tot_frame
        self.model_type = model_type
        self.lock = threading.Lock()
        
        self.handle_models()
        self._initialize()
        
        threads = []
        for _ in range(self.nt):
            thread = CompactMT(self.device, self.model, self.nt, self.half, self.read_buffer, self.write_buffer, self.lock)
            thread.start()
            threads.append(thread)
        
        for thread in threads:
            thread.join()
        
        while threading.active_count() > 1 and self.write_buffer.qsize() > 0:
            time.sleep(0.1)
            
        self.pbar.close()
        self.vid_out.release()
        self.videogen.reader.close()
    
    def handle_models(self):
        if not os.path.exists("src/compact/weights"):
            os.mkdir("src/compact/weights")

        if self.model_type == "compact":
            self.filename = "2x_Bubble_AnimeScale_Compact_v1.pth"
            if not os.path.exists(os.path.join(os.path.abspath("src/compact/weights"), self.filename)):
                print("Downloading Compact model...")
                url = f"https://github.com/Bubblemint864/AI-Models/releases/download/2x_Bubble_AnimeScale_Compact_v1/{self.filename}"
                response = requests.get(url)
                if response.status_code == 200:
                    with open(os.path.join("src/compact/weights", self.filename), "wb") as file:
                        file.write(response.content)
        else:
            self.model_type = "ultracompact"
            self.filename = "sudo_shuffle_cugan_9.584.969.pth"
            if not os.path.exists(os.path.join(os.path.abspath("src/compact/weights"), self.filename)):
                print("Downloading UltraCompact model...")
                url = f"https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/sudo_shuffle_cugan_9.584.969.pth"
                response = requests.get(url)
                if response.status_code == 200:
                    with open(os.path.join("src/compact/weights", self.filename), "wb") as file:
                        file.write(response.content)
        
    def _initialize(self):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            if self.half:
                torch.set_default_tensor_type(torch.cuda.HalfTensor)
                
        model_path = os.path.abspath(os.path.join("src/compact/weights", self.filename))
        self.model = SRVGGNetCompact(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_conv=16,
                upscale=self.scale,
                act_type="prelu",
                )

        self.model.load_state_dict(torch.load(model_path, map_location="cpu")["params"])
        self.model.eval().cuda() if torch.cuda.is_available() else self.model.eval()
        
        if self.half:
            self.model.half()
            
        self.pbar = tqdm(total=self.tot_frame)
        self.write_buffer = Queue(maxsize=500)
        self.read_buffer = Queue(maxsize=500)
        
        self.videogen = VideoFileClip(self.video_file)
        w_new, h_new = int(self.w * self.scale), int(self.h * self.scale)
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        self.vid_out = cv2.VideoWriter(self.output, fourcc, self.videogen.fps, (w_new, h_new))
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
            
class CompactMT(threading.Thread):
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
            return self.model(frame)
        
    def process_frame(self, frame):
        frame = frame.astype(np.float32) / 255.0 
        frame = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0)
        if torch.cuda.is_available():
            frame = frame.cuda()
        frame = self.inference(frame)
        frame = frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
        frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
        return frame
    
    def run(self):  
        while True:
            frame = self.read_buffer.get()
            if frame is None:
                break
            result = self.process_frame(frame)
            with self.lock:
                self.write_buffer.put(result)