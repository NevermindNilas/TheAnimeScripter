import os
import requests
import numpy as np
import torch
from torch.nn import functional as F
from torch import nn as nn
import _thread
from tqdm import tqdm
from queue import Queue
import sys
import threading
import cv2
import time
from moviepy.editor import VideoFileClip
from .network import SwinIR as SwinIR_model

class Swin():
    def __init__(self, video_file, output, model_type, scale, half, nt, kind_model, tot_frame):
        self.video_file = video_file
        self.output = output
        self.model_type = model_type
        self.scale = scale
        self.half = half
        self.nt = nt
        self.kind_model = kind_model
        self.tot_frame = tot_frame
        self.lock = threading.Lock()
        
        self._initialize()
        
        threads = []
        for _ in range(self.nt):
            thread = SwinMT(self.device, self.model, self.nt, self.half, self.read_buffer, self.write_buffer, self.vid_out, self.lock, self.h, self.w)
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
        
        self.kind_model = self.kind_model.lower()
        if self.kind_model == None:
            sys.exit("Please specify a model type")
            
        elif self.kind_model != "small" and "medium" and "large":
            sys.exit("Invalid kind_model type, please choose from: small, medium or large")
            
        model_type = {
            'small': f'002_lightweightSR_DIV2K_s64w8_SwinIR-S_x{self.scale}.pth',
            'medium': f'001_classicalSR_DF2K_s64w8_SwinIR-M_x{self.scale}.pth',
            'large': f'003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth'
        }

        if self.kind_model == "Large" and self.scale != 4:
            sys.exit("Large model only support scale 4")
        
        model_hyperparams = {'upscale': self.scale, 'in_chans': 3, 'img_size': 64, 'window_size': 8,
                             'img_range': 1., 'mlp_ratio': 2, 'resi_connection': '1conv'}

        if self.kind_model == 'medium':
            self.model = SwinIR_model(depths=[6] * 6, embed_dim=180, num_heads=[6] * 6,
                                 upsampler='pixelshuffle', **model_hyperparams)

        elif self.kind_model == 'small':
            self.model = SwinIR_model(depths=[6] * 4, embed_dim=60, num_heads=[6] * 4,
                                 upsampler='pixelshuffledirect', **model_hyperparams)

        elif self.kind_model == 'large':
            self.model = SwinIR_model(depths=[6] * 6, embed_dim=180, num_heads=[6] * 6,
                                 upsampler='nearest+conv', **model_hyperparams)

        self.filename = model_type[self.kind_model]
        
        if not os.path.exists("src/swinir/weights"):
            os.makedirs("src/swinir/weights")
        
        if not os.path.exists(os.path.join(os.path.abspath("src/swinir/weights"), self.filename)):
            print(f"Downloading SwinIR model...")   
            url = f"https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/{self.filename}"
            response = requests.get(url)
            if response.status_code == 200:
                with open(os.path.join("src/swinir/weights", self.filename), "wb") as file:
                    file.write(response.content)
        
        self.pretrained_weights = torch.load(os.path.join("src/swinir/weights", self.filename))
        
        if self.kind_model == "small":
            self.pretrained_weights = self.pretrained_weights['params']
        elif self.kind_model == "medium":
            self.pretrained_weights = self.pretrained_weights['params']
        elif self.kind_model == "large":
            self.pretrained_weights = self.pretrained_weights['params_ema']
        
        return self.model, self.pretrained_weights
                    
    def _initialize(self):  
        self.handle_models()
        
        self.model.load_state_dict(self.pretrained_weights, strict=True)
        self.model.eval()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if torch.cuda.is_available():
            self.model.cuda()
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            if self.half:
                torch.set_default_tensor_type(torch.cuda.HalfTensor)
                self.model.half()
    
        self.pbar = tqdm(total=self.tot_frame)
        self.write_buffer = Queue(maxsize=500)
        self.read_buffer = Queue(maxsize=500)
        
        self.videogen = VideoFileClip(self.video_file)
        self.w, self.h = self.videogen.size
        self.w_new, self.h_new = int(self.w * self.scale), int(self.h * self.scale)
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        self.vid_out = cv2.VideoWriter(self.output, fourcc, self.videogen.fps, (self.w_new, self.h_new))
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
    
    def make_inference(self, frame):
        if self.half:
            frame = frame.half()
        return self.model(frame)
    
class SwinMT(threading.Thread):
    def __init__(self, device, model, nt, half, read_buffer, write_buffer, vid_out, lock, h, w):
        threading.Thread.__init__(self)
        self.device = device
        self.model = model
        self.nt = nt
        self.half = half
        self.read_buffer = read_buffer
        self.write_buffer = write_buffer
        self.vid_out = vid_out
        self.lock = lock
        self.h = h
        self.w = w
    
    def inference(self, frame):
        if self.half:
            frame = frame.half()
        with torch.no_grad():
            return self.model(frame)
    
    def pad_frame(self, frame):
        frame = torch.cat([frame, torch.flip(frame, [2])], 2)[:, :, :self.h, :]
        frame = torch.cat([frame, torch.flip(frame, [3])], 3)[:, :, :, :self.w]
        return frame
    
    def process_frame(self, frame):
        frame = frame.astype(np.float32) / 255.0
        frame = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).cuda()
        if self.w % 8 != 0 or self.h % 8 != 0:
            frame = self._pad_image(frame)
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