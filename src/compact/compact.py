import os, requests, numpy as np, torch, _thread, threading, cv2, time, concurrent.futures

from torch import nn as nn
from tqdm import tqdm
from queue import Queue
from moviepy.editor import VideoFileClip
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

from .srvgg_arch import SRVGGNetCompact

class Compact():
    def __init__(self,video, output, multi, half, w, h, nt, tot_frame, fps, model_type, ffmpeg_params):
        self.video = video
        self.output = output
        self.scale = multi
        self.half = half
        self.w = w
        self.h = h
        self.nt = nt
        self.fps = fps
        self.tot_frame = tot_frame
        self.model_type = model_type
        self.ffmpeg_params = ffmpeg_params
        self.processed_frames = {}
        
        self.handle_models()
        self._initialize()
        
        self.threads_are_running = True
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.nt) as executor:
            for _ in range(self.nt):
                executor.submit(CompactMT(self.model, self.read_buffer, self.processed_frames, self.half).run)
                
        while self.processing_index < self.tot_frame:
            time.sleep(0.1)
        
        self.threads_are_running = False
    
    def handle_models(self):

        if self.model_type == "compact":
            if not os.path.exists("src/compact/weights"):
                os.makedirs("src/compact/weights")
            self.filename = "2x_Bubble_AnimeScale_Compact_v1.pth"
            if not os.path.exists(os.path.join(os.path.abspath("src/compact/weights"), self.filename)):
                print("Downloading Compact model...")
                url = f"https://github.com/Bubblemint864/AI-Models/releases/download/2x_Bubble_AnimeScale_Compact_v1/{self.filename}"
                response = requests.get(url)
                if response.status_code == 200:
                    with open(os.path.join("src/compact/weights", self.filename), "wb") as file:
                        file.write(response.content)
                        
        elif self.model_type == "ultracompact":
            if not os.path.exists("src/compact/weights"):
                os.makedirs("src/compact/weights")
            self.filename = "sudo_UltraCompact_2x_1.121.175_G.pth"
            if not os.path.exists(os.path.join(os.path.abspath("src/compact/weights"), self.filename)):
                print("Downloading UltraCompact model...")
                url = f"https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/{self.filename}"
                response = requests.get(url)
                if response.status_code == 200:
                    with open(os.path.join("src/compact/weights", self.filename), "wb") as file:
                        file.write(response.content)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            if self.half:
                torch.set_default_tensor_type(torch.cuda.HalfTensor)
                
        model_path = os.path.abspath(os.path.join("src/compact/weights", self.filename))
        
        num_conv = 16 if self.model_type == "compact" else 8
        self.model = SRVGGNetCompact(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_conv=num_conv,
                upscale=self.scale,
                act_type="prelu",
                )

        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval().cuda() if torch.cuda.is_available() else self.model.eval()
        if self.half:
            self.model.half()
            
    def _initialize(self):
        
        self.video = VideoFileClip(self.video)
        self.frames = self.video.iter_frames()
        self.writer = FFMPEG_VideoWriter(self.output, (self.w * self.scale, self.h * self.scale), self.fps, ffmpeg_params=self.ffmpeg_params)
        self.pbar = tqdm(total=self.tot_frame, desc="Writing frames", unit="frames")
            
        self.read_buffer = Queue(maxsize=500)
        _thread.start_new_thread(self.build_buffer, ())
        _thread.start_new_thread(self.write_thread, ())
   
    def build_buffer(self):
        for index, frame in enumerate(self.frames):
                if frame is None:
                    break
                self.read_buffer.put((index, frame))
        
        for _ in range(self.nt):
                self.read_buffer.put(None)
        self.video.close()
            
    def write_thread(self):
        self.processing_index = 0
        while True:
            if self.processing_index not in self.processed_frames:
                if self.processed_frames.get(self.processing_index) is None and self.threads_are_running is False:
                    break
                time.sleep(0.1)
                continue
            self.pbar.update(1)
            self.writer.write_frame(self.processed_frames[self.processing_index])
            del self.processed_frames[self.processing_index]
            self.processing_index += 1
        self.writer.close()
        self.pbar.close()
            
class CompactMT():
    def __init__(self, model, read_buffer, processed_frames, half):
        self.model = model
        self.read_buffer = read_buffer
        self.processed_frames = processed_frames
        self.half = half
        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available:
            self.model = self.model.cuda()

    def inference(self, frame):
        with torch.no_grad():
            if self.half:
                frame = frame.half()
            return self.model(frame)

    def process_frame(self, frame):
        frame = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().div_(255)
        if self.cuda_available:
            frame = frame.cuda()
        frame = self.inference(frame)
        frame = frame.squeeze(0).permute(1, 2, 0).mul_(255).clamp_(0, 255).byte()
        return frame.cpu().numpy()

    def run(self):  
        while True:
            index, frame = self.read_buffer.get()
            if index is None:
                break
            frame = self.process_frame(frame)
            self.processed_frames[index] = frame