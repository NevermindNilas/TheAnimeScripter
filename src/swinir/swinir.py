import os, requests, numpy as np, torch, _thread, threading, time, concurrent.futures, sys

from torch import nn as nn
from tqdm import tqdm
from queue import Queue
from moviepy.editor import VideoFileClip
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

from .network import SwinIR as SwinIR_model

class Swin():
    def __init__(self, video, output, model_type, scale, half, nt, metadata, kind_model, ffmpeg_params):
        self.video = video
        self.output = output
        self.model_type = model_type
        self.scale = scale
        self.half = half
        self.nt = nt
        self.metadata = metadata
        self.kind_model = kind_model
        self.ffmpeg_params = ffmpeg_params
        self.processed_frames = {}
        
        self.handle_models()
        self._initialize()
        
        self.threads_are_running = True
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.nt) as executor:
            for _ in range(self.nt):
                executor.submit(SwinMT(self.model, self.read_buffer, self.processed_frames, self.half, self.metadata).run)
                
        while self.processing_index < self.metadata["nframes"]:
            time.sleep(0.1)

        self.threads_are_running = False
    
    def handle_models(self):
        
        self.kind_model = self.kind_model.lower()
        if self.kind_model == None:
            sys.exit("Please specify a model type")
            
        elif self.kind_model != "small" and self.kind_model != "medium" and self.kind_model != "large":
            sys.exit("Invalid kind_model type, please choose from: small, medium or large")
            
        model_type = {
            'small': f'002_lightweightSR_DIV2K_s64w8_SwinIR-S_x{self.scale}.pth',
            'medium': f'001_classicalSR_DF2K_s64w8_SwinIR-M_x{self.scale}.pth',
            'large': f'003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth'
        }

        if self.kind_model == "large" and self.scale != 4:
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
        
        self.model.load_state_dict(self.pretrained_weights, strict=True)
        self.model.eval().cuda() if torch.cuda.is_available() else self.model.eval()
        
        if self.half:
            self.model.half()
        
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            if self.half:
                torch.set_default_tensor_type(torch.cuda.HalfTensor)
    
        
        self.video = VideoFileClip(self.video)
        self.frames = self.video.iter_frames()
        self.writer = FFMPEG_VideoWriter(self.output, (self.metadata["width"] * self.scale, self.metadata["height"] * self.scale), self.metadata["fps"], ffmpeg_params=self.ffmpeg_params)
        self.pbar = tqdm(total=self.metadata["nframes"], desc="Writing frames", unit="frames")
        
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
    
class SwinMT(threading.Thread):
    def __init__(self, model, read_buffer, processed_frames, half, metadata):
        self.model = model
        self.half = half
        self.read_buffer = read_buffer
        self.h = metadata["height"]
        self.w = metadata["width"]
        self.processed_frames = processed_frames
        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available:
            self.model = self.model.cuda()

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
        frame = torch.from_numpy(frame.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).div_(255)
        if self.cuda_available:
            frame = frame.cuda()
        if self.w % 8 != 0 or self.h % 8 != 0:
            frame = self.pad_frame(frame)
        frame = self.inference(frame)
        frame = frame.squeeze(0).permute(1, 2, 0).mul_(255).clamp_(0, 255).byte()
        return frame.cpu().numpy()
    
    def process_frame_cpu(self, frame):
        frame = np.transpose(frame, (2, 0, 1))
        frame = torch.from_numpy(frame).unsqueeze(0).float().div_(255)
        frame = self.inference(frame)
        frame = frame.squeeze(0).permute(1, 2, 0).mul_(255).clamp_(0, 255).byte()
        return frame.numpy()

    def run(self):
        # CPU Only not supported yet
        if self.cuda_available:
            while True:
                index, frame = self.read_buffer.get()
                if index is None:
                    break
                frame = self.process_frame(frame)
                self.processed_frames[index] = frame
        else:
            while True:
                index, frame = self.read_buffer.get()
                if index is None:
                    break
                frame = self.process_frame_cpu(frame)
                self.processed_frames[index] = frame