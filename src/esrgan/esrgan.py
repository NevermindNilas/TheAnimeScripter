"""
TO DO
"""
import os, concurrent.futures, time, _thread, torch

from tqdm import tqdm
from queue import Queue
from moviepy.editor import VideoFileClip
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

class Esrgan():
    def __init__(self, video, output, w, h, fps, model_type, kind_model, nt, tot_frame, half):
        self.video = video
        self.output = output
        self.w = w
        self.h = h
        self.fps = fps
        self.model_type = model_type
        self.kind_model = kind_model
        self.nt = nt
        self.tot_frame = tot_frame
        self.half = half
        
        self.processed_frames = {}
        self.threads_are_running = True
        
        self.handle_model()
        self.initialize()

        self.threads_are_running = True

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.nt) as executor:
            for _ in range(self.nt):
                executor.submit(EsrganMT(self.model, self.read_buffer, self.processed_frames, self.half, self.w, self.h).run)
                
        while self.processing_index < self.tot_frame:
            time.sleep(0.1)
        
        self.threads_are_running = False

    def handle_model(self):
        """
        TO DO
        """
    
    def initialize(self):
        self.video = VideoFileClip(self.video)
        self.frames = self.video.iter_frames()
        self.writer = FFMPEG_VideoWriter(self.output, (self.w * self.scale, self.h * self.scale), self.fps, ffmpeg_params=["-b:v", "10000k", "-vcodec", "mpeg4"]) # TO DO: Change this
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

class EsrganMT():
    def __init__(self, model, read_buffer, processed_frames, half, w, h):
        self.model = model
        self.read_buffer = read_buffer
        self.processed_frames = processed_frames
        self.half = half
        self.cuda_available = torch.cuda.is_available()
        self.w = w
        self.h = h
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
    
    def process_frame_cpu(self, frame):
        """
        TO:DO: Implement this
        """
        frame = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().div_(255)
        frame = self.inference(frame)
        frame = frame.squeeze(0).permute(1, 2, 0).mul_(255).clamp_(0, 255).byte()
        return frame.numpy()
    

    def run(self):
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