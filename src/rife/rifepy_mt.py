from moviepy.editor import VideoFileClip
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from tqdm import tqdm

import concurrent.futures, time, _thread, torch
from queue import Queue


class Rife():
    def __init__(self, video, output, nt, fps, tot_frame, w, h, UHD, scale):
        """
        Attempt to Multithread Rife through an index system,
        
        The video will be 'split' into nt parts, each thread will process a part of the video and then write it to the output.
        
        each frame has an index so there should not be any issues with frames being written out of order.
        """
        self.video = video
        self.output = output
        self.nt = nt
        self.fps = fps
        self.tot_frame = tot_frame
        self.w = w
        self.h = h
        self.UHD = UHD
        self.scale = scale
        
        self.initialize()
        
        self.threads_are_running = True
        
        # TO:DO ADD LOGIC TO SPLIT THE VIDEO INTO NT PARTS AND THEN PROCESS EACH PART IN A THREAD
        
        self.split_video = tot_frame // nt
        
        # Possible bottleneck, if the interpolation works too fast, the read buffer might not be able to keep up and the threads will be idle.
        # Need some thinkering to find the best method

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.nt) as executor:
            for _ in range(self.nt):
                executor.submit(RifeMT(self.model, self.read_buffer, self.processed_frames, self.half, self.w, self.h).run)
                
        while self.processing_index < self.tot_frame:
            time.sleep(0.1)
        
        self.threads_are_running = False

    def initialize(self):
        
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
    
    def write_buffer(self):
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

class RifeMT():
    pass