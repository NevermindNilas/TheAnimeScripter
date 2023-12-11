import torch, threading, time, _thread, concurrent.futures
from tqdm import tqdm
from moviepy.editor import VideoFileClip
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from multiprocessing import Queue

class Segment():
    def __init__(self, video, output, nt, half, w, h, fps, tot_frame, kind_model):
        self.video = video
        self.output = output
        self.nt = nt
        self.half = half
        self.w = w
        self.h = h
        self.fps = fps
        self.tot_frame = tot_frame
        self.kind_model = kind_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processed_frames = {}
        self.threads_are_running = True

        self.initialize()
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.nt) as executor:
            for _ in range(self.nt):
                executor.submit(SegmentMT(self.read_buffer, self.processed_frames, self.kind_model).run)
                
        while self.processing_index < self.tot_frame:
            time.sleep(0.1)
        
        self.threads_are_running = False
        
    def initialize(self):
        
        self.pbar = tqdm(total=self.tot_frame)
        self.read_buffer = Queue(maxsize=500)
        self.video = VideoFileClip(self.video)
        self.frames = self.video.iter_frames()
        self.writer = FFMPEG_VideoWriter(self.output, (self.w, self.h), self.fps, codec="png", preset="medium", withmask=True)
        
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
                
class SegmentMT(threading.Thread):
    def __init__(self, read_buffer, processed_frames, kind_model):

        self.read_buffer = read_buffer
        self.processed_frames = processed_frames
        self.kind_model = kind_model
        self.kind_model = self.kind_model.lower()
        if self.kind_model == None:
            model = "isnet_anime"
        else:
            model = self.kind_model
        
        print("Sadly, due to onnxruntime limitations, CUDA drivers past 11.8 won't work meaning this process is being done on the CPU")
        print("If you want to use your GPU, you will have to downgrade your CUDA drivers, houray innovation!")
        
        self.session = new_session(model)
        
    def inference(self, frame):
        frame = remove(frame, session=self.session)
        return frame
    
    def run(self):
        while True:
            index, frame = self.read_buffer.get()
            if index is None:
                break
            frame = self.inference(frame)
            self.processed_frames[index] = frame