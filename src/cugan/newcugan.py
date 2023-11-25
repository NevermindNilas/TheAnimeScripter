import os, torch, threading, time, requests
import numpy as np

from tqdm import tqdm
from moviepy.editor import VideoFileClip
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from multiprocessing import Queue
from .cugan_arch import UpCunet2x, UpCunet3x, UpCunet4x, UpCunet2x_fast

class Cugan:
    def __init__(self, video, output, half, nt, model_type, pro, w, h, fps, scale, tot_frame, model):
        self.video = video
        self.output = output
        self.half = half
        self.nt = nt
        self.model_type = model_type
        self.pro = pro
        self.w = w
        self.h = h
        self.fps = fps
        self.scale = scale
        self.tot_frame = tot_frame
        self.model = model
        self.processed_frames = {}
        
        self.initialize()
    
    def handle_model(self):
        if self.model_type == "shufflecugan":
            self.model = UpCunet2x_fast(in_channels=3, out_channels=3)
            self.filename = "sudo_shuffle_cugan_9.584.969.pth"
        else:
            model_path_prefix = "cugan_pro" if self.pro else "cugan"
            model_path_suffix = "-latest" if not self.pro else ""
            model_path_middle = f"up{self.scale}x"
            model_map = {2: UpCunet2x, 3: UpCunet3x, 4: UpCunet4x}
            self.model = model_map[self.scale](in_channels=3, out_channels=3)
            self.filename = f"{model_path_prefix}_{model_path_middle}{model_path_suffix}-{self.kind_model}.pth"
        
        if not os.path.exists("src/cugan/weights"):
            os.makedirs("src/cugan/weights")

        if not os.path.exists(os.path.join(os.path.abspath("src/cugan/weights"), self.filename)):
            print("Downloading Cugan model...")
            url = f"https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/{self.filename}"
            response = requests.get(url)
            if response.status_code == 200:
                with open(os.path.join("src/cugan/weights", self.filename), "wb") as file:
                    file.write(response.content)
        
        model_path = os.path.abspath(os.path.join("src/cugan/weights", self.filename))
        
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
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
                    
    def start_threads(self):
        threads = []
        for _ in range(self.nt):
            thread = threading.Thread(target=self.process_frames, args=(self.model, self.read_buffer, self.processed_frames, self.half))
            thread.start()
            threads.append(thread)
        
        for thread in threads:
            thread.join()
    
    def initialize(self):
        self.handle_model()
        self.start_threads()
        
        self.video = VideoFileClip(self.video)
        self.frames = self.video.iter_frames()
        self.writer = FFMPEG_VideoWriter(self.output, (self.w * self.scale, self.h * self.scale), self.fps) # barebones video writer
        self.pbar = tqdm(total=self.tot_frame, desc="Writing frames", unit="frames")
        
        self.read_buffer = Queue(maxsize=500)
        
    def build_buffer(self):
        try:
            for index, frame in enumerate(self.frames):
                if frame is None:
                    break
                self.read_buffer.put((index, frame))
        except:
            for _ in range(self.nt):
                self.read_buffer.put(None)
        self.video.close()
            
    def write_buffer(self):
        processing_index = 0
        time.sleep(1) # Allowing some frames to be built in the buffer before writing
        while True:
            if processing_index not in self.proccessed_frames:
                break
            self.writer.write_frame(self.processed_frames[processing_index])
            del self.processed_frames[processing_index]
            self.pbar.update(1)
            processing_index += 1
        
        self.pbar.close()
        self.writer.close()

class CuganMT(threading.Thread):
    def __init__(self, model, read_buffer, processed_frames, half):
        threading.Thread.__init__(self)
        self.model = model
        self.read_buffer = read_buffer
        self.processed_frames = processed_frames
        self.half = half
    
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
            index, frame = self.read_buffer.get()
            if index is None:
                break
            frame = self.process_frame(frame)
            with self.lock:
                self.processed_frames.put(index,frame)