import os, torch, threading, time, requests, _thread, numpy as np, concurrent.futures

from tqdm import tqdm
from moviepy.editor import VideoFileClip
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from multiprocessing import Queue
from .cugan_arch import UpCunet2x, UpCunet3x, UpCunet4x, UpCunet2x_fast

'''
ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of default=nokey=1:noprint_wrappers=1 .\input\input_shufflecugan.mp4 
'''

class Cugan:
    def __init__(self, video, output, multi, half, kind_model, pro, w, h, nt, fps, tot_frame, model_type):
        """
        The principle behind everything is that we start a thread using _thread.start_new_thread in order to iterate and append the frames onto a buffer.
        and then we start yet another one in order to write the processed frames onto the output video.

        Each frame has an index associated with it, and we use that index to write the frames in the correct order and to avoid
        any race conditions that may occur over time.
        """
        self.video = video
        self.output = output
        self.half = half
        self.nt = nt
        self.model_type = model_type
        self.pro = pro
        self.w = w
        self.h = h
        self.fps = fps
        self.scale = multi
        self.tot_frame = tot_frame
        self.kind_model = kind_model
        self.processed_frames = {}
        
        self.handle_model()
        self.initialize()
        
        self.threads_are_running = True
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.nt) as executor:
            for _ in range(self.nt):
                executor.submit(CuganMT(self.model, self.read_buffer, self.processed_frames, self.half).run)
                
        while self.processing_index < self.tot_frame:
            time.sleep(0.1)
        
        self.threads_are_running = False
        
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
            print(f"Downloading {self.model_type.upper()}  model...")
            url = f"https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/{self.filename}"
            response = requests.get(url)
            if response.status_code == 200:
                with open(os.path.join("src/cugan/weights", self.filename), "wb") as file:
                    file.write(response.content)
        
        model_path = os.path.abspath(os.path.join("src/cugan/weights", self.filename))
        
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval().cuda() if torch.cuda.is_available() else self.model.eval()

        if torch.cuda.is_available():
            print("ASOIFHNAOFBNASOFASO")
            
        if self.half:
            self.model.half()
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            if self.half:
                torch.set_default_tensor_type(torch.cuda.HalfTensor)
    
    def initialize(self):
        self.video = VideoFileClip(self.video)
        self.frames = self.video.iter_frames()
        self.writer = FFMPEG_VideoWriter(self.output, (self.w * self.scale, self.h * self.scale), self.fps)
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
        
class CuganMT():
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