import os, torch, time, requests, _thread, concurrent.futures, numpy as np

from tqdm import tqdm
from moviepy.editor import VideoFileClip
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from multiprocessing import Queue

class Depth:
    def __init__(self, video, output, half, metadata, nt, model_type, ffmpeg_params):
        """
        Depth Estimation using Intel's Midas model, 
        due to size restrictions the only model that will be supported is the small Swin2T model.
        """
        self.video = video
        self.output = output
        self.half = half
        self.metadata = metadata
        self.nt = nt
        self.model_type = model_type
        self.ffmpeg_params = ffmpeg_params
        self.processed_frames = {}

        self.handle_model()
        self.initialize()

        self.threads_are_running = True

        self.nt = 1 # Force setting it to 1 cuz it can't work with 2 threads, it's a bug or a feature :D
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.nt) as executor:
            for _ in range(self.nt):
                executor.submit(DepthMT(self.device, self.model, self.read_buffer, self.processed_frames, self.half).run)
        while self.processing_index < self.metadata["nframes"]:
            time.sleep(0.1)
        
        self.threads_are_running = False
        
    def handle_model(self):
        if not os.path.exists("src/midas/weights"):
            os.mkdir("src/midas/weights")
        """
        if not os.path.exists(os.path.join(os.path.abspath("src/midas/weights"), "dpt_swin2_tiny_256.pt")):
            print("Downloading model weights...")
            url = "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin2_tiny_256.pt"
            response = requests.get(url)
            if response.status_code == 200:
                with open(os.path.join("src/midas/weights", "dpt_swin2_tiny_256.pt"), "wb") as file:
                    file.write(response.content)
        """

        os.environ['TORCH_HOME'] = os.path.abspath("src/midas/weights")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_type = "DPT_Hybrid"
        
        self.model = torch.hub.load("intel-isl/MiDaS", model_type, pretrained=True).to(self.device)
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            if(self.half):
                torch.set_default_tensor_type(torch.cuda.HalfTensor)

        self.model.eval()
        self.model.cuda()

    def initialize(self):
        self.video = VideoFileClip(self.video)
        self.frames = self.video.iter_frames()
        self.writer = FFMPEG_VideoWriter(self.output, (self.metadata["width"],  self.metadata["height"]), self.metadata["fps"], ffmpeg_params=self.ffmpeg_params)
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

class DepthMT():
    def __init__(self, device, model, read_buffer, processed_frames, half):
        self.device = device
        self.model = model
        self.read_buffer = read_buffer
        self.processed_frames = processed_frames
        self.half = half
        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available:
            self.model.cuda()
    
    def process_frame(self, frame):
        with torch.no_grad():
            img = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().div_(255)
            if self.half:
                img = img.half()
                self.model = self.model.half()
            img = img.to(self.device)
            prediction = self.model(img)
            depth_map = prediction[0]
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255
            return depth_map.byte().cpu().numpy()
    
    def run(self):
        while True:
            index, frame = self.read_buffer.get()
            if index is None:
                break
            frame = self.process_frame(frame)
            self.processed_frames[index] = frame
