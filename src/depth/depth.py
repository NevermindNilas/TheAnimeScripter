import os
import torch
import logging
import subprocess
import numpy as np
import _thread
import cv2
import wget

from torchvision.transforms import Compose
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from multiprocessing import Queue
import torch.nn.functional as F

from .dpt import DPT_DINOv2
from .util.transform import Resize, NormalizeImage, PrepareForNet

os.environ['TORCH_HOME'] = os.path.dirname(os.path.realpath(__file__))


class Depth():
    def __init__(self, input, output, ffmpeg_path, width, height, fps, nframes, half, inpoint=0, outpoint=0, encode_method="x264"):

        self.input = input
        self.output = output
        self.ffmpeg_path = ffmpeg_path
        self.width = width
        self.height = height
        self.fps = fps
        self.nframes = nframes
        self.half = half
        self.inpoint = inpoint
        self.outpoint = outpoint
        self.encode_method = encode_method

        self.handle_model()

        self.pbar = tqdm(
            total=self.nframes, desc="Processing Frames", unit="frames", dynamic_ncols=True, colour="green")

        self.read_buffer = Queue(maxsize=500)
        self.processed_frames = Queue()

        _thread.start_new_thread(self.build_buffer, ())
        _thread.start_new_thread(self.write_buffer, ())

        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(self.start_process)

    def handle_model(self):
        
        # I will add an option to choose later, but the small model is 90% (subjectively) as good as the large model and 3x the performance.
        model = "vits"

        if model == 'vits':
            self.model = DPT_DINOv2(encoder='vits', features=64, out_channels=[
                                    48, 96, 192, 384], localhub=False)
            url = "https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vits14.pth?download=true"
        elif model == 'vitb':
            self.model = DPT_DINOv2(encoder='vitb', features=128, out_channels=[
                                    96, 192, 384, 768], localhub=False)
            url = "https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitb14.pth?download=true"
        elif model == 'vitl':
            url = "https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitl14.pth?download=true"
            self.model = DPT_DINOv2(encoder='vitl', features=256, out_channels=[
                                    256, 512, 1024, 1024], localhub=False)

        weightsDir = os.path.join(os.path.dirname(
            os.path.realpath(__file__)), "weights")

        model_path = os.path.join(
            weightsDir, f"depth_anything_{model}14.pth")
        
        if not os.path.exists(model_path):
            os.makedirs(weightsDir, exist_ok=True)
            wget.download(url, model_path)

        self.cudaIsAvailable = torch.cuda.is_available()

        if self.cudaIsAvailable:
            self.device = torch.device("cuda")
            self.model = self.model.cuda()
        else:
            self.device = torch.device("cpu")

        if not os.path.exists(model_path):
            raise Exception(
                f"Model {model_path} does not exist. Please download it from https://huggingface.co/spaces/LiheYoung/Depth-Anything")

        self.model.load_state_dict(torch.load(
            model_path, map_location='cpu'), strict=True)

        if self.half and self.cudaIsAvailable:
            self.model = self.model.half()
        else:
            self.half = False

        self.transform = Compose([
            Resize(
                width=518,
                height=518,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

    def build_buffer(self):

        from src.ffmpegSettings import decodeSettings
        command: list = decodeSettings(
            self.input, self.inpoint, self.outpoint, False, 0, self.ffmpeg_path)

        self.reading_done = False
        try:
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            frame_size = self.width * self.height * 3

            frame_count = 0
            for chunk in iter(lambda: process.stdout.read(frame_size), b''):
                if len(chunk) != frame_size:
                    logging.error(
                        f"Read {len(chunk)} bytes but expected {frame_size}")
                    break

                frame = np.frombuffer(chunk, dtype=np.uint8).reshape(
                    (self.height, self.width, 3))

                self.read_buffer.put_nowait(frame)
                frame_count += 1

            stderr = process.stderr.read().decode()
            if stderr:
                if "bitrate=" not in stderr:
                    logging.error(f"ffmpeg error: {stderr}")
        except:
            logging.exception("An error occurred during reading")
        finally:
            # For terminating the pipe and subprocess properly
            process.stdout.close()
            process.stderr.close()
            process.terminate()
            self.read_buffer.put(None)
            self.reading_done = True

    def start_process(self):
        try:
            while True:
                frame = self.read_buffer.get()
                if frame is None:
                    self.reading_done = True
                    break
                
                frame = frame/255
                frame = self.transform({'image': frame})['image']
                frame = torch.from_numpy(frame).unsqueeze(0).to(self.device)

                if self.half and self.cudaIsAvailable:
                    frame = frame.half()
                    
                with torch.no_grad():
                    depth = self.model(frame)

                depth = F.interpolate(depth[None], (self.height, self.width), mode='bilinear', align_corners=False)[0, 0]
                depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
                
                depth = depth.cpu().numpy().astype(np.uint8)
                cv2.imshow("Depth", depth)
                cv2.waitKey(2)
                depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2RGB)
                
                self.processed_frames.put(depth)

        except Exception as e:
            logging.exception("An error occurred during processing")

        finally:
            self.processing_finished = True

    def write_buffer(self):

        from src.ffmpegSettings import encodeSettings

        command: list = encodeSettings(self.encode_method, self.width, self.height,
                                       self.fps, self.output, self.ffmpeg_path, False, 0)

        pipe = subprocess.Popen(
            command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        try:
            while True:
                frame = self.processed_frames.get()
                if frame is None:
                    if self.processing_finished == True:
                        break
                
                pipe.stdin.write(frame.tobytes())
                self.pbar.update(1)

        except Exception as e:
            logging.exception("An error occurred during writing")

        finally:
            pipe.stdin.close()
            pipe.wait()
            self.pbar.close()

            return
