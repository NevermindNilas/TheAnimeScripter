import os
import torch
import logging
import subprocess
import numpy as np
import cv2
import wget

from torchvision.transforms import Compose
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from queue import Queue
import torch.nn.functional as F

from .dpt import DPT_DINOv2
from .util.transform import Resize, NormalizeImage, PrepareForNet

os.environ['TORCH_HOME'] = os.path.dirname(os.path.realpath(__file__))


class Depth():
    def __init__(self, input, output, ffmpeg_path, width, height, fps, nframes, half, inpoint=0, outpoint=0, encode_method="x264", depth_method="small"):

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
        self.depth_method = depth_method

        self.handle_model()

        self.pbar = tqdm(
            total=self.nframes, desc="Processing Frames", unit="frames", dynamic_ncols=True, colour="green")

        self.read_buffer = Queue(maxsize=500)
        self.processed_frames = Queue()

        with ThreadPoolExecutor(max_workers=3) as executor:
            executor.submit(self.build_buffer)
            executor.submit(self.process)
            executor.submit(self.write_buffer)

    def handle_model(self):

        match self.depth_method:
            case "small":
                model = "vits"
                self.model = DPT_DINOv2(encoder='vits', features=64, out_channels=[
                                        48, 96, 192, 384], localhub=False)
                url = "https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vits14.pth?download=true"
            case "base":
                model = "vitb"
                self.model = DPT_DINOv2(encoder='vitb', features=128, out_channels=[
                                        96, 192, 384, 768], localhub=False)
                url = "https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitb14.pth?download=true"

            case "large":
                model = "vitl"
                url = "https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitl14.pth?download=true"
                self.model = DPT_DINOv2(encoder='vitl', features=256, out_channels=[
                                        256, 512, 1024, 1024], localhub=False)

        weightsDir = os.path.join(os.path.dirname(
            os.path.realpath(__file__)), "weights")

        model_path = os.path.join(
            weightsDir, f"depth_anything_{model}14.pth")

        if not os.path.exists(model_path):
            print("Couldn't find the depth model, downloading it now...")

            logging.info(
                "Couldn't find the depth model, downloading it now...")

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

        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        self.reading_done = False
        frame_size = self.width * self.height * 3
        frame_count = 0

        try:
            for chunk in iter(lambda: process.stdout.read(frame_size), b''):
                if len(chunk) != frame_size:
                    logging.error(
                        f"Read {len(chunk)} bytes but expected {frame_size}")
                    continue
                frame = np.frombuffer(chunk, dtype=np.uint8).reshape(
                    (self.height, self.width, 3))

                self.read_buffer.put(frame)
                frame_count += 1

        except Exception as e:
            logging.exception(
                f"An error occurred during reading, {e}")
            raise e

        finally:
            logging.info(
                f"Built buffer with {frame_count} frames")

            process.stdout.close()
            process.terminate()

            self.reading_done = True
            self.read_buffer.put(None)

    def process(self):
        self.processing_done = False
        frame_count = 0
        try:
            while True:
                frame = self.read_buffer.get()
                if frame is None:
                    if self.reading_done == True and self.read_buffer.empty():
                        break
                    else:
                        continue

                frame = frame/255
                frame = self.transform({'image': frame})['image']
                frame = torch.from_numpy(frame).unsqueeze(0).to(self.device)

                if self.half and self.cudaIsAvailable:
                    frame = frame.half()

                with torch.no_grad():
                    depth = self.model(frame)

                depth = F.interpolate(
                    depth[None], (self.height, self.width), mode='bilinear', align_corners=False)[0, 0]
                depth = (depth - depth.min()) / \
                    (depth.max() - depth.min()) * 255.0
                depth = depth.cpu().numpy().astype(np.uint8)

                print(depth.shape)
                self.processed_frames.put(depth)
                frame_count += 1

        except Exception as e:
            logging.exception(
                f"An error occurred during reading, {e}")
            raise e

        finally:
            logging.info(
                f"Processed {frame_count} frames")

            self.processing_done = True
            self.processed_frames.put(None)

    def write_buffer(self):

        from src.ffmpegSettings import encodeSettings
        command: list = encodeSettings(self.encode_method, self.width, self.height,
                                       self.fps, self.output, self.ffmpeg_path, False, 0, grayscale=True)

        pipe = subprocess.Popen(
            command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        frame_count = 0
        try:
            while True:
                frame = self.processed_frames.get()
                if frame is None:
                    if self.processing_done == True and self.processed_frames.empty():
                        break
                    else:
                        continue

                frame_count += 1
                # frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                pipe.stdin.write(frame.tobytes())
                self.pbar.update()

        except Exception as e:
            logging.exception(
                f"An error occurred during reading, {e}")
            raise e

        finally:
            logging.info(
                f"Wrote {frame_count} frames")

            pipe.stdin.close()
            self.pbar.close()

            return
