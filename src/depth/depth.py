import os
import torch
import logging
import subprocess
import numpy as np
import _thread
import cv2

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from multiprocessing import Queue

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
        # Tested them all, this one is the best, I will add an option to choose later
        model_type = "DPT_Hybrid"
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.hub.load(
            "intel-isl/MiDaS", model_type, pretrained=True).to(self.device)
        midas_transforms = torch.hub.load(
            "intel-isl/MiDaS", "transforms", verbose=False)

        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

        self.model.eval()

        if torch.cuda.is_available():
            self.model.cuda()

        # Checking for BF16 support, if not, then we will use FP32
        if self.half:
            try:
                torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16)
                logging.info(
                    "Using BF16 mixed precision")
            except:
                logging.warning(
                    "Your GPU does not support BF16 mixed precision, using FP32 instead")
                self.half = False

    def build_buffer(self):

        ffmpeg_command = [
            self.ffmpeg_path,
            "-i", str(self.input),
        ]

        if self.outpoint != 0:
            ffmpeg_command.extend(
                ["-ss", str(self.inpoint), "-to", str(self.outpoint)])

        ffmpeg_command.extend([
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-v", "quiet",
            "-stats",
            "-",
        ])

        self.reading_done = False
        try:
            process = subprocess.Popen(
                ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            logging.info(f"Running command: {ffmpeg_command}")

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

                frame = self.transform(frame).to(self.device)

                with torch.cuda.amp.autocast(enabled=self.half, dtype=torch.bfloat16) if self.half else torch.no_grad():
                    prediction = self.model(frame)

                    prediction = torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size=(self.height, self.width),
                        mode="bicubic",
                        align_corners=False,
                    ).squeeze()

                output = prediction.float().cpu().detach().numpy()
                formatted = (output * 255 / np.max(output)).astype('uint8')

                formatted_rgb = cv2.cvtColor(formatted, cv2.COLOR_GRAY2RGB)
                self.processed_frames.put_nowait(formatted_rgb)

        except Exception as e:
            logging.exception("An error occurred during processing")

        finally:
            self.processing_finished = True

    def write_buffer(self):

        from src.encode_settings import encode_settings

        command: list = encode_settings(self.encode_method, self.width, self.height,
                                        self.fps, self.output, self.ffmpeg_path, 0, 0)

        logging.info(
            f"Encoding options: {' '.join(command)}")

        pipe = subprocess.Popen(
            command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        try:
            while True:
                frame = self.processed_frames.get()
                if frame is None:
                    if self.processing_finished == True:
                        break

                frame = np.ascontiguousarray(frame)
                pipe.stdin.write(frame.tobytes())
                self.pbar.update(1)

        except Exception as e:
            logging.exception("An error occurred during writing")

        finally:
            pipe.stdin.close()
            pipe.wait()
            self.pbar.close()

            return
