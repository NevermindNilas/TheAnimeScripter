import os
import torch
import logging
import subprocess
import numpy as np
import _thread
import time

from tqdm import tqdm
from multiprocessing import Queue

os.environ['TORCH_HOME'] = os.path.dirname(os.path.realpath(__file__))

class Depth():
    def __init__(self, input, output, ffmpeg_path, width, height, fps, nframes, half, inpoint=0, outpoint=0):

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

        self.handle_model()
        self.threads_done = False
        self.read_buffer = Queue(maxsize=100)
        self.processed_frames = Queue(maxsize=100)

        _thread.start_new_thread(self.build_buffer, ())

    def handle_model(self):
        # Tested them all, this one is the best, I will add an option to choose later
        model_type = "DPT_Hybrid"
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.hub.load(
            "intel-isl/MiDaS", model_type, pretrained=True).to(self.device)
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

        self.model.eval()

        if torch.cuda.is_available():
            self.model.cuda()

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

        try:
            process = subprocess.Popen(
                ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            logging.info(f"Running command: {ffmpeg_command}")

            frame_size = self.width * self.height * 3

            for chunk in iter(lambda: process.stdout.read(frame_size), b''):
                if len(chunk) != frame_size:
                    logging.error(
                        f"Read {len(chunk)} bytes but expected {frame_size}")
                    break
                frame = np.frombuffer(chunk, dtype=np.uint8).reshape(
                    (self.height, self.width, 3))
                self.read_buffer.put(frame)

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

    def write_buffer(self):
        command = [self.ffmpeg_path,
                   '-y',
                   '-f', 'rawvideo',
                   '-vcodec', 'rawvideo',
                   '-s', f'{self.width}x{self.height}',
                   '-pix_fmt', 'gray',
                   '-r', str(self.fps),
                   '-i', '-',
                   '-an',
                   '-c:v', 'libx264',
                   '-preset', 'veryfast',
                   '-crf', '15',
                   '-tune', 'animation',
                   '-movflags', '+faststart',
                   self.output]

        pipe = subprocess.Popen(
            command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        try:
            while True:
                if not self.processed_frames:
                    if self.read_buffer.empty() and self.threads_done == True:
                        break
                    continue

                frame = self.processed_frames.get()

                # Write the frame to FFmpeg
                if frame is not None:
                    pipe.stdin.write(frame.tobytes())

                self.pbar.update(1)
        except Exception as e:
            logging.exception("An error occurred during writing")

        finally:
            # Close the pipe
            pipe.stdin.close()
            pipe.wait()

            self.pbar.close()
            self.writing_finished = True

    def start_process(self):
        try:
            while True:
                frame = self.read_buffer.get()
                if frame is None:
                    self.threads_done = True
                    break

                frame = self.transform(frame).to(self.device)

                if self.half:
                    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                        with torch.no_grad():
                            prediction = self.model(frame)

                            prediction = torch.nn.functional.interpolate(
                                prediction.unsqueeze(1),
                                size=(self.height, self.width),
                                mode="bicubic",
                                align_corners=False,
                            ).squeeze()

                    output = prediction.float().cpu().numpy()
                    formatted = (output * 255 / np.max(output)).astype('uint8')
                else:
                    with torch.no_grad():
                        prediction = self.model(frame)

                        prediction = torch.nn.functional.interpolate(
                            prediction.unsqueeze(1),
                            size=(self.height, self.width),
                            mode="bicubic",
                            align_corners=False,
                        ).squeeze()

                    output = prediction.cpu().numpy()
                    formatted = (output * 255 / np.max(output)).astype('uint8')

                self.processed_frames.put(formatted)

        except Exception as e:
            logging.exception("An error occurred during processing")

        finally:
            self.processing_finished = True

    def run(self):
        self.pbar = tqdm(
            total=self.nframes, desc="Processing Frames", unit="frames", dynamic_ncols=True, colour="green")

        _thread.start_new_thread(self.start_process, ())
        _thread.start_new_thread(self.write_buffer, ())

        while True:
            if self.threads_done == True and self.read_buffer.empty() and self.processed_frames.empty():
                break
            time.sleep(0.1)
