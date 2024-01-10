import time
import torch
import _thread
import logging
import subprocess
import numpy as np

from tqdm import tqdm
from queue import Queue, SimpleQueue
from concurrent.futures import ThreadPoolExecutor

# https://github.com/JalaliLabUCLA/phycv/blob/main/scripts/run_vevid_lite_video.py


class Vevid:
    def __init__(self, input, output, height, width, fps, half, ffmpeg_path, nframes, inpoint=0, outpoint=0, b=0.5, g=0.6):
        self.input = input
        self.output = output
        self.height = height
        self.width = width
        self.fps = fps
        self.half = half
        self.ffmpeg_path = ffmpeg_path
        self.inpoint = inpoint
        self.outpoint = outpoint

        self.processing_finished = False
        self.nframes = nframes
        # Params based on script
        self.b = 0.5
        self.G = 0.6

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.is_available():
            from phycv import VEVID_GPU
            self.vevid = VEVID_GPU(device=self.device)

            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            if self.half:
                torch.set_default_dtype(torch.float16)

        else:
            from phycv import VEVID
            self.vevid = VEVID(device=self.device)

        self.read_buffer = Queue(maxsize=100)
        self.processed_frames = Queue(maxsize=100)

        _thread.start_new_thread(self.build_buffer, ())

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

            frame_count = 0
            for chunk in iter(lambda: process.stdout.read(frame_size), b''):
                if len(chunk) != frame_size:
                    logging.error(
                        f"Read {len(chunk)} bytes but expected {frame_size}")
                    break
                frame = np.frombuffer(chunk, dtype=np.uint8).reshape(
                    (self.height, self.width, 3))
                self.read_buffer.put(frame)
                frame_count += 1

            self.pbar.total = frame_count
            self.pbar.refresh()

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

    def process(self):
        while True:
            frame = self.read_buffer.get()
            if frame is None:
                if self.reading_done == True and self.read_buffer.empty():
                    break

            frame = torch.from_numpy(frame).permute(
                2, 0, 1).mul_(1/255)

        
            frame = frame.to(self.device)
            self.vevid.load_img(img_array=frame)

            # Going to support Lite mode only for now
            self.vevid.apply_kernel(b = self.b, G = self.G, lite=True)

            frame = self.vevid.vevid_output.cpu().numpy()
            
            print("Procesed frame")
            self.processed_frames.put_nowait(frame)

    def write_buffer(self):
        command = [self.ffmpeg_path,
                   '-y',
                   '-f', 'rawvideo',
                   '-vcodec', 'rawvideo',
                   '-s', f'{self.width}x{self.height}',
                   '-pix_fmt', 'rgb24',
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
                if not self.processed_frames.empty():
                    frame = self.processed_frames.get()
                    if frame is None:
                        if self.threads_done == True:
                            break

                    # Write the frame to FFmpeg
                    pipe.stdin.write(frame.tobytes())

                    self.pbar.update(1)
        except Exception as e:
            logging.exception("An error occurred during writing")

        finally:
            # Close the pipe
            pipe.stdin.close()
            pipe.wait()
            self.pbar.close()
            self.processing_finished = True

    def run(self):
        self.pbar = tqdm(
            total=self.nframes, desc="Processing Frames", unit="frames", dynamic_ncols=True, colour="green")
        
        _thread.start_new_thread(self.process, ())
        _thread.start_new_thread(self.write_buffer, ())

        while True:
            if self.processing_finished == True and self.read_buffer.empty() and self.processed_frames.empty():
                break
            time.sleep(0.1)
