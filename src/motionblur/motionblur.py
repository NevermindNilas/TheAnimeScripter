import subprocess
import _thread
import logging
import numpy as np
import time
import cv2

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from multiprocessing import Queue


class Motionblur():
    def __init__(self, input, output, ffmpeg_path, width, height, fps, nframes, inpoint, outpoint, motion_blur_sens, interpolate_method, interpolate_factor, half):

        self.input = input
        self.output = output
        self.ffmpeg_path = ffmpeg_path
        self.width = width
        self.height = height
        self.fps = fps
        self.nframes = nframes
        self.inpoint = inpoint
        self.outpoint = outpoint
        self.motion_blur_sens = motion_blur_sens
        self.interpolate_method = interpolate_method
        self.interpolate_factor = interpolate_factor
        self.half = half

        self.handle_model()

        self.read_buffer = Queue(maxsize=500)
        self.processed_frames = Queue(maxsize=500)

        self.writing_finished = False
        _thread.start_new_thread(self.build_buffer, ())

    def handle_model(self):

        from src.rife.rife import Rife

        UHD = False if self.height < 3840 or self.width < 2160 else True
        self.interpolate_process = Rife(
            self.interpolate_factor, self.half, self.width, self.height, UHD, self.interpolate_method)

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

            self.reading_done = True

    def start_process(self):
        prev_frame = None
        try:
            while True:
                frame = self.read_buffer.get()

                if frame is None:
                    self.reading_done = True
                    break

                if prev_frame is not None:
                    self.interpolate_process.run(prev_frame, frame)

                    for i in range(self.interpolate_factor - 1):
                        result = self.interpolate_process.make_inference(
                            (i + 1) * 1. / (self.interpolate_factor + 1))

                        self.processed_frames.put(result)
                        
                    prev_frame = frame
                else:
                    self.processed_frames.put(frame)
                    prev_frame = frame

        except Exception as e:
            logging.exception("An error occurred during processing")

        finally:
            self.processed_frames.put(None)
            self.processing_done = True

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

        logging.info(
            f"Encoding options: {' '.join(command)}")

        pipe = subprocess.Popen(
            command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        try:
            while True:
                frame = self.processed_frames.get()
                if frame is None:
                    if self.processing_done == True:
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

    def run(self):
        self.pbar = tqdm(
            total=self.nframes, desc="Processing Frames", unit="frames", dynamic_ncols=True, colour="green")

        _thread.start_new_thread(self.write_buffer, ())

        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(self.start_process)

        while self.reading_done == False and self.writing_finished == False:
            time.sleep(0.1)