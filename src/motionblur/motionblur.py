import subprocess
import _thread
import logging
import numpy as np

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from multiprocessing import Queue

"""
For this to work really fast, I need to write in C or Rust probably

I am looking into TMix and TBlend but I am not sure how to implement it yet due to the fact

Until then, the I/O Bottleneck will be the limiting factor
"""


class Motionblur():
    def __init__(self, input, output, ffmpeg_path, width, height, fps, nframes, inpoint, outpoint, interpolate_method, interpolate_factor, half, encode_method, dedup, dedup_strenght):
        self.input = input
        self.output = output
        self.ffmpeg_path = ffmpeg_path
        self.width = width
        self.height = height
        self.fps = fps
        self.nframes = nframes
        self.inpoint = inpoint
        self.outpoint = outpoint
        self.interpolate_method = interpolate_method
        self.interpolate_factor = interpolate_factor
        self.half = half
        self.encode_method = encode_method
        self.dedup = dedup
        self.dedup_strenght = dedup_strenght

        self.handle_model()

        self.read_buffer = Queue(maxsize=500)
        self.interpolated_frames = Queue(maxsize=500)
        self.processed_frames = Queue(maxsize=500)

        self.pbar = tqdm(
            desc="Processing", total=self.nframes, unit="frames", unit_scale=True, dynamic_ncols=True, leave=False, colour="green")

        self.writing_finished = False
        _thread.start_new_thread(self.build_buffer, ())
        _thread.start_new_thread(self.clear_write_buffer, ())

        with ThreadPoolExecutor(max_workers=2) as executor:
            executor.submit(self.interpolate_frames)
            executor.submit(self.blend_frames)

    def handle_model(self):

        from src.rife.rife import Rife

        UHD = False if self.height < 3840 or self.width < 2160 else True
        self.interpolate_process = Rife(
            self.interpolate_factor, self.half, self.width, self.height, UHD, self.interpolate_method)

    def build_buffer(self):

        from src.ffmpegSettings import decodeSettings

        command: list = decodeSettings(
            self.input, self.inpoint, self.outpoint, self.dedup, self.dedup_strenght, self.ffmpeg_path)

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

        finally:
            stderr = process.stderr.read().decode()
            if stderr:
                if "bitrate=" not in stderr:
                    logging.error(f"ffmpeg error: {stderr}")

            logging.info(f"Built buffer with {frame_count} frames")

            self.pbar.total = frame_count
            self.pbar.refresh()

            # For terminating the pipe and subprocess properly
            process.stdout.close()
            process.stderr.close()
            process.terminate()

            self.reading_done = True
            self.read_buffer.put(None)

    def interpolate_frames(self):
        prev_frame = None
        self.interpolation_done = False
        try:
            while True:
                frame = self.read_buffer.get()
                if frame is None:
                    if self.reading_done == True and self.read_buffer.empty():
                        break
                if prev_frame is not None:
                    self.interpolate_process.run(prev_frame, frame)
                    for i in range(self.interpolate_factor - 1):
                        result = self.interpolate_process.make_inference(
                            (i + 1) * 1. / (self.interpolate_factor + 1))
                        self.interpolated_frames.put_nowait(result)
                    prev_frame = frame
                else:
                    self.processed_frames.put(frame)
                    prev_frame = frame

        except Exception as e:
            logging.exception(f"An error occurred during interpolation {e}")

        finally:
            self.interpolation_done = True
            self.interpolated_frames.put(None)

    def blend_frames(self):
        self.processing_done = False
        frame_buffer = []
        try:
            while True:
                frame = self.interpolated_frames.get()
                if frame is None:
                    if self.interpolated_frames.qsize() < self.interpolate_factor and self.interpolation_done == True:
                        break
                frame_buffer.append(frame)
                if len(frame_buffer) == self.interpolate_factor:
                    motion_blur_frame = np.zeros_like(
                        frame, dtype=np.float32)

                    weights = np.exp(-0.5 * (np.linspace(-1, 1,
                                     self.interpolate_factor)**2))
                    weights /= np.sum(weights)

                    total_weight = 0

                    for i in range(self.interpolate_factor):
                        weight = weights[i]
                        motion_blur_frame += weight * frame_buffer[i]
                        total_weight += weight
                    motion_blur_frame /= total_weight
                    motion_blur_frame = motion_blur_frame.astype(np.uint8)

                    self.processed_frames.put_nowait(motion_blur_frame)
                    frame_buffer = []

        except Exception as e:
            logging.exception(f"An error occurred during blending {e}")

        finally:
            self.processing_done = True
            self.processed_frames.put(None)
            
    def clear_write_buffer(self):
        from src.ffmpegSettings import encodeSettings

        command: list = encodeSettings(self.encode_method, self.width, self.height,
                                       self.fps, self.output, self.ffmpeg_path, False, 0)

        pipe = subprocess.Popen(
            command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        try:
            while True:
                frame = self.processed_frames.get()
                if frame is None:
                    if self.processing_done == True and self.processed_frames.empty():
                        break

                frame = np.ascontiguousarray(frame)
                pipe.stdin.write(frame.tobytes())
                self.pbar.update(1)

        except Exception as e:
            logging.exception(f"An error occurred during writing {e}")

        finally:
            pipe.stdin.close()
            pipe.wait()
            self.pbar.close()
