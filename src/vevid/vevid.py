import subprocess
import logging
import _thread
import numpy as np
import torch

import cv2
from queue import Queue
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from phycv import VEVID_GPU


class Vevid():
    def __init__(self, input, output, inpoint, outpoint, nframes, dedup, dedup_sens, encode_method, ffmpeg_path, width, height, fps, half):
        self.input = input
        self.output = output
        self.inpoint = inpoint
        self.outpoint = outpoint
        self.dedup = dedup
        self.dedup_sens = dedup_sens
        self.encode_method = encode_method
        self.ffmpeg_path = ffmpeg_path
        self.width = width
        self.height = height
        self.fps = fps
        self.half = half
        self.nframes = nframes

        self.read_buffer = Queue(maxsize=500)
        self.processed_frames = Queue()

        self.handle_model()

        self.pbar = tqdm(total=self.nframes, desc="Processing",
                         unit="frames", unit_scale=True, colour="green")

        _thread.start_new_thread(self.build_buffer, ())
        _thread.start_new_thread(self.write_buffer, ())

        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(self.process)

    def handle_model(self):
        self.cuda_is_available = torch.cuda.is_available()

        self.device = "cuda" if self.cuda_is_available else "cpu"

        if self.cuda_is_available:
            from phycv import VEVID_GPU
            self.model = VEVID_GPU(device=self.device)
        else:
            from phycv import VEVID
            self.model = VEVID(device=self.device)

        """
        # Looking to initialize the kernel only once here
        # TO DO
        """

    def build_buffer(self):
        from src.ffmpegSettings import decodeSettings

        command: list = decodeSettings(
            self.input, self.inpoint, self.outpoint, self.dedup, self.dedup_sens, self.ffmpeg_path)

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

            if self.dedup:
                # This can and will add aditional delay to the pbar where it seems to be out of sync
                # with the actual writing thread
                self.pbar.total = frame_count
                self.pbar.refresh()

            process.stdout.close()
            process.terminate()

            self.reading_done = True
            self.read_buffer.put(None)

    def process(self):
        # For now these are fixed params but they will be adjustable once I can
        # Send these params in a more elegant way from AE
        b = 0.5
        G = 1.8
        self.processing_done = False
        frame_count = 0
        try:
            if self.cuda_is_available:
                while True:
                    frame = self.read_buffer.get()
                    if frame is None:
                        if self.reading_done == True and self.read_buffer.empty():
                            break
                        else:
                            continue
                    
                    frame = torch.from_numpy(frame)
                    frame = torch.permute(frame, (2, 0, 1)) / 255.0
                    frame = frame.to(self.device)
                    self.model.load_img(img_array=frame)
                    self.model.apply_kernel(b, G, lite=True)
                    frame = self.model.vevid_output.permute(1, 2, 0).cpu().numpy()
                    frame = (frame * 255).astype(np.uint8)

                    self.processed_frames.put(frame)
                    frame_count += 1
            
            else:
                pass

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
                                       self.fps, self.output, self.ffmpeg_path, False, 0)

        process = subprocess.Popen(
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
                frame = np.ascontiguousarray(frame)
                process.stdin.write(frame.tobytes())
                self.pbar.update()

        except Exception as e:
            logging.exception(
                f"An error occurred during reading, {e}")
            raise e

        finally:
            logging.info(
                f"Wrote {frame_count} frames")

            process.stdin.close()
            process.terminate()

            self.pbar.close()
