import os
import argparse
import _thread
import time
import logging

from threading import Lock
from tqdm import tqdm
from moviepy.editor import VideoFileClip
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from multiprocessing import Queue
from concurrent.futures import ThreadPoolExecutor

"""
I have absolutely no clue how to avoid race conditions,

I have attempted with locks but I am not sure if I am doing it right,

Also attempted turning the interpolation output into a different dictionary and then checking for each index
if there is a frame at index + int but that didn't work either

It seems like MT is not going to be possible right now.

"""

<<<<<<< HEAD

class Main:
=======
class main:
>>>>>>> 36340abe592881990b8485ea3f31757db570dbe8
    def __init__(self, args):
        self.input = os.path.normpath(args.input)
        self.output = os.path.normpath(args.output)
        self.interpolate = args.interpolate
        self.interpolate_factor = args.interpolate_factor
        self.upscale = args.upscale
        self.upscale_factor = args.upscale_factor
<<<<<<< HEAD
        self.upscale_method = args.upscale_method.lower()
=======
        self.upscale_method = args.upscale_method
>>>>>>> 36340abe592881990b8485ea3f31757db570dbe8
        self.cugan_kind = args.cugan_kind
        self.dedup = args.dedup
        self.dedup_sens = args.dedup_sens
        self.dedup_method = args.dedup_method
        self.nt = args.nt
        self.lock = Lock()
        self.half = args.half

<<<<<<< HEAD
        self.intitialize()
        self.threads_are_running = True
        with ThreadPoolExecutor(max_workers=1) as executor:
            futures = {executor.submit(self.start_process)
                       for _ in range(self.nt)}

        while self.read_buffer.qsize() > 0 or self.processed_frames.qsize() > 0:
            time.sleep(0.1)
=======
        self.processed_frames = Queue()

        self.intitialize()

        self.threads = []
        self.threads_are_running = True
        
        with ThreadPoolExecutor(max_workers=self.nt) as executor:
            for _ in range(self.nt):
                thread = executor.submit(self.start_process)
                self.threads.append(thread)
>>>>>>> 36340abe592881990b8485ea3f31757db570dbe8

        while not all(thread.done() for thread in self.threads):
            time.sleep(0.01)
            
        self.threads_are_running = False

        if self.dedup_method == "FFMPEG":
            if self.interpolate or self.upscale:
                os.remove(self.input)
<<<<<<< HEAD

=======
                
>>>>>>> 36340abe592881990b8485ea3f31757db570dbe8
    def intitialize(self):

        if self.dedup:
            if self.dedup_method == "SSIM":
                from dedup import DedupSSIM
<<<<<<< HEAD
                logging.info(f" {self.dedup_method} is not available yet")
                # self.dedup_process = DedupSSIM()

            if self.dedup_method == "MSE":
                from dedup import DedupMSE
                logging.info(f" {self.dedup_method} is not available yet")
                # self.dedup_process = DedupMSE()

            if self.dedup_method == "FFmpeg":
                from src.dedup.dedup import DedupFFMPEG
                self.dedup_process = DedupFFMPEG(self.input, self.output)
                self.input = self.dedup_process.run()
=======
                print("Not available yet")
                # self.dedup = DedupSSIM()

            if self.dedup_method == "MSE":
                from dedup import DedupMSE
                print("Not available yet")
                # self.dedup = DedupMSE()

            if self.dedup_method == "FFMPEG":
                from src.dedup.dedup import DedupFFMPEG
                self.dedup = DedupFFMPEG(self.input, self.output)
                self.input = self.dedup.run()
>>>>>>> 36340abe592881990b8485ea3f31757db570dbe8

        # Metadata needs a little time to be written.
        time.sleep(1)

        self.video = VideoFileClip(self.input)
        self.frames = self.video.iter_frames()
        self.ffmpeg_params = ["-c:v", "libx264", "-preset", "fast", "-crf",
                              "15", "-tune", "animation", "-movflags", "+faststart", "-y"]
        
<<<<<<< HEAD
        self.fps = self.video.fps * \
            self.interpolate_factor if self.interpolate else self.video.fps
=======
        self.fps = self.video.fps * self.interpolate_factor if self.interpolate else self.video.fps
>>>>>>> 36340abe592881990b8485ea3f31757db570dbe8

        self.frame_size = (self.video.w * self.upscale_factor, self.video.h *
                           self.upscale_factor) if self.upscale else (self.video.w, self.video.h)

        self.writer = FFMPEG_VideoWriter(
            self.output, self.frame_size, self.fps, ffmpeg_params=self.ffmpeg_params)

<<<<<<< HEAD
        if self.upscale:
            if self.upscale_method == "shufflecugan" or "cugan":
                from src.cugan.cugan_node import Cugan
                self.upscale_process = Cugan(
                    self.upscale_method, self.upscale_factor, self.cugan_kind, self.half)
            elif self.upscale_method == "compact" or "ultracompact":
                from src.compact.compact import Compact
                self.upscale_process = Compact(
                    self.upscale_method, self.upscale_factor, self.half)
            elif self.upscale_method == "swinir":
                logging.info(f"{self.upscale_method}, not yet implemented")
            else:
                logging.info(f"There was an error in choosing the upscale method, {self.upscale_method} is not a valid option")

=======
        self.pbar = tqdm(total=self.video.reader.nframes,
                         desc="Processing", unit="frames", colour="green")

        if self.upscale:
            if self.upscale_method == "shufflecugan" or "cugan":
                from src.cugan.cugan_node import Cugan
                self.upscale = Cugan(self.upscale_method, self.upscale_factor, self.cugan_kind, self.half)
            else:
                print("not yet implemented")
                
>>>>>>> 36340abe592881990b8485ea3f31757db570dbe8
        if self.interpolate:
            from src.rife.rife import Rife
            UHD = True if self.frame_size[0] > 3840 or self.frame_size[1] > 2160 else False

<<<<<<< HEAD
            self.interpolate_process = Rife(
                self.interpolate_factor, self.half, self.frame_size, UHD)
            self.pbar = tqdm(total=self.video.reader.nframes * self.interpolate_factor,
                             desc="Processing", unit="frames", colour="green") if self.interpolate else self.pbar
        else:
            self.pbar = tqdm(total=self.video.reader.nframes,
                             desc="Processing", unit="frames", colour="green")
=======
            self.interpolate = Rife(
                self.interpolate_factor, self.half, self.frame_size, UHD)

>>>>>>> 36340abe592881990b8485ea3f31757db570dbe8

        self.read_buffer = Queue(maxsize=500)
        self.processed_frames = Queue(maxsize=500)

        _thread.start_new_thread(self.build_buffer, ())
<<<<<<< HEAD
        _thread.start_new_thread(self.clear_write_buffer, ())

    def build_buffer(self):
        for frame in self.frames:
            self.read_buffer.put((frame))

        for _ in range(self.nt):
            self.read_buffer.put((None))  # Put two Nones into the queue

    def start_process(self):
        prev_frame = None
        try:
            while True:
                frame = self.read_buffer.get()
                if frame is None:
                    break
                
                if self.upscale:
                    frame = self.upscale_process.run(frame)

                if self.interpolate: 
                    if prev_frame is not None:
                        results = self.interpolate_process.run(
                            prev_frame, frame, self.interpolate_factor, self.frame_size)
                        for result in results:
                            self.processed_frames.put(result)
                        prev_frame = frame
                    else:
                        prev_frame = frame

                self.processed_frames.put((frame))
        except Exception as e:
            raise e

    def clear_write_buffer(self):
        self.processing_index = 0
        while True:
            if self.processed_frames.empty():
                if self.read_buffer.empty() and self.threads_are_running == False:
                    break
                else:
                    continue

            frame = self.processed_frames.get(self.processing_index)

            """
            Attempt to write interpolated frames using MT, but I run into race conditions regardless of what I do
            if self.interpolate:
                counter = 0.001
                while True:
                    if self.interpolation_queue:
                        while index + counter in self.interpolation_queue:
                            frame = self.interpolation_queue.pop(index + counter)
                            self.writer.write_frame(frame)
                            counter += 0.001
                        break 
                
            """
            self.writer.write_frame(frame)
            self.processing_index += 1
            self.pbar.update(1)
            
        self.writer.close()
        self.video.close()
        self.pbar.close()

=======
        _thread.start_new_thread(self.write_buffer, ())

    def build_buffer(self):
        for index, frame in enumerate(self.frames):
            if frame is None:
                break
                
            self.read_buffer.put((index, frame))

        for _ in range(self.nt):
            self.read_buffer.put(None)

        self.video.close()

    def start_process(self):
        prev_frame = None
        while True:
            item = self.read_buffer.get()
            if item is None:
                print("\nThread finished")
                break
            index, frame = item
            if self.upscale:
                frame = self.upscale.run(frame)
                
            if self.interpolate and prev_frame is not None:
                results = self.interpolate.run(
                    prev_frame, frame, self.interpolate_factor, self.frame_size)
                if results is not None:
                    for result in results:
                        with self.lock:
                            self.processed_frames.put((index, result))
                else:
                    print("\nFailed to interpolate frame")
                    
            self.pbar.update(1)
            with self.lock:
                self.processed_frames.put((index, frame))
            prev_frame = frame

    def write_buffer(self):
        buffer = []
        while True:
            while not self.processed_frames.empty():
                with self.lock:
                    frame = self.processed_frames.get()
                buffer.append(frame)

            buffer.sort(key=lambda x: x[0])

            for _, frame in buffer:
                self.writer.write_frame(frame)

            buffer = []

            if not self.threads_are_running and self.processed_frames.empty():
                print("\nWrite_buffer finished")
                break

            time.sleep(0.1)

        self.writer.close()
        self.pbar.close()
>>>>>>> 36340abe592881990b8485ea3f31757db570dbe8

if __name__ == "__main__":
    log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'log.txt')
    logging.basicConfig(filename=log_file_path, filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input", type=str, required=True)
    argparser.add_argument("--output", type=str, required=True)
    argparser.add_argument("--interpolate", type=bool, default=False)
    argparser.add_argument("--interpolate_factor", type=int, default=2)
    argparser.add_argument("--upscale", type=bool, default=False)
    argparser.add_argument("--upscale_factor", type=int, default=2)
    argparser.add_argument("--upscale_method", type=str,
                           default="shufflecugan")
    argparser.add_argument("--cugan_kind", type=str, default="no-denoise")
    argparser.add_argument("--dedup", type=bool, default=False)
    argparser.add_argument("--dedup_sens", type=int, default=5)
    argparser.add_argument("--dedup_method", type=str, default="FFMPEG")
    argparser.add_argument("--nt", type=int, default=2)
    argparser.add_argument("--half", type=bool, default=True)

    args = argparser.parse_args()

    args_dict = vars(args)
    for arg in args_dict:
        logging.info(f"{arg}: {args_dict[arg]}")
        
    if args.input is not None:
        Main(args)
    else:
<<<<<<< HEAD
        logging.info("No input file specified")
=======
        print("Please select a video file")
    

>>>>>>> 36340abe592881990b8485ea3f31757db570dbe8
