import os
import argparse
import _thread
import time
import logging

from tqdm import tqdm
from moviepy.editor import VideoFileClip
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from multiprocessing import Queue
from concurrent.futures import ThreadPoolExecutor
from collections import deque


"""
I have absolutely no clue how to avoid race conditions,

I have attempted with locks but I am not sure if I am doing it right,

Also attempted turning the interpolation output into a different dictionary and then checking for each index
if there is a frame at index + int but that didn't work either

It seems like MT is not going to be possible right now.

TO:DO

    - FFMPEG has a to bytes feature through piping, maybe use that instead of having to decode - encode - decode and encode again.
    - Add back MT support but for upscaling only.
    - Provide a bundled version with all of the dependencies included ( need assitance with this ).
    - Add more functionalities to Cugan-AMD.
    - Fix SwinIR.
    - Improve performance.
    - Add testing.
    - Add DepthMap process.
    - Maybe add TRT.
"""

ffmpeg_params = ["-c:v", "libx264", "-preset", "veryfast", "-crf",
                 "15", "-tune", "animation", "-movflags", "+faststart", "-y"]


class Main:
    def __init__(self, args):
        self.input = os.path.normpath(args.input)
        self.output = os.path.normpath(args.output)
        self.interpolate = args.interpolate
        self.interpolate_factor = args.interpolate_factor
        self.upscale = args.upscale
        self.upscale_factor = args.upscale_factor
        self.upscale_method = args.upscale_method.lower()
        self.cugan_kind = args.cugan_kind
        self.dedup = args.dedup
        self.dedup_sens = args.dedup_sens
        self.dedup_method = args.dedup_method
        self.nt = args.nt
        self.half = args.half
        self.inpoint = args.inpoint
        self.outpoint = args.outpoint

        self.Do_not_process = False

        self.intitialize()

        self.threads_are_running = True
        if self.Do_not_process:
            logging.info("The user has selected no other processing other than Dedup, exiting")
            return
        else:
            futures = []
            with ThreadPoolExecutor(max_workers=self.nt) as executor:
                executor.submit(self.start_process)

            while self.read_buffer.qsize() > 0 or len(self.processed_frames) > 0:
                time.sleep(0.1)

        self.threads_are_running = False

        if self.dedup_method == "ffmpeg" and self.Do_not_process == False:
            if self.interpolate or self.upscale:
                os.remove(self.input)

    def intitialize(self):
        
        if self.interpolate == False and self.upscale == False and self.dedup == True:
            self.Do_not_process = True
        
        if self.outpoint != 0:
            if self.dedup:
                from src.trim_input import trim_input_dedup
                self.trim_dedup_process = trim_input_dedup(
                    self.input, self.output, self.inpoint, self.outpoint, self.Do_not_process)
                self.input = self.trim_dedup_process.run()
                self.dedup = False
            else:
                from src.trim_input import trim_input
                self.trim_process = trim_input(
                    self.input, self.output, self.inpoint, self.outpoint, self.Do_not_process)
                self.input = self.trim_process.run()                
            
            logging.info(f"The new input is: {self.input}")
                
        if self.dedup:
            from src.dedup.dedup import DedupFFMPEG
            self.dedup_process = DedupFFMPEG(
                self.input, self.output, self.Do_not_process)
            self.input = self.dedup_process.run()
            logging.info(f"The new input is: {self.input}")

        if self.Do_not_process == True:
            return
        
        # Metadata needs a little time to be written.
        time.sleep(0.5)

        self.video = VideoFileClip(self.input)
        self.frames = self.video.iter_frames()

        self.fps = self.video.fps * \
            self.interpolate_factor if self.interpolate else self.video.fps

        self.frame_size = (self.video.w * self.upscale_factor, self.video.h *
                           self.upscale_factor) if self.upscale else (self.video.w, self.video.h)

        self.writer = FFMPEG_VideoWriter(
            self.output, self.frame_size, self.fps, ffmpeg_params=ffmpeg_params)

        if self.upscale:
            if self.upscale_method == "shufflecugan" or self.upscale_method == "cugan":
                from src.cugan.cugan import Cugan
                self.upscale_process = Cugan(
                    self.upscale_method, self.upscale_factor, self.cugan_kind, self.half)
            elif self.upscale_method == "cugan-amd":
                from src.cugan.cugan import CuganAMD
                self.upscale_process = CuganAMD(
                    self.nt, self.upscale_factor
                )
            elif self.upscale_method == "compact" or self.upscale_method == "ultracompact" or self.upscale_method == "superultracompact":
                from src.compact.compact import Compact
                self.upscale_process = Compact(
                    self.upscale_method, self.half)
            elif self.upscale_method == "swinir":
                from src.swinir.swinir import Swinir
                self.upscale_process = Swinir(
                    self.upscale_factor, self.half)
                print("processing swinir")
            else:
                logging.info(
                    f"There was an error in choosing the upscale method, {self.upscale_method} is not a valid option")

        if self.interpolate:
            from src.rife.rife import Rife
            UHD = True if self.frame_size[0] > 3840 or self.frame_size[1] > 2160 else False

            self.interpolate_process = Rife(
                self.interpolate_factor, self.half, self.frame_size, UHD)
            self.pbar = tqdm(total=self.video.reader.nframes * self.interpolate_factor,
                             desc="Processing", unit="frames", colour="green")
        else:
            self.pbar = tqdm(total=self.video.reader.nframes,
                             desc="Processing", unit="frames", colour="green")

        self.read_buffer = Queue(maxsize=500)
        self.processed_frames = deque()

        _thread.start_new_thread(self.build_buffer, ())
        _thread.start_new_thread(self.clear_write_buffer, ())

    def build_buffer(self):
        for frame in self.frames:
            self.read_buffer.put(frame)

        for _ in range(self.nt):
            self.read_buffer.put(None)

    def start_process(self):
        prev_frame = None
        try:
            while True:
                frame = self.read_buffer.get()
                if frame is None:
                    break

                if self.upscale:
                    frame = self.upscale_process.run(frame)

                if self.interpolate and prev_frame is not None:
                    results = self.interpolate_process.run(
                        prev_frame, frame, self.interpolate_factor, self.frame_size)
                    for result in results:
                        self.processed_frames.append(result)
                    prev_frame = frame
                elif self.interpolate:
                    prev_frame = frame

                self.processed_frames.append(frame)
        except Exception as e:
            logging.exception("An error occurred during processing")

    def clear_write_buffer(self):
        self.processing_index = 0
        while True:
            if not self.processed_frames:
                if self.read_buffer.empty() and self.threads_are_running == False:
                    break
                else:
                    continue

            frame = self.processed_frames.popleft()

            self.writer.write_frame(frame)
            self.processing_index += 1
            self.pbar.update(1)

        self.writer.close()
        self.video.close()
        self.pbar.close()


if __name__ == "__main__":

    log_file_path = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'log.txt')
    logging.basicConfig(filename=log_file_path, filemode='w',
                        format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input", type=str, required=True)
    argparser.add_argument("--output", type=str, required=True)
    argparser.add_argument("--interpolate", type=int, default=0)
    argparser.add_argument("--interpolate_factor", type=int, default=2)
    argparser.add_argument("--upscale", type=int, default=0)
    argparser.add_argument("--upscale_factor", type=int, default=2)
    argparser.add_argument("--upscale_method",  type=str,
                           default="ShuffleCugan")
    argparser.add_argument("--cugan_kind", type=str, default="no-denoise")
    argparser.add_argument("--dedup", type=int, default=0)
    argparser.add_argument("--dedup_sens", type=int, default=5)
    argparser.add_argument("--dedup_method", type=str, default="ffmpeg")
    argparser.add_argument("--nt", type=int, default=1)
    argparser.add_argument("--half", type=int, default=1)
    argparser.add_argument("--inpoint", type=float, default=0)
    argparser.add_argument("--outpoint", type=float, default=0)

    try:
        args = argparser.parse_args()
    except Exception as e:
        logging.info(e)

    """
    Whilst this is ugly, it was easier to work with the Extendscript interface this way
    """
    args.interpolate = True if args.interpolate == 1 else False
    args.upscale = True if args.upscale == 1 else False
    args.dedup = True if args.dedup == 1 else False
    args.half = True if args.half == 1 else False

    args.upscale_method = args.upscale_method.lower()
    args.cugan_kind = args.cugan_kind.lower()
    args.dedup_method = args.dedup_method.lower()

    args_dict = vars(args)
    for arg in args_dict:
        logging.info(f"{arg}: {args_dict[arg]}")

    if args.output and not os.path.isabs(args.output):
        dir_path = os.path.dirname(args.input)
        args.output = os.path.join(dir_path, args.output)

    if args.upscale_method in ["shufflecugan", "compact", "ultracompact", "superultracompact", "swinir"] and args.upscale_factor != 2:
        logging.info(
            f"{args.upscale_method} only supports 2x upscaling, setting upscale_factor to 2, please use Cugan for 3x/4x upscaling")
        args.upscale_factor = 2

    if args.upscale_factor not in [2, 3, 4]:
        logging.info(
            f"{args.upscale_factor} is not a valid upscale factor, setting upscale_factor to 2")
        args.upscale_factor = 2

    if args.nt > 1:
        logging.info(
            "Multi-threading is not supported yet, setting nt back to 1")
        args.nt = 1

    if args.input is not None and args.output is not None:
        Main(args)
    else:
        logging.info("No input or output was specified, exiting")
