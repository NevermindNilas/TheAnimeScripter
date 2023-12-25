import os
import argparse
import _thread
import logging
import subprocess
import numpy as np
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from moviepy.editor import VideoFileClip
from multiprocessing import Queue
from collections import deque

"""
22/12/2023 - Massive refactoring compared to older iterations, expect more in the future

TO:DO
    - Add testing.
    - Fix Rife padding, again.
    - Play around with better mpdecimate params
    - Find a way to add awarpsharp2 to the pipeline
    - Add bounding box support for Segmentation
"""

warnings.filterwarnings("ignore")

ffmpeg_params = ["-c:v", "libx264", "-preset", "veryfast", "-crf",
                 "15", "-tune", "animation", "-movflags", "+faststart", "-y"]

mpdecimate_params = "mpdecimate=hi=64*24:lo=64*12:frac=0.1,setpts=N/FRAME_RATE/TB"


class Main:
    def __init__(self, args):
        # args is a dictionary
        self.input = os.path.normpath(args.input)
        self.output = os.path.normpath(args.output)
        self.interpolate = args.interpolate
        self.interpolate_factor = args.interpolate_factor
        self.upscale = args.upscale
        self.upscale_factor = args.upscale_factor
        self.upscale_method = args.upscale_method
        self.cugan_kind = args.cugan_kind
        self.dedup = args.dedup
        self.dedup_sens = args.dedup_sens
        self.dedup_method = args.dedup_method
        self.nt = args.nt
        self.half = args.half
        self.inpoint = args.inpoint
        self.outpoint = args.outpoint
        self.sharpen = args.sharpen
        self.sharpen_sens = args.sharpen_sens
        self.segment = args.segment

        # This is necessary on the top since the script heavily relies on FFMPEG, I will look into FFMPEG Reader from moviepy but it lacks the filtering abillity, so I will have to implement that myself using SSIM/MSE later on
        self.check_ffmpeg()
        
        if self.segment == True:
            from src.segment.segment import Segment
            
            self.get_video_metadata()
            
            process = Segment(self.input, self.output, self.ffmpeg_path, self.width, self.height, self.fps, self.nframes, self.inpoint, self.outpoint)
            result = process.run()
            logging.info("The user only wanted to segment, exiting after processing")
            return
        
        # There's no need to start the decode encode cycle if the user only wants to dedup
        # Therefore I just hand the input to ffmpeg and call upon mpdecimate
        if self.interpolate == False and self.upscale == False and self.dedup == True:
            if self.sharpen == True:
                mpdecimate_params += f',cas={self.sharpen_sens}'

            if self.outpoint != 0:
                from src.trim_input import trim_input_dedup
                trim_input_dedup(self.input, self.output, self.inpoint,
                                 self.outpoint, mpdecimate_params, self.ffmpeg_path).run()
            else:
                from src.dedup.dedup import DedupFFMPEG
                DedupFFMPEG(self.input, self.output,
                            mpdecimate_params, self.ffmpeg_path).run()

            logging.info("The user only wanted to dedup, exiting")
            return

        self.get_video_metadata()
        self.intitialize_models()
        self.intitialize()

        self.threads_done = False

        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(self.start_process)

        while self.read_buffer.qsize() > 0 and len(self.processed_frames) > 0:
            time.sleep(0.1)

        self.threads_done = True

    def intitialize(self):

        self.read_buffer = Queue(maxsize=500)
        self.processed_frames = deque()

        _thread.start_new_thread(self.build_buffer, ())
        _thread.start_new_thread(self.clear_write_buffer, ())

    def intitialize_models(self):

        self.new_width = self.width
        self.new_height = self.height
        self.fps = self.fps * self.interpolate_factor if self.interpolate else self.fps

        if self.upscale:

            # Setting new width and height for processing
            self.new_width *= self.upscale_factor
            self.new_height *= self.upscale_factor
            logging.info(
                f"Upscaling to {self.new_width}x{self.new_height}")

            if self.upscale_method == "shufflecugan" or self.upscale_method == "cugan":
                from src.cugan.cugan import Cugan
                self.upscale_process = Cugan(
                    self.upscale_method, int(self.upscale_factor), self.cugan_kind, self.half, self.width, self.height)

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
                    self.upscale_factor, self.half, self.width, self.height)

            else:
                logging.info(
                    f"There was an error in choosing the upscale method, {self.upscale_method} is not a valid option")

        if self.interpolate:
            from src.rife.rife import Rife

            UHD = True if self.new_width >= 3840 and self.new_height >= 2160 else False
            self.interpolate_process = Rife(
                int(self.interpolate_factor), self.half, self.new_width, self.new_height, UHD)

    def build_buffer(self):
        if self.interpolate:
            self.nframes *= self.interpolate_factor

        self.pbar = tqdm(
            total=None if self.dedup else self.nframes, desc="Processing Frames", unit="frames", dynamic_ncols=True, colour="green")

        ffmpeg_command = [
            self.ffmpeg_path,
            "-i", str(self.input),
        ]
        if self.outpoint != 0:
            ffmpeg_command.extend(
                ["-ss", str(self.inpoint), "-to", str(self.outpoint)])

        if self.dedup == True:
            ffmpeg_command.extend(
                ["-vf", mpdecimate_params, "-an"])

        ffmpeg_command.extend([
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-v", "quiet",
            "-stats",
            "-",
        ])

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

        stderr = process.stderr.read().decode()
        if stderr:
            if "bitrate=" not in stderr:
                logging.error(f"ffmpeg error: {stderr}")

        # For terminating the pipe and subprocess properly
        process.stdout.close()
        process.stderr.close()
        process.terminate()

        for _ in range(self.nt):
            self.read_buffer.put(None)

        logging.info(f"Read {frame_count} frames")

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
                            prev_frame, frame, self.interpolate_factor)
                        for result in results:
                            self.processed_frames.append(result)

                        prev_frame = frame
                    else:
                        prev_frame = frame

                self.processed_frames.append(frame)
        except Exception as e:
            logging.exception("An error occurred during processing")

    def clear_write_buffer(self):
        command = [self.ffmpeg_path,
                   '-y',
                   '-f', 'rawvideo',
                   '-vcodec', 'rawvideo',
                   '-s', f'{self.new_width}x{self.new_height}',
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

        if self.sharpen:
            command.insert(-1, '-vf')
            command.insert(-1, f'cas={self.sharpen_sens}')

        pipe = subprocess.Popen(
            command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        try:
            while True:
                if not self.processed_frames:
                    if self.read_buffer.empty() and self.threads_done == True:
                        break
                    else:
                        continue

                frame = self.processed_frames.popleft()

                # Write the frame to FFmpeg
                pipe.stdin.write(frame.tobytes())

                self.pbar.update(1)
        except Exception as e:
            logging.exception("An error occurred during writing")

        # Close the pipe
        pipe.stdin.close()
        pipe.wait()

        self.pbar.close()

    def get_video_metadata(self):
        clip = VideoFileClip(self.input)
        self.width = clip.size[0]
        self.height = clip.size[1]
        self.fps = clip.fps
        self.nframes = clip.reader.nframes

        logging.info(
            f"Video Metadata: {self.width}x{self.height} @ {self.fps}fps, {self.nframes} frames")

        clip.close()

    def check_ffmpeg(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))

        self.ffmpeg_path = os.path.join(dir_path, "ffmpeg", "ffmpeg.exe")

        # Check if FFMPEG exists at that path
        if not os.path.exists(self.ffmpeg_path):
            print("Couldn't find FFMPEG, downloading it now")
            print("This might add an aditional 1-5 seconds to the startup time of the process until FFMPEG is downloaded and caches are built, but it will only happen once")
            logging.info("The user doesn't have FFMPEG, downloading it now")
            ffmpeg_bat_location = os.path.join(dir_path, "get_ffmpeg.bat")
            subprocess.call(ffmpeg_bat_location, shell=True)


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
    argparser.add_argument("--sharpen", type=int, default=0)
    argparser.add_argument("--sharpen_sens", type=float, default=50)
    argparser.add_argument("--segment", type=int, default=0)

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
    args.sharpen = True if args.sharpen == 1 else False
    args.segment = True if args.segment == 1 else False

    args.upscale_method = args.upscale_method.lower()
    args.cugan_kind = args.cugan_kind.lower()
    args.dedup_method = args.dedup_method.lower()

    args.sharpen_sens /= 100  # CAS works from 0.0 to 1.0

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
