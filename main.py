"""
    The Anime Scripter is a tool that allows you to automate the process of
    Video Upscaling, Interpolating and many more all inside of the Adobe Suite
    Copyright (C) 2023-present Nilas Tiago

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as
    published by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see {http://www.gnu.org/licenses/}.

    Home: https://github.com/NevermindNilas/TheAnimeScripter
"""

import os
import argparse
import subprocess
import numpy as np
import warnings
import sys
import logging

from tqdm import tqdm
from queue import Queue
from concurrent.futures import ThreadPoolExecutor

from src.checkSpecs import checkSystem
from src.getVideoMetadata import getVideoMetadata
# Some default values

if getattr(sys, 'frozen', False):
    main_path = os.path.dirname(sys.executable)
else:
    main_path = os.path.dirname(os.path.abspath(__file__))

scriptVersion = "0.2.2"
warnings.filterwarnings("ignore")


class videoProcessor:
    def __init__(self, args):
        self.input = args.input
        self.output = args.output
        self.interpolate = args.interpolate
        self.interpolate_factor = args.interpolate_factor
        self.interpolate_method = args.interpolate_method
        self.upscale = args.upscale
        self.upscale_factor = args.upscale_factor
        self.upscale_method = args.upscale_method
        self.cugan_kind = args.cugan_kind
        self.dedup = args.dedup
        self.dedup_method = args.dedup_method
        self.dedup_sens = args.dedup_sens
        self.half = args.half
        self.inpoint = args.inpoint
        self.outpoint = args.outpoint
        self.sharpen = args.sharpen
        self.sharpen_sens = args.sharpen_sens
        self.segment = args.segment
        self.scenechange = args.scenechange
        self.scenechange_sens = args.scenechange_sens
        self.depth = args.depth
        self.depth_method = args.depth_method
        self.encode_method = args.encode_method
        self.motion_blur = args.motion_blur
        self.ffmpeg_path = args.ffmpeg_path
        self.ensemble = args.ensemble
        self.resize = args.resize
        self.resize_factor = args.resize_factor
        self.resize_method = args.resize_method

        logging.info(
            "\n============== Video Metadata ==============")
        self.width, self.height, self.fps, self.nframes = getVideoMetadata(
            self.input, self.inpoint, self.outpoint)

        logging.info(
            "\n============== Processing Outputs ==============")

        if self.resize:
            aspect_ratio = self.width / self.height
            self.width = int(self.width * self.resize_factor)
            self.height = int(self.width / aspect_ratio)

        if self.scenechange:
            from src.scenechange.scene_change import Scenechange

            logging.info(
                "Detecting scene changes")

            scenechange = Scenechange(
                self.input, self.ffmpeg_path, self.scenechange_sens, main_path)

            scenechange.run()

            return

        if self.depth:
            from src.depth.depth import Depth

            logging.info(
                "Detecting depth")

            Depth(self.input, self.output, self.ffmpeg_path, self.width, self.height,
                  self.fps, self.nframes, self.half, self.inpoint, self.outpoint, self.encode_method, self.depth_method)

            return

        if self.segment:
            from src.segment.segment import Segment

            logging.info(
                "Segmenting video")

            Segment(self.input, self.output, self.ffmpeg_path, self.width,
                    self.height, self.fps, self.nframes, self.inpoint, self.outpoint, self.encode_method)

            return

        if self.motion_blur:
            from src.motionblur.motionblur import motionBlur

            logging.info(
                "Adding motion blur")

            motionBlur(self.input, self.output, self.ffmpeg_path, self.width,
                       self.height, self.fps, self.nframes, self.inpoint, self.outpoint, self.interpolate_method, self.interpolate_factor, self.half, self.encode_method, self.dedup, self.dedup_sens)

            return

        # If the user only wanted dedup / dedup + sharpen, we can skip the rest of the code and just run the dedup function from within FFMPEG
        if self.interpolate == False and self.upscale == False and self.dedup == True:
            if self.sharpen == True:
                self.dedup_sens += f',cas={self.sharpen_sens}'

            logging.info(
                "Deduping video")

            if self.outpoint != 0:
                from src.dedup.dedup import trim_input_dedup
                trim_input_dedup(self.input, self.output, self.inpoint,
                                 self.outpoint, self.dedup_sens, self.ffmpeg_path, self.encode_method)

            else:
                from src.dedup.dedup import dedup_ffmpeg
                dedup_ffmpeg(self.input, self.output,
                             self.dedup_sens, self.ffmpeg_path, self.encode_method)

            return
        self.intitialize_models()
        self.start()

    def intitialize_models(self):
        self.new_width = self.width
        self.new_height = self.height
        self.fps = self.fps * self.interpolate_factor if self.interpolate else self.fps

        if self.upscale:
            self.new_width *= self.upscale_factor
            self.new_height *= self.upscale_factor
            logging.info(
                f"Upscaling to {self.new_width}x{self.new_height}")

            match self.upscale_method:
                case "shufflecugan" | "cugan":
                    from src.cugan.cugan import Cugan
                    self.upscale_process = Cugan(
                        self.upscale_method, int(self.upscale_factor), self.cugan_kind, self.half, self.width, self.height)
                case "cugan-ncnn":
                    from src.cugan.cugan import CuganNCNN
                    self.upscale_process = CuganNCNN(
                        1, self.upscale_factor)
                case "compact" | "ultracompact" | "superultracompact":
                    from src.compact.compact import Compact
                    self.upscale_process = Compact(
                        self.upscale_method, self.half, self.width, self.height)
                case "swinir":
                    from src.swinir.swinir import Swinir
                    self.upscale_process = Swinir(
                        self.upscale_factor, self.half, self.width, self.height)
                case "span":
                    from src.span.span import SpanSR
                    self.upscale_process = SpanSR(
                        self.upscale_factor, self.half, self.width, self.height)
                case "omnisr":
                    from src.omnisr.omnisr import OmniSR
                    self.upscale_process = OmniSR(
                        self.upscale_factor, self.half, self.width, self.height)

        if self.interpolate:
            UHD = True if self.new_width >= 3840 and self.new_height >= 2160 else False
            match self.interpolate_method:
                case "rife" | "rife4.6" | "rife4.13-lite" | "rife4.14-lite" | "rife4.14":
                    from src.rife.rife import Rife

                    self.interpolate_process = Rife(
                        int(self.interpolate_factor), self.half, self.new_width, self.new_height, UHD, self.interpolate_method, self.ensemble)
                case "rife-ncnn" | "rife4.6-ncnn" | "rife4.13-lite-ncnn" | "rife4.14-lite-ncnn" | "rife4.14-ncnn":
                    from src.rifencnn.rifencnn import rifeNCNN
                    self.interpolate_process = rifeNCNN(
                        UHD, self.interpolate_method, self.ensemble)

                case "gmfss":
                    from src.gmfss.gmfss_fortuna_union import GMFSS
                    self.interpolate_process = GMFSS(
                        int(self.interpolate_factor), self.half, self.new_width, self.new_height, UHD, self.ensemble)

    def start(self):
        self.pbar = tqdm(total=self.nframes, desc="Processing Frames",
                         unit="frames", colour="green")

        self.read_buffer = Queue(maxsize=500)
        self.processed_frames = Queue()

        with ThreadPoolExecutor(max_workers=3) as executor:
            executor.submit(self.build_buffer)
            executor.submit(self.process)
            executor.submit(self.write_buffer)

    def build_buffer(self):
        from src.ffmpegSettings import decodeSettings

        command: list = decodeSettings(
            self.input, self.inpoint, self.outpoint, self.dedup, self.dedup_sens, self.ffmpeg_path, self.resize, self.resize_factor, self.resize_method)

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
            logging.info(
                f"Something went wrong while reading the frames, {e}")
        finally:
            logging.info(
                f"Built buffer with {frame_count} frames")
            if self.interpolate:
                frame_count = frame_count * self.interpolate_factor

            self.pbar.total = frame_count
            self.pbar.refresh()
            process.stdout.close()
            process.terminate()
            self.reading_done = True
            self.read_buffer.put(None)

    def process(self):
        prev_frame = None
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
                if self.upscale:
                    frame = self.upscale_process.run(frame)
                if self.interpolate:
                    if prev_frame is not None:
                        self.interpolate_process.run(prev_frame, frame)
                        for i in range(self.interpolate_factor - 1):
                            result = self.interpolate_process.make_inference(
                                (i + 1) * 1. / (self.interpolate_factor + 1))
                            self.processed_frames.put(result)
                            frame_count += 1
                        prev_frame = frame
                    else:
                        prev_frame = frame
                self.processed_frames.put(frame)
                frame_count += 1

        except Exception as e:
            logging.info(
                f"Something went wrong while processing the frames, {e}")

        finally:
            if prev_frame is not None:
                self.processed_frames.put(prev_frame)
                frame_count += 1

            logging.info(
                f"Processed {frame_count} frames")

            self.processing_done = True
            self.processed_frames.put(None)

    def write_buffer(self):

        from src.ffmpegSettings import encodeSettings
        command: list = encodeSettings(self.encode_method, self.new_width, self.new_height,
                                       self.fps, self.output, self.ffmpeg_path, self.sharpen, self.sharpen_sens, grayscale=False)

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
                frame = np.ascontiguousarray(frame)
                pipe.stdin.write(frame.tobytes())
                self.pbar.update()

        except Exception as e:
            logging.info(
                f"Something went wrong while writing the frames, {e}")

        finally:
            logging.info(
                f"Wrote {frame_count} frames")

            pipe.stdin.close()
            self.pbar.close()


if __name__ == "__main__":
    log_file_path = os.path.join(main_path, "log.txt")
    logging.basicConfig(filename=log_file_path, filemode='w',
                        format='%(message)s', level=logging.INFO)

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--version", action="store_true")
    argparser.add_argument("--input", type=str)
    argparser.add_argument("--output", type=str)
    argparser.add_argument("--interpolate", type=int,
                           choices=[0, 1], default=0)
    argparser.add_argument("--interpolate_factor", type=int, default=2)
    argparser.add_argument("--interpolate_method", type=str, choices=["rife", "rife4.6", "rife4.14", "rife4.14-lite", "rife4.13-lite",
                           "gmfss", "rife-ncnn", "rife4.6-ncnn", "rife4.13-lite-ncnn", "rife4.14-lite-ncnn", "rife4.14-ncnn"], default="rife")
    argparser.add_argument("--ensemble", type=int, choices=[0, 1], default=0)
    argparser.add_argument("--upscale", type=int, choices=[0, 1], default=0)
    argparser.add_argument("--upscale_factor", type=int,
                           choices=[2, 3, 4], default=2)
    argparser.add_argument("--upscale_method",  type=str, choices=[
                           "shufflecugan", "compact", "ultracompact", "superultracompact", "swinir", "span", "cugan-ncnn", "omnisr"], default="shufflecugan")
    argparser.add_argument("--cugan_kind", type=str, choices=[
                           "no-denoise", "conservative", "denoise1x", "denoise2x"], default="no-denoise")
    argparser.add_argument("--dedup", type=int, choices=[0, 1], default=0)
    argparser.add_argument("--dedup_method", type=str, default="ffmpeg")
    argparser.add_argument("--dedup_sens", type=float, default=35)
    argparser.add_argument("--nt", type=int, default=1)
    argparser.add_argument("--half", type=int, choices=[0, 1], default=1)
    argparser.add_argument("--inpoint", type=float, default=0)
    argparser.add_argument("--outpoint", type=float, default=0)
    argparser.add_argument("--sharpen", type=int, choices=[0, 1], default=0)
    argparser.add_argument("--sharpen_sens", type=float, default=50)
    argparser.add_argument("--segment", type=int, choices=[0, 1], default=0)
    argparser.add_argument("--scenechange", type=int,
                           choices=[0, 1], default=0)
    argparser.add_argument("--scenechange_sens", type=float, default=50)
    argparser.add_argument("--depth", type=int, choices=[0, 1], default=0)
    argparser.add_argument("--depth_method", type=str,
                           choices=["small", "base", "large"], default="small")
    argparser.add_argument("--encode_method", type=str, choices=["x264", "x264_animation", "nvenc_h264",
                           "nvenc_h265", "qsv_h264", "qsv_h265", "nvenc_av1", "av1", "h264_amf", "hevc_amf"], default="x264")
    argparser.add_argument("--motion_blur", type=int,
                           choices=[0, 1], default=0)
    argparser.add_argument("--ytdlp", type=str, default="")
    argparser.add_argument("--ytdlp_quality", type=int,
                           choices=[0, 1], default=0)
    argparser.add_argument("--resize", type=int, choices=[0, 1], default=0)
    argparser.add_argument("--resize_factor", type=float, default=2,
                           help="Resize factor for the decoded video, can also be values between 0 & 1 for downscaling though it needs more work, it will always keep the desired aspect ratio")
    argparser.add_argument("--resize_method", type=str, choices=[
        "fast_bilinear", "bilinear", "bicubic", "experimental", "neighbor", "area", "bicublin", "gauss", "sinc", "lanczos",
        "spline"], default="bicubic", help="Choose the desired resizer, I am particularly happy with lanczos for upscaling and area for downscaling")  # Thank god for ChatGPT
    args = argparser.parse_args()

    if args.version:
        print(scriptVersion)
        sys.exit()
    else:
        args.version = scriptVersion

    # Whilst this is ugly, it was easier to work with the Extendscript interface this way
    args.ytdlp_quality = True if args.ytdlp_quality == 1 else False
    args.interpolate = True if args.interpolate == 1 else False
    args.scenechange = True if args.scenechange == 1 else False
    args.ensemble = True if args.ensemble == 1 else False
    args.sharpen = True if args.sharpen == 1 else False
    args.upscale = True if args.upscale == 1 else False
    args.segment = True if args.segment == 1 else False
    args.resize = True if args.resize == 1 else False
    args.dedup = True if args.dedup == 1 else False
    args.depth = True if args.depth == 1 else False
    args.half = True if args.half == 1 else False

    args.sharpen_sens /= 100  # CAS works from 0.0 to 1.0
    args.scenechange_sens /= 100  # same for scene change

    logging.info("============== Arguments ==============")

    args_dict = vars(args)
    for arg in args_dict:
        logging.info(f"{arg.upper()}: {args_dict[arg]}")

    logging.info("\n============== Arguments Checker ==============")

    if args.upscale_factor not in [2, 3, 4] or (args.upscale_method in ["shufflecugan", "compact", "ultracompact", "superultracompact", "swinir", "span"] and args.upscale_factor != 2):
        logging.info(
            f"Invalid upscale factor for {args.upscale_method}. Setting upscale_factor to 2.")
        args.upscale_factor = 2

    if args.dedup:
        from src.ffmpegSettings import get_dedup_strength
        # Dedup Sens will be overwritten with the mpdecimate params in order to minimize on the amount of variables used throughout the script
        args.dedup_sens = get_dedup_strength(args.dedup_sens)
        logging.info(f"Setting dedup params to {args.dedup_sens}")

    if args.output is None:
        logging.info("No output was specified, generating output name")
        import random
        randomNumber = random.randint(0, 100000)

        if not os.path.exists(os.path.join(main_path, "output")):
            os.makedirs(os.path.join(main_path, "output"), exist_ok=True)

        args.output = os.path.join(
            main_path, "output", "TAS" + str(randomNumber) + ".mp4")

        logging.info(f"Output name: {args.output}")

    if not args.ytdlp == "":
        logging.info(f"Downloading {args.ytdlp} video")
        from src.ytdlp import ytdlp
        ytdlp(args.ytdlp, args.output, args.ytdlp_quality, args.encode_method)
        sys.exit()

    args.ffmpeg_path = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), "src", "ffmpeg", "ffmpeg.exe")

    if not os.path.exists(args.ffmpeg_path):
        from src.get_ffmpeg import get_ffmpeg
        args.ffmpeg_path = get_ffmpeg()

    logging.info("\n============== System Checker ==============")
    checkSystem()
    
    if args.input is not None:
        videoProcessor(args)
    else:
        print("No input was specified, exiting")
        logging.info("No input was specified, exiting")
