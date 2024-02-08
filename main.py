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
import warnings
import sys
import logging
import numpy as np
import subprocess
import time

from tqdm import tqdm
from queue import Queue
from concurrent.futures import ThreadPoolExecutor

from src.checkSpecs import checkSystem
from src.getVideoMetadata import getVideoMetadata
from src.initializeModels import intitialize_models

if getattr(sys, 'frozen', False):
    main_path = os.path.dirname(sys.executable)
else:
    main_path = os.path.dirname(os.path.abspath(__file__))

scriptVersion = "1.1.7"
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
        self.availableRam = args.availableRam
        self.custom_model = args.custom_model
        self.custom_encoder = args.custom_encoder
        self.nt = args.nt
        
        self.width, self.height, self.fps, self.nframes = getVideoMetadata(
            self.input, self.inpoint, self.outpoint)

        logging.info(
            "\n============== Processing Outputs ==============")

        if self.resize:
            aspect_ratio = self.width / self.height
            self.width = round(self.width * self.resize_factor / 2) * 2
            self.height = round(self.width / aspect_ratio / 2) * 2
            
            logging.info(
                f"Resizing to {self.width}x{self.height}, aspect ratio: {aspect_ratio}")

        if self.scenechange:
            from src.scenechange.scene_change import Scenechange

            logging.info(
                "Detecting scene changes")

            scenechange = Scenechange(
                self.input, self.scenechange_sens, main_path, self.inpoint, self.outpoint)

            scenechange.run()

            return

        if self.depth:
            from src.depth.depth import Depth

            logging.info(
                "Detecting depth")

            Depth(self.input, self.output, self.ffmpeg_path, self.width, self.height,
                  self.fps, self.nframes, self.half, self.inpoint, self.outpoint, self.encode_method, self.depth_method, self.custom_encoder, self.availableRam)

            return

        if self.segment:
            from src.segment.segment import Segment

            logging.info(
                "Segmenting video")

            Segment(self.input, self.output, self.ffmpeg_path, self.width,
                    self.height, self.fps, self.nframes, self.inpoint, self.outpoint, self.encode_method, self.custom_encoder, self.availableRam)

            return

        if self.motion_blur:
            from src.motionblur.motionblur import motionBlur

            logging.info(
                "Adding motion blur")

            motionBlur(self.input, self.output, self.ffmpeg_path, self.width,
                       self.height, self.fps, self.nframes, self.inpoint, self.outpoint, self.interpolate_method, self.interpolate_factor, self.half, self.encode_method, self.dedup, self.dedup_sens, self.custom_encoder)

            return

        # If the user only wanted dedup / dedup + sharpen, we can skip the rest of the code and just run the dedup function from within FFMPEG
        if self.interpolate == False and self.upscale == False:
            if self.dedup == True or self.resize == True:
                filters = []

                if self.sharpen:
                    filters.append(f'cas={self.sharpen_sens}')

                if self.resize:
                    filters.append(f"scale={self.width}x{self.height}:flags={self.resize_method}")

                if self.dedup:
                    filters.append(f'{self.dedup_sens}')
                    
                logging.info(
                    "Deduping video")

                if self.outpoint != 0:
                    from src.dedup.dedup import trim_input_dedup
                    trim_input_dedup(self.input, self.output, self.inpoint,
                                    self.outpoint, filters, self.ffmpeg_path, self.encode_method)

                else:
                    from src.dedup.dedup import dedup_ffmpeg
                    dedup_ffmpeg(self.input, self.output,
                                filters, self.ffmpeg_path, self.encode_method)

                return
        
        self.start()
        
    def start(self):
        self.new_width, self.new_height, self.upscale_process, self.interpolate_process = intitialize_models(self)
        self.fps = self.fps * self.interpolate_factor if self.interpolate else self.fps 
        self.pbar = tqdm(total=self.nframes, desc="Processing Frames",
                            unit="frames", colour="green")
        
        self.read_buffer = Queue()
        self.processed_frames = Queue()
        
        with ThreadPoolExecutor(max_workers= 3) as executor:
            executor.submit(self.build_buffer)
            executor.submit(self.process)
            executor.submit(self.write_buffer)
        

    def build_buffer(self):
        from src.ffmpegSettings import decodeSettings

        command: list = decodeSettings(
            self.input, self.inpoint, self.outpoint, self.dedup, self.dedup_sens, self.ffmpeg_path, self.resize, self.width, self.height, self.resize_method)

        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        self.reading_done = False
        frame_size = self.width * self.height * 3
        frame_count = 0
        
        # See issue https://github.com/NevermindNilas/TheAnimeScripter/issues/10
        buffer_limit = 250 if self.availableRam < 8 else 500 if self.availableRam < 16 else 1000
        try:
            for chunk in iter(lambda: process.stdout.read(frame_size), b''):
                if len(chunk) != frame_size:
                    logging.error(
                        f"Read {len(chunk)} bytes but expected {frame_size}")
                    continue
                frame = np.frombuffer(chunk, dtype=np.uint8).reshape(
                    (self.height, self.width, 3))

                if self.read_buffer.qsize() > buffer_limit:
                    while self.processed_frames.qsize() > buffer_limit:
                        time.sleep(1)

                self.read_buffer.put(frame)
                frame_count += 1
                
        except Exception as e:
            logging.exception(
                f"Something went wrong while reading the frames, {e}")
        finally:
            logging.info(
                f"Built buffer with {frame_count} frames")
            if self.interpolate:
                frame_count = frame_count * self.interpolate_factor

            process.stdout.close()
            process.terminate()
            self.pbar.total = frame_count
            self.pbar.refresh()
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
            logging.exception(
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
                                       self.fps, self.output, self.ffmpeg_path, self.sharpen, self.sharpen_sens, self.custom_encoder, grayscale=False)

        pipe = subprocess.Popen(
            command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        frame_count = 0
        try:
            while True:
                frame = self.processed_frames.get()
                if frame is None:
                    if self.processing_done == True and self.processed_frames.empty() and self.reading_done == True:
                        break
                    else:
                        continue

                frame_count += 1
                frame = np.ascontiguousarray(frame)
                pipe.stdin.write(frame.tobytes())
                self.pbar.update()

        except Exception as e:
            logging.exception(
                f"Something went wrong while writing the frames, {e}")

        finally:
            logging.info(
                f"Wrote {frame_count} frames")

            pipe.stdin.close()
            self.pbar.close()

            stderr_output = pipe.stderr.read().decode()
            
            logging.info(
                "\n============== FFMPEG Output Log ============")
            
            if stderr_output:
                logging.info(stderr_output)

            # Hope this works
            pipe.terminate()

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
                           "shufflecugan", "shufflecugan_directml", "cugan", "compact", "ultracompact", "superultracompact", "swinir", "span", "span-ncnn", "cugan-ncnn", "omnisr"], default="shufflecugan")
    argparser.add_argument("--cugan_kind", type=str, choices=[
                           "no-denoise", "conservative", "denoise1x", "denoise2x"], default="no-denoise")
    argparser.add_argument("--custom_model", type=str, default="")
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
    argparser.add_argument("--encode_method", type=str, choices=["x264", "x264_animation", "x265", "nvenc_h264",
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
        "point", "spline", "spline16", "spline36"], default="bicubic", help="Choose the desired resizer, I am particularly happy with lanczos for upscaling and area for downscaling")
    argparser.add_argument("--custom_encoder", type=str, default="")
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
    args.scenechange_sens = 100 - args.scenechange_sens  # To keep up the previous logic where 0 is the least sensitive and 100 is the most sensitive

    logging.info("============== Arguments ==============")

    args_dict = vars(args)
    for arg in args_dict:
        logging.info(f"{arg.upper()}: {args_dict[arg]}")

    logging.info("\n============== Arguments Checker ==============")
    if args.interpolate:
        if args.nt >= 2:
            logging.info(
                f"Interpolation is enabled, setting nt to 1")
            args.nt = 1
    
    if args.dedup:
        from src.ffmpegSettings import get_dedup_strength
        # Dedup Sens will be overwritten with the mpdecimate params in order to minimize on the amount of variables used throughout the script
        args.dedup_sens = get_dedup_strength(args.dedup_sens)
        logging.info(f"Setting dedup params to {args.dedup_sens}")

    if args.output is None:
        from src.generateOutput import outputNameGenerator
        args.output = outputNameGenerator(args, main_path)

    if not args.ytdlp == "":
        logging.info(f"Downloading {args.ytdlp} video")
        from src.ytdlp import VideoDownloader
        VideoDownloader(args.ytdlp, args.output, args.ytdlp_quality, args.encode_method, args.custom_encoder)
        sys.exit()

    args.ffmpeg_path = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), "src", "ffmpeg", "ffmpeg.exe")

    if not os.path.exists(args.ffmpeg_path):
        from src.get_ffmpeg import get_ffmpeg
        args.ffmpeg_path = get_ffmpeg()

    args.availableRam = checkSystem()
    
    if args.input is not None:
        videoProcessor(args)
    else:
        print("No input was specified, exiting")
        logging.info("No input was specified, exiting")
