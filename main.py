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

from threading import Semaphore
from concurrent.futures import ThreadPoolExecutor
from src.argumentsChecker import argumentChecker
from src.getVideoMetadata import getVideoMetadata
from src.initializeModels import intitialize_models
from src.ffmpegSettings import BuildBuffer, WriteBuffer
from src.generateOutput import outputNameGenerator


if getattr(sys, "frozen", False):
    mainPath = os.path.dirname(sys.executable)
else:
    mainPath = os.path.dirname(os.path.abspath(__file__))

scriptVersion = "1.4.2"
warnings.filterwarnings("ignore")

"""
TO:DO
 - Look more into ESRGAN's poor performance compared to NCNN
 - Add shufflecugan-ncnn pipeline
"""


class VideoProcessor:
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
        self.ffmpeg_path = args.ffmpeg_path
        self.ensemble = args.ensemble
        self.resize = args.resize
        self.resize_factor = args.resize_factor
        self.resize_method = args.resize_method
        self.custom_model = args.custom_model
        self.custom_encoder = args.custom_encoder
        self.nt = args.nt
        self.buffer_limit = args.buffer_limit
        self.audio = args.audio
        self.denoise = args.denoise
        self.denoise_method = args.denoise_method
        self.sample_size = args.sample_size

        self.width, self.height, self.fps = getVideoMetadata(
            self.input, self.inpoint, self.outpoint
        )

        self.fps = (
            self.fps * self.interpolate_factor if self.interpolate else self.fps
        )
        
        logging.info("\n============== Processing Outputs ==============")

        if self.resize:
            aspect_ratio = self.width / self.height
            self.width = round(self.width * self.resize_factor / 2) * 2
            self.height = round(self.width / aspect_ratio / 2) * 2

            logging.info(
                f"Resizing to {self.width}x{self.height}, aspect ratio: {aspect_ratio}"
            )

        if self.scenechange:
            from src.scenechange.scene_change import Scenechange

            logging.info("Detecting scene changes")

            scenechange = Scenechange(
                self.input,
                self.scenechange_sens,
                mainPath,
                self.inpoint,
                self.outpoint,
            )

            scenechange.run()

            return

        if self.depth:
            from src.depth.depth import Depth

            logging.info("Detecting depth")

            Depth(
                self.input,
                self.output,
                self.ffmpeg_path,
                self.width,
                self.height,
                self.fps,
                self.half,
                self.inpoint,
                self.outpoint,
                self.encode_method,
                self.depth_method,
                self.custom_encoder,
                self.nt,
                self.buffer_limit,
            )

            return

        if self.segment:
            from src.segment.segment import Segment

            logging.info("Segmenting video")

            Segment(
                self.input,
                self.output,
                self.ffmpeg_path,
                self.width,
                self.height,
                self.fps,
                self.inpoint,
                self.outpoint,
                self.encode_method,
                self.custom_encoder,
                self.nt,
                self.buffer_limit,
            )

            return

        self.start()

    def processFrame(self, frame):
        try:
            if (
                self.dedup
                and self.dedup_process is not None
            ):
                result = self.dedup_process.run(frame)
                if result:
                    self.dedupCount += 1
                    self.semaphore.release()
                    return

            if self.upscale:
                frame = self.upscale_process.run(frame)

            if self.interpolate:
                run = self.interpolate_process.run(frame)
                if run:
                    for i in range(self.interpolate_factor - 1):
                        result = self.interpolate_process.make_inference(
                            (i + 1) * 1.0 / (self.interpolate_factor + 1)
                        )
                        self.writeBuffer.write(result)
                    
                    self.interpolate_process.cacheFrame()
            
            self.writeBuffer.write(frame)

        except Exception as e:
            logging.exception(f"Something went wrong while processing the frames, {e}")
        finally:
            self.semaphore.release()

    def process(self):
        frameCount = 0
        self.dedupCount = 0
        self.prevFrame = None
        with ThreadPoolExecutor(max_workers=self.nt) as executor:
            while True:
                frame = self.readBuffer.read()
                if frame is None:
                    if (
                        self.readBuffer.isReadingDone()
                        and self.readBuffer.getSizeOfQueue() == 0
                    ):
                        break

                self.semaphore.acquire()
                executor.submit(self.processFrame, frame)
                frameCount += 1

        if self.prevFrame is not None:
            self.writeBuffer.write(self.prevFrame)

        logging.info(f"Processed {frameCount} frames")
        if self.dedupCount > 0:
            logging.info(f"Deduplicated {self.dedupCount} frames")

        self.writeBuffer.close()

    def start(self):
        try:
            (
                self.new_width,
                self.new_height,
                self.upscale_process,
                self.interpolate_process,
                self.denoise_process,
                self.dedup_process,
            ) = intitialize_models(self)

            self.readBuffer = BuildBuffer(
                self.input,
                self.ffmpeg_path,
                self.inpoint,
                self.outpoint,
                self.dedup,
                self.dedup_sens,
                self.dedup_method,
                self.width,
                self.height,
                self.resize,
                self.resize_method,
                self.buffer_limit,
            )

            self.writeBuffer = WriteBuffer(
                self.input,
                self.output,
                self.ffmpeg_path,
                self.encode_method,
                self.custom_encoder,
                self.new_width,
                self.new_height,
                self.fps,
                self.buffer_limit,
                self.sharpen,
                self.sharpen_sens,
                grayscale=False,
                transparent=False,
                audio=self.audio,
            )

        except Exception as e:
            logging.exception(f"Something went wrong, {e}")

        self.semaphore = Semaphore(self.nt * 4)
        with ThreadPoolExecutor(max_workers=3) as executor:
            executor.submit(self.readBuffer.start, verbose=True)
            executor.submit(self.process)
            executor.submit(self.writeBuffer.start, verbose=True)


if __name__ == "__main__":
    log_file_path = os.path.join(mainPath, "log.txt")
    logging.basicConfig(
        filename=log_file_path, filemode="w", format="%(message)s", level=logging.INFO
    )

    command_line_args = sys.argv
    logging.info("============== Command Line Arguments ==============")
    command_line_string = " ".join(command_line_args)

    logging.info(f"{command_line_string}\n")

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--version", action="store_true")
    argparser.add_argument("--input", type=str)
    argparser.add_argument("--output", type=str)
    argparser.add_argument("--interpolate", type=int, choices=[0, 1], default=0)
    argparser.add_argument("--interpolate_factor", type=int, default=2)
    argparser.add_argument(
        "--interpolate_method",
        type=str,
        choices=[
            "rife",
            "rife4.6",
            "rife4.14",
            "rife4.14-lite",
            "rife4.13-lite",
            "gmfss",
            "rife-ncnn",
            "rife4.6-ncnn",
            "rife4.13-lite-ncnn",
            "rife4.14-lite-ncnn",
            "rife4.14-ncnn",
        ],
        default="rife",
    )
    argparser.add_argument("--ensemble", type=int, choices=[0, 1], default=0)
    argparser.add_argument("--upscale", type=int, choices=[0, 1], default=0)
    argparser.add_argument("--upscale_factor", type=int, choices=[2, 3, 4], default=2)
    argparser.add_argument(
        "--upscale_method",
        type=str,
        choices=[
            "shufflecugan",
            "cugan",
            "compact",
            "ultracompact",
            "superultracompact",
            "swinir",
            "span",
            "span-ncnn",
            "cugan-ncnn",
            "omnisr",
            "realesrgan",
            "realesrgan-ncnn",
            "shufflecugan-ncnn",
        ],
        default="shufflecugan",
    )
    argparser.add_argument(
        "--cugan_kind",
        type=str,
        choices=["no-denoise", "conservative", "denoise1x", "denoise2x"],
        default="no-denoise",
    )
    argparser.add_argument("--custom_model", type=str, default="")
    argparser.add_argument("--dedup", type=int, choices=[0, 1], default=0)
    argparser.add_argument(
        "--dedup_method", type=str, default="ffmpeg", choices=["ffmpeg", "ssim"]
    )
    argparser.add_argument("--dedup_sens", type=float, default=35)
    argparser.add_argument("--sample_size", type=int, default=32)
    argparser.add_argument("--nt", type=int, default=1)
    argparser.add_argument("--half", type=int, choices=[0, 1], default=1)
    argparser.add_argument("--inpoint", type=float, default=0)
    argparser.add_argument("--outpoint", type=float, default=0)
    argparser.add_argument("--sharpen", type=int, choices=[0, 1], default=0)
    argparser.add_argument("--sharpen_sens", type=float, default=50)
    argparser.add_argument("--segment", type=int, choices=[0, 1], default=0)
    argparser.add_argument("--scenechange", type=int, choices=[0, 1], default=0)
    argparser.add_argument("--scenechange_sens", type=float, default=50)
    argparser.add_argument("--depth", type=int, choices=[0, 1], default=0)
    argparser.add_argument(
        "--depth_method", type=str, choices=["small", "base", "large"], default="small"
    )
    argparser.add_argument(
        "--encode_method",
        type=str,
        choices=[
            "x264",
            "x264_animation",
            "x265",
            "nvenc_h264",
            "nvenc_h265",
            "qsv_h264",
            "qsv_h265",
            "nvenc_av1",
            "av1",
            "h264_amf",
            "hevc_amf",
            "vp9",
            "qsv_vp9",
            "prores",
        ],
        default="x264",
    )
    argparser.add_argument("--ytdlp", type=str, default="")
    argparser.add_argument("--ytdlp_quality", type=int, choices=[0, 1], default=0)
    argparser.add_argument("--resize", type=int, choices=[0, 1], default=0)
    argparser.add_argument(
        "--resize_factor",
        type=float,
        default=2,
        help="Resize factor for the decoded video, can also be values between 0 & 1 for downscaling though it needs more work, it will always keep the desired aspect ratio",
    )
    argparser.add_argument(
        "--resize_method",
        type=str,
        choices=[
            "fast_bilinear",
            "bilinear",
            "bicubic",
            "experimental",
            "neighbor",
            "area",
            "bicublin",
            "gauss",
            "sinc",
            "lanczos",
            "point",
            "spline",
            "spline16",
            "spline36",
        ],
        default="bicubic",
        help="Choose the desired resizer, I am particularly happy with lanczos for upscaling and area for downscaling",
    )
    argparser.add_argument("--custom_encoder", type=str, default="")
    argparser.add_argument("--buffer_limit", type=int, default=50)
    argparser.add_argument(
        "--audio",
        type=int,
        choices=[0, 1],
        default=1,
        help="Keep the audio track and later merge it back into the video, if dedup is true this will be set to False automatically",
    )
    argparser.add_argument("--denoise", type=int, choices=[0, 1], default=0)
    argparser.add_argument("--denoise_method", type=str, default="scunet", choices=["scunet", "nafnet", "dpir", "kbnet"])

    args = argparser.parse_args()
    args.ffmpeg_path = argumentChecker(args, mainPath, scriptVersion)

    if os.path.isfile(args.input):
        print(f"Processing {args.input}")
        if args.output is None:
            outputFolder = os.path.join(mainPath, "output")
            os.makedirs(os.path.join(outputFolder), exist_ok=True)
            args.output = os.path.join(outputFolder, outputNameGenerator(args))
        elif os.path.isdir(args.output):
            args.output = os.path.join(args.output, outputNameGenerator(args))

        VideoProcessor(args)

    elif os.path.isdir(args.input):
        videoFiles = [
            os.path.join(args.input, file)
            for file in os.listdir(args.input)
            if file.endswith((".mp4", ".mkv", ".mov", ".avi"))
        ]

        for videoFile in videoFiles:
            args.input = os.path.abspath(videoFile)
            logging.info(f"Processing {args.input}")
            print(f"Processing {args.input}")

            if args.output is None:
                outputFolder = os.path.join(mainPath, "output")
                os.makedirs(os.path.join(outputFolder), exist_ok=True)
                args.output = os.path.join(outputFolder, outputNameGenerator(args))

            VideoProcessor(args)
            args.output = None

    else:
        print("No input was specified, exiting")
        logging.info("No input was specified, exiting")
