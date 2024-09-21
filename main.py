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
import sys
import logging
import warnings
from platform import system
from signal import signal, SIGINT, SIG_DFL
from time import time

from alive_progress import alive_bar
from concurrent.futures import ThreadPoolExecutor
from src.argumentsChecker import createParser
from src.getVideoMetadata import getVideoMetadata
from src.initializeModels import initializeModels, Segment, Depth, Stabilize, AutoClip
from src.ffmpegSettings import BuildBuffer, WriteBuffer
from src.generateOutput import outputNameGenerator
from src.coloredPrints import green, blue, red
from queue import Queue
from torch import multiprocessing as mp

warnings.filterwarnings("ignore")

if system() == "Windows":
    mainPath = os.path.join(os.getenv("APPDATA"), "TheAnimeScripter")
else:
    mainPath = os.path.join(
        os.getenv("XDG_CONFIG_HOME", os.path.expanduser("~/.config")),
        "TheAnimeScripter",
    )

if not os.path.exists(mainPath):
    os.makedirs(mainPath)

if getattr(sys, "frozen", False):
    isFrozen = True
    outputPath = os.path.dirname(sys.executable)
else:
    isFrozen = False
    outputPath = os.path.dirname(os.path.abspath(__file__))


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
        self.dedup = args.dedup
        self.dedup_method = args.dedup_method
        self.dedup_sens = args.dedup_sens
        self.half = args.half
        self.inpoint = args.inpoint
        self.outpoint = args.outpoint
        self.sharpen = args.sharpen
        self.sharpen_sens = args.sharpen_sens
        self.segment = args.segment
        self.autoclip = args.autoclip
        self.autoclip_sens = args.autoclip_sens
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
        self.buffer_limit = args.buffer_limit
        self.audio = args.audio
        self.denoise = args.denoise
        self.denoise_method = args.denoise_method
        self.sample_size = args.sample_size
        self.benchmark = args.benchmark
        self.segment_method = args.segment_method
        self.scenechange = args.scenechange
        self.scenechange_sens = args.scenechange_sens
        self.scenechange_method = args.scenechange_method
        self.upscale_skip = args.upscale_skip
        self.bit_depth = args.bit_depth
        self.stabilize = args.stabilize
        self.preview = args.preview
        self.forceStatic = args.static
        self.depth_quality = args.depth_quality
        self.interpolate_skip = args.interpolate_skip

        self.width, self.height, self.fps, self.totalFrames, self.pixFMT, self.audio = (
            getVideoMetadata(self.input, self.inpoint, self.outpoint)
        )

        self.outputFPS = (
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

        if self.stabilize:
            logging.info("Stabilizing video")
            Stabilize(self)

        elif self.autoclip:
            logging.info("Detecting scene changes")
            AutoClip(self, mainPath)

        elif self.depth:
            logging.info("Depth Estimation")
            Depth(self)

        elif self.segment:
            logging.info("Segmenting video")
            Segment(self)

        else:
            self.start()

    def processFrame(self, frame):
        if self.dedup:
            if self.dedup_process(frame):
                self.dedupCount += 1
                return

        # frame = darkenLines(frame)

        if self.scenechange:
            self.isSceneChange = self.scenechange_process(frame)
            if self.isSceneChange:
                self.sceneChangeCounter += 1

        if self.denoise:
            frame = self.denoise_process(frame)

        if self.interpolate:
            if self.isSceneChange:
                self.interpolate_process.cacheFrameReset(frame)
            else:
                self.interpolate_process(frame, self.interpQueue)

        if self.upscale:
            if self.interpolate:
                if self.isSceneChange:
                    frame = self.upscale_process(frame)
                    for _ in range(self.interpolate_factor):
                        self.writeBuffer.write(frame)
                else:
                    while not self.interpQueue.empty():
                        self.writeBuffer.write(
                            self.upscale_process(self.interpQueue.get())
                        )
                    self.writeBuffer.write(self.upscale_process(frame))
            else:
                self.writeBuffer.write(self.upscale_process(frame))

        else:
            if self.interpolate:
                if self.isSceneChange or not self.interpQueue.empty():
                    for _ in range(self.interpolate_factor - 1):
                        frameToWrite = (
                            frame if self.isSceneChange else self.interpQueue.get()
                        )
                        self.writeBuffer.write(frameToWrite)

            self.writeBuffer.write(frame)

        if self.preview:
            self.preview.add(frame.clamp(0, 255).byte().cpu().numpy())

    def process(self):
        frameCount = 0
        self.dedupCount = 0
        self.isSceneChange = False
        self.sceneChangeCounter = 0
        increment = 1 if not self.interpolate else self.interpolate_factor
        if self.interpolate:
            self.interpQueue = Queue(maxsize=self.interpolate_factor)

        try:
            with alive_bar(
                total=self.totalFrames * increment,
                title="Processing Frame: ",
                length=30,
                stats="| {rate}",
                elapsed="Elapsed Time: {elapsed}",
                monitor=" {count}/{total} | [{percent:.0%}] | ",
                # stats_end="{total_time} • {rate:.2f}/s",
                unit="frames",
                spinner=None,
            ) as bar:
                for _ in range(self.totalFrames):
                    frame = self.readBuffer.read()
                    if frame is None:
                        self.writeBuffer.close()
                        break
                    self.processFrame(frame)
                    frameCount += 1
                    bar(increment)
        except Exception as e:
            logging.exception(f"Something went wrong while processing the frames, {e}")

        logging.info(f"Processed {frameCount} frames")
        if self.dedupCount > 0:
            logging.info(f"Deduplicated {self.dedupCount} frames")

        if self.scenechange:
            logging.info(f"Detected {self.sceneChangeCounter} scene changes")

        self.writeBuffer.close()
        if self.preview:
            self.preview.close()

    def start(self):
        try:
            (
                self.new_width,
                self.new_height,
                self.upscale_process,
                self.interpolate_process,
                self.denoise_process,
                self.dedup_process,
                self.scenechange_process,
            ) = initializeModels(self)

            starTime: float = time()
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
                totalFrames=self.totalFrames,
            )

            self.writeBuffer = WriteBuffer(
                self.input,
                self.output,
                self.ffmpeg_path,
                self.encode_method,
                self.custom_encoder,
                self.new_width,
                self.new_height,
                self.outputFPS,
                self.buffer_limit,
                self.sharpen,
                self.sharpen_sens,
                grayscale=False,
                transparent=False,
                audio=self.audio,
                benchmark=self.benchmark,
                bitDepth=self.bit_depth,
                inpoint=self.inpoint,
                outpoint=self.outpoint,
                preview=self.preview,
            )

            if self.preview:
                from src.previewSettings import Preview

                self.preview = Preview()

            self.writeBuffer.start()
            with ThreadPoolExecutor(max_workers=3 if self.preview else 2) as executor:
                executor.submit(self.readBuffer.start)
                executor.submit(self.process)
                if self.preview:
                    executor.submit(self.preview.start)

            elapsedTime: float = time() - starTime
            logging.info(
                f"Total Time: {elapsedTime:.2f} seconds FPS: {self.totalFrames / elapsedTime:.2f}"
            )
            print(
                green(
                    f"Total Time: {elapsedTime:.2f} seconds FPS: {self.totalFrames / elapsedTime:.2f}"
                )
            )

        except Exception as e:
            logging.exception(f"Something went wrong while starting the processes, {e}")


if __name__ == "__main__":
    mp.freeze_support()
    if system() == "Windows":
        mp.set_start_method("spawn", force=True)

    signal(SIGINT, SIG_DFL)
    logging.basicConfig(
        filename=os.path.join(mainPath, "log.txt"),
        filemode="w",
        format="%(message)s",
        level=logging.INFO,
    )
    logging.info("============== Command Line Arguments ==============")
    logging.info(f"{' '.join(sys.argv)}\n")

    args = createParser(isFrozen, mainPath, outputPath)

    if os.path.isfile(args.input) and not args.input.endswith(".txt"):
        print(green(f"Processing {args.input}"))
        if args.output is None:
            outputFolder = os.path.join(outputPath, "output")
            os.makedirs(os.path.join(outputFolder), exist_ok=True)
            args.output = os.path.join(outputFolder, outputNameGenerator(args))
        elif os.path.isdir(args.output):
            args.output = os.path.join(args.output, outputNameGenerator(args))

        VideoProcessor(args)

    elif os.path.isdir(args.input):
        videoFiles = [
            os.path.join(args.input, file)
            for file in os.listdir(args.input)
            if file.endswith((".mp4", ".mkv", ".mov", ".avi", ".webm"))
        ]

        toPrint = f"Processing {len(videoFiles)} files"
        logging.info(toPrint)
        print(blue(toPrint))
        copyArgsOutput = args.output if args.output else None
        if args.output:
            os.makedirs(args.output, exist_ok=True)

        for videoFile in videoFiles:
            args = (isFrozen, mainPath, outputPath)
            args.input = os.path.abspath(videoFile)
            toPrint = f"Processing {args.input}"
            logging.info(toPrint)
            print(green(toPrint))

            if copyArgsOutput is None:
                outputFolder = os.path.join(outputPath, "output")
                os.makedirs(outputFolder, exist_ok=True)
                args.output = os.path.join(outputFolder, outputNameGenerator(args))
            elif os.path.isdir(copyArgsOutput):
                args.output = os.path.join(copyArgsOutput, outputNameGenerator(args))

            print(green(f"Output File: {args.output}"))
            VideoProcessor(args)
            args.output = copyArgsOutput
    else:
        try:
            if args.input.endswith(".txt"):
                with open(args.input, "r") as file:
                    videoFiles = [line.strip().strip('"') for line in file.readlines()]
            else:
                videoFiles = args.input.split(";")
            toPrint = f"Processing {len(videoFiles)} files"
            logging.info(toPrint)
            print(blue(toPrint))

            copyArgsOutput = args.output if args.output else None
            if args.output:
                os.makedirs(args.output, exist_ok=True)

            for videoFile in videoFiles:
                args.input = os.path.abspath(videoFile)
                toPrint = f"Processing {args.input}"
                logging.info(toPrint)
                print(green(toPrint))

                if copyArgsOutput is None:
                    outputFolder = os.path.join(outputPath, "output")
                    os.makedirs(outputFolder, exist_ok=True)
                    args.output = os.path.join(outputFolder, outputNameGenerator(args))
                elif os.path.isdir(copyArgsOutput):
                    args.output = os.path.join(
                        copyArgsOutput, outputNameGenerator(args)
                    )

                print(green(f"Output File: {args.output}"))
                VideoProcessor(args)
                args.output = copyArgsOutput
        except Exception:
            toPrint = f"File or directory {args.input} does not exist, exiting"
            print(red(toPrint))
            logging.info(toPrint)
