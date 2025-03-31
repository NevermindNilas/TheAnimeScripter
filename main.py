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
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import src.constants as cs

from src.utils.coloredPrints import green
from src.utils.argumentsChecker import createParser
from src.utils.getVideoMetadata import getVideoMetadata
from src.utils.progressBarLogic import ProgressBarLogic
from src.utils.inputOutputHandler import handleInputOutputs
from src.utils.initializeModels import (
    initializeModels,
    Segment,
    Depth,
    AutoClip,
    ObjectDetection,
)
from src.utils.logAndPrint import logAndPrint

warnings.filterwarnings("ignore")


class VideoProcessor:
    def __init__(
        self,
        args,
        results=None,
    ):
        self.input = results["videoPath"]
        self.output = results["outputPath"]
        self.encode_method = results["encodeMethod"]
        self.custom_encoder = results["customEncoder"]

        self._initProcessingParams(args)
        self._initVideoMetadata(args)
        self._configureProcessingOptions(args)
        self._selectProcessingMethod()

    def _initProcessingParams(self, args):
        self.interpolate: bool = args.interpolate
        self.interpolate_factor: int = args.interpolate_factor
        self.interpolate_method: str = args.interpolate_method
        self.upscale: bool = args.upscale
        self.upscale_factor: int = args.upscale_factor
        self.upscale_method: str = args.upscale_method
        self.dedup: bool = args.dedup
        self.dedup_method: str = args.dedup_method
        self.dedup_sens: float = args.dedup_sens
        self.half: bool = args.half
        self.inpoint: float = args.inpoint
        self.outpoint: float = args.outpoint
        self.sharpen: bool = args.sharpen
        self.sharpen_sens: float = args.sharpen_sens
        self.autoclip: bool = args.autoclip
        self.autoclip_sens: float = args.autoclip_sens
        self.depth: bool = args.depth
        self.depth_method: str = args.depth_method
        self.ensemble: bool = args.ensemble
        self.resize: bool = args.resize
        self.resize_factor: float = args.resize_factor
        self.resize_method: str = args.resize_method
        self.custom_model: str = args.custom_model
        self.restore: bool = args.restore
        self.restore_method: str = args.restore_method
        self.benchmark: bool = args.benchmark
        self.segment: bool = args.segment
        self.segment_method: str = args.segment_method
        self.scenechange: bool = args.scenechange
        self.scenechange_sens: float = args.scenechange_sens
        self.scenechange_method: str = args.scenechange_method
        self.bit_depth: str = args.bit_depth
        self.preview: bool = args.preview
        self.forceStatic: bool = args.static
        self.depth_quality: str = args.depth_quality
        self.decode_threads: int = args.decode_threads
        self.realtime: bool = args.realtime
        self.dynamic_scale: bool = args.dynamic_scale
        self.static_step: bool = args.static_step
        self.slowmo: bool = args.slowmo
        self.interpolate_first: bool = args.interpolate_first
        self.obj_detect: bool = args.obj_detect

    def _initVideoMetadata(self, args) -> None:
        videoMetadata = getVideoMetadata(
            self.input,
            args.inpoint,
            args.outpoint,
        )

        self.width: int = videoMetadata["Width"]
        self.height: int = videoMetadata["Height"]
        self.fps: float = videoMetadata["FPS"]
        self.totalFrames: int = videoMetadata["TotalFramesToBeProcessed"]

    def _configureProcessingOptions(self, args) -> None:
        logging.info("\n============== Processing Outputs ==============")

        if self.slowmo:
            self.outputFPS = self.fps
        else:
            self.outputFPS = (
                self.fps * args.interpolate_factor if args.interpolate else self.fps
            )

        if self.resize:
            aspectRatio = self.width / self.height
            self.width = round(self.width * self.resize_factor / 2) * 2
            self.height = round(self.width / aspectRatio / 2) * 2
            logging.info(
                f"Resizing to {self.width}x{self.height} using {self.resize_factor} factor and {self.resize_method}"
            )

    def _selectProcessingMethod(self) -> None:
        if self.autoclip:
            logging.info("Detecting scene changes")
            AutoClip(self)

        elif self.depth:
            logging.info("Depth Estimation")
            Depth(self)

        elif self.segment:
            logging.info("Segmenting video")
            Segment(self)

        elif self.obj_detect:
            logging.info("Object Detection")
            ObjectDetection(self)

        else:
            self.start()

    def processFrame(self, frame: any) -> None:
        if self.dedup and self.dedup_process(frame):
            self.dedupCount += 1
            return

        if self.scenechange:
            self.isSceneChange = self.scenechange_process(frame)
            if self.isSceneChange:
                self.sceneChangeCounter += 1

        if self.restore:
            frame = self.restore_process(frame)

        if self.interpolate:
            if isinstance(self.interpolate_factor, float):
                intFactor = int(self.interpolate_factor)
                fracPart = self.interpolate_factor - intFactor
                self.frameCounter += fracPart
                framesToInsert = intFactor - 1

                if self.frameCounter >= 1.0:
                    framesToInsert += 1
                    self.frameCounter -= 1.0

                self.framesToInsert = framesToInsert
            else:
                self.framesToInsert = int(self.interpolate_factor) - 1

        if self.interpolate_first:
            self.ifInterpolateFirst(frame)
        else:
            self.ifInterpolateLast(frame)

        if self.preview:
            self.preview.add(
                frame.squeeze(0).permute(1, 2, 0).mul(255).byte().cpu().numpy()
            )

    def ifInterpolateFirst(self, frame: any) -> None:
        if self.interpolate:
            if self.isSceneChange:
                self.interpolate_process.cacheFrameReset(frame)
            else:
                self.interpolate_process(frame, self.interpQueue, self.framesToInsert)

        if self.upscale:
            if self.interpolate:
                if self.isSceneChange:
                    frame = self.upscale_process(frame)
                    for _ in range(self.framesToInsert):
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
                    for _ in range(self.framesToInsert):
                        frameToWrite = (
                            frame if self.isSceneChange else self.interpQueue.get()
                        )
                        self.writeBuffer.write(frameToWrite)

            self.writeBuffer.write(frame)

    def ifInterpolateLast(self, frame: any) -> None:
        if self.upscale:
            frame = self.upscale_process(frame)

        if self.interpolate:
            if self.isSceneChange:
                for _ in range(self.framesToInsert):
                    self.writeBuffer.write(frame)
                self.interpolate_process.cacheFrameReset(frame)
            else:
                self.interpolate_process(frame, self.writeBuffer)

        self.writeBuffer.write(frame)

    def process(self):
        frameCount = 0
        self.dedupCount = 0
        self.isSceneChange = False
        self.sceneChangeCounter = 0
        self.frameCounter = 0

        self.framesToInsert = self.interpolate_factor - 1

        if self.interpolate:
            if isinstance(self.interpolate_factor, float):
                increment = self.interpolate_factor
            else:
                increment = int(self.interpolate_factor)
        else:
            increment = 1
        if self.interpolate and self.interpolate_first:
            self.interpQueue = Queue(maxsize=round(self.interpolate_factor))

        try:
            with ProgressBarLogic(self.totalFrames * increment) as bar:
                for _ in range(self.totalFrames):
                    frame = self.readBuffer.read()
                    self.processFrame(frame)
                    frameCount += 1
                    bar(increment)
                    if self.readBuffer.isReadFinished():
                        if self.readBuffer.isQueueEmpty():
                            bar.updateTotal(newTotal=frameCount * increment)
                            break

            if self.preview:
                self.preview.close()
            self.writeBuffer.close()

        except Exception as e:
            logging.exception(f"Something went wrong while processing the frames, {e}")

        logging.info(f"Processed {frameCount} frames")
        if self.dedupCount > 0:
            logging.info(f"Deduplicated {self.dedupCount} frames")

        if self.scenechange:
            logging.info(f"Detected {self.sceneChangeCounter} scene changes")

    def start(self):
        from src.utils.ffmpegSettings import BuildBuffer, WriteBuffer

        try:
            (
                self.new_width,
                self.new_height,
                self.upscale_process,
                self.interpolate_process,
                self.restore_process,
                self.dedup_process,
                self.scenechange_process,
            ) = initializeModels(self)

            starTime: float = time()

            self.readBuffer = BuildBuffer(
                videoInput=self.input,
                inpoint=self.inpoint,
                outpoint=self.outpoint,
                totalFrames=self.totalFrames,
                fps=self.fps,
                half=self.half,
                decodeThreads=self.decode_threads,
                resize=self.resize,
                resizeMethod=self.resize_method,
                width=self.width,
                height=self.height,
            )

            self.writeBuffer = WriteBuffer(
                input=self.input,
                output=self.output,
                encode_method=self.encode_method,
                custom_encoder=self.custom_encoder,
                width=self.new_width,
                height=self.new_height,
                fps=self.outputFPS,
                sharpen=self.sharpen,
                sharpen_sens=self.sharpen_sens,
                grayscale=False,
                transparent=False,
                benchmark=self.benchmark,
                bitDepth=self.bit_depth,
                inpoint=self.inpoint,
                outpoint=self.outpoint,
                realtime=self.realtime,
                slowmo=self.slowmo,
            )

            if self.preview:
                from src.utils.previewSettings import Preview

                self.preview = Preview()

            with ThreadPoolExecutor(max_workers=4 if self.preview else 3) as executor:
                executor.submit(self.readBuffer)
                executor.submit(self.writeBuffer)
                executor.submit(self.process)
                if self.preview:
                    executor.submit(self.preview.start)

            elapsedTime: float = time() - starTime
            totalFPS: float = (
                self.totalFrames
                / elapsedTime
                * (1 if not self.interpolate else self.interpolate_factor)
            )
            logging.info(
                f"Total Execution Time: {elapsedTime:.2f} seconds - FPS: {totalFPS:.2f}"
            )
            print(
                green(
                    f"Total Execution Time: {elapsedTime:.2f} seconds - FPS: {totalFPS:.2f}"
                )
            )

        except Exception as e:
            logging.exception(f"Something went wrong while starting the processes, {e}")


def main():
    try:
        cs.SYSTEM = system()
        cs.MAINPATH = (
            os.path.join(os.getenv("APPDATA"), "TheAnimeScripter")
            if cs.SYSTEM == "Windows"
            else os.path.join(
                os.getenv("XDG_CONFIG_HOME", os.path.expanduser("~/.config")),
                "TheAnimeScripter",
            )
        )
        os.makedirs(cs.MAINPATH, exist_ok=True)

        isFrozen = hasattr(sys, "_MEIPASS")
        baseOutputPath = (
            os.path.dirname(sys.executable)
            if isFrozen
            else os.path.dirname(os.path.abspath(__file__))
        )

        # Setup logging
        signal(SIGINT, SIG_DFL)
        logging.basicConfig(
            filename=os.path.join(cs.MAINPATH, "TAS-Log.log"),
            filemode="w",
            format="%(message)s",
            level=logging.INFO,
        )
        logging.info("============== Command Line Arguments ==============")
        logging.info(f"{' '.join(sys.argv)}\n")

        args = createParser(isFrozen, baseOutputPath, cs.SYSTEM)
        outputPath = os.path.join(baseOutputPath, "output")
        results = handleInputOutputs(args, isFrozen, outputPath)

        totalVideos = len(results)
        if totalVideos == 0:
            logAndPrint("No videos found to process", colorFunc="red")
            sys.exit(1)

        if totalVideos > 1:
            logAndPrint(f"Total Videos found: {totalVideos}", colorFunc="green")
            folderTimer = time()

        for idx, i in enumerate(results, 1):
            try:
                if totalVideos > 1:
                    logAndPrint(
                        f"Processing Video {idx}/{totalVideos}: {results[i]['videoPath']}",
                        colorFunc="green",
                    )
                else:
                    logAndPrint(
                        f"Processing Video: {results[i]['videoPath']}",
                        colorFunc="green",
                    )

                logAndPrint(
                    f"Output Path: {results[i]['outputPath']}", colorFunc="green"
                )
                VideoProcessor(
                    args,
                    results=results[i],
                )

            except Exception as e:
                logAndPrint(
                    f"Error processing video {results[i]['videoPath']}: {str(e)}",
                    colorFunc="red",
                )
                logging.exception(f"Error processing video {results[i]['videoPath']}")

        if totalVideos > 1:
            totalTime = time() - folderTimer
            logAndPrint(
                f"Total Execution Time: {totalTime:.2f} seconds | "
                f"Average per video: {totalTime / totalVideos:.2f} seconds",
                colorFunc="green",
            )

    except KeyboardInterrupt:
        logAndPrint("Process interrupted by user", colorFunc="yellow")
        sys.exit(0)
    except Exception as e:
        logAndPrint(f"An unexpected error occurred: {str(e)}", colorFunc="red")
        logging.exception("Fatal error in main execution")
        sys.exit(1)


if __name__ == "__main__":
    main()
