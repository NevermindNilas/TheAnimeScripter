"""
The Anime Scripter - AI Video Enhancement Toolkit

A high-performance AI video enhancement toolkit specialized for anime and general video content.
Provides professional-grade AI upscaling, interpolation, and restoration capabilities.

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
import src.constants as cs


from platform import system
from signal import signal, SIGINT, SIG_DFL
from time import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from fractions import Fraction

from src.utils.coloredPrints import green
from src.utils.argumentsChecker import createParser
from src.utils.getVideoMetadata import getVideoMetadata
from src.utils.inputOutputHandler import processInputOutputPaths
from src.initializeModels import (
    initializeModels,
    segment,
    depth,
    autoClip,
    objectDetection,
)
from src.utils.logAndPrint import logAndPrint

warnings.filterwarnings("ignore")


class VideoProcessor:
    """
    Main video processing class that handles AI-powered video enhancement operations.
    
    Supports upscaling, interpolation, restoration, deduplication, and various other
    video processing operations using different AI models and hardware backends.
    """
    
    def __init__(self, args, results=None):
        """
        Initialize the VideoProcessor with command line arguments and processing results.
        
        Args:
            args: Parsed command line arguments containing processing options
            results: Dictionary containing video path, output path, and encoding settings
        """
        self.input = results["videoPath"]
        self.output = results["outputPath"]
        self.encodeMethod = results["encodeMethod"]
        self.customEncoder = results["customEncoder"]

        self._initProcessingParams(args)
        self._initVideoMetadata(args)
        self._configureProcessingOptions(args)
        self._selectProcessingMethod()

    def _initProcessingParams(self, args):
        """
        Initialize processing parameters from command line arguments.
        
        Args:
            args: Parsed command line arguments
        """
        # Core processing flags
        self.interpolate: bool = args.interpolate
        self.upscale: bool = args.upscale
        self.restore: bool = args.restore
        self.dedup: bool = args.dedup
        self.sharpen: bool = args.sharpen
        self.autoclip: bool = args.autoclip
        self.depth: bool = args.depth
        self.segment: bool = args.segment
        self.scenechange: bool = args.scenechange
        self.objDetect: bool = args.obj_detect
        
        # Processing parameters
        self.interpolateFactor: int = args.interpolate_factor
        self.interpolateMethod: str = args.interpolate_method
        self.upscaleFactor: int = args.upscale_factor
        self.upscaleMethod: str = args.upscale_method
        self.dedupMethod: str = args.dedup_method
        self.dedupSens: float = args.dedup_sens
        self.restoreMethod: str = args.restore_method
        self.depthMethod: str = args.depth_method
        self.segmentMethod: str = args.segment_method
        self.scenechangeMethod: str = args.scenechange_method
        self.objDetectMethod: str = args.obj_detect_method
        
        # Quality and performance settings
        self.half: bool = args.half
        self.ensemble: bool = args.ensemble
        self.forceStatic: bool = args.static
        self.dynamicScale: bool = args.dynamic_scale
        self.staticStep: bool = args.static_step
        self.compileMode: str = args.compile_mode
        
        # Video processing settings
        self.inpoint: float = args.inpoint
        self.outpoint: float = args.outpoint
        self.resize: bool = args.resize
        self.resizeFactor: float = args.resize_factor
        self.bitDepth: str = args.bit_depth
        self.slowmo: bool = args.slowmo
        self.interpolateFirst: bool = args.interpolate_first
        
        # Enhancement settings
        self.sharpenSens: float = args.sharpen_sens
        self.autoclipSens: float = args.autoclip_sens
        self.scenechangeSens: float = args.scenechange_sens
        self.depthQuality: str = args.depth_quality
        
        # Utility settings
        self.customModel: str = args.custom_model
        self.benchmark: bool = args.benchmark
        self.preview: bool = args.preview

    def _initVideoMetadata(self, args) -> None:
        """
        Initialize video metadata by analyzing the input video file.
        
        Args:
            args: Command line arguments containing inpoint and outpoint
        """
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
        """
        Configure processing options based on the selected operations.
        
        Args:
            args: Command line arguments
        """
        logging.info("\n============== Processing Outputs ==============")

        # Calculate output FPS based on interpolation settings
        if self.slowmo:
            self.outputFPS = self.fps
        else:
            self.outputFPS = (
                self.fps * self.interpolateFactor if self.interpolate else self.fps
            )

        # Handle video resizing if requested
        if self.resize:
            aspectRatio = self.width / self.height
            self.width = round(self.width * self.resizeFactor / 2) * 2
            self.height = round(self.width / aspectRatio / 2) * 2
            logging.info(
                f"Resizing to {self.width}x{self.height} using {self.resizeFactor} factor."
            )

    def _selectProcessingMethod(self) -> None:
        """
        Select and execute the appropriate processing method based on user options.
        
        Prioritizes specialized operations (autoclip, depth, segment, object detection)
        over standard video processing.
        """
        if self.autoclip:
            logging.info("Detecting scene changes")
            autoClip(self)
        elif self.depth:
            logging.info("Depth Estimation")
            depth(self)
        elif self.segment:
            logging.info("Segmenting video")
            segment(self)
        elif self.objDetect:
            logging.info("Object Detection")
            objectDetection(self)
        else:
            self.start()

    def processFrame(self, frame: any) -> None:
        """
        Process a single video frame through the configured enhancement pipeline.
        
        Args:
            frame: Input video frame tensor
        """
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
            if isinstance(self.interpolateFactor, float):
                currentIDX = self.frameCounter
                nextIDX = currentIDX + 1

                outputStart = (currentIDX * self.factorNum) // self.factorDen
                outputEnd = (nextIDX * self.factorNum) // self.factorDen

                self.framesToInsert = outputEnd - outputStart - 1

                self.timesteps = []
                for i in range(1, self.framesToInsert + 1):
                    outputIDX = outputStart + i
                    t = (outputIDX * self.factorDen % self.factorNum) / self.factorNum
                    self.timesteps.append(t)

                self.frameCounter += 1
            else:
                self.framesToInsert = int(self.interpolateFactor) - 1
                self.timesteps = None

        if self.interpolateFirst:
            self.ifInterpolateFirst(frame)
        else:
            self.ifInterpolateLast(frame)

        if self.preview:
            self.preview.add(
                frame.squeeze(0).permute(1, 2, 0).mul(255).byte().cpu().numpy()
            )

    def ifInterpolateFirst(self, frame: any) -> None:
        """
        Process frame with interpolation-first pipeline order.
        
        Args:
            frame: Input video frame tensor
        """
        if self.interpolate:
            if self.isSceneChange:
                self.interpolate_process.cacheFrameReset(frame)
            else:
                self.interpolate_process(
                    frame, self.interpQueue, self.framesToInsert, self.timesteps
                )

        if self.upscale:
            if self.interpolate:
                if self.isSceneChange:
                    frame = self.upscale_process(frame)
                    for _ in range(self.framesToInsert + 1):
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
        """
        Process frame with interpolation-last pipeline order.
        
        Args:
            frame: Input video frame tensor
        """
        if self.upscale:
            frame = self.upscale_process(frame)

        if self.interpolate:
            if self.isSceneChange:
                for _ in range(self.framesToInsert + 1):
                    self.writeBuffer.write(frame)
                self.interpolate_process.cacheFrameReset(frame)
            else:
                self.interpolate_process(
                    frame, self.writeBuffer, self.framesToInsert, self.timesteps
                )
        self.writeBuffer.write(frame)

    def process(self):
        """
        Main processing loop that handles frame-by-frame video processing.
        
        Processes all frames through the configured enhancement pipeline and
        tracks processing statistics.
        """
        frameCount = 0
        self.dedupCount = 0
        self.isSceneChange = False
        self.sceneChangeCounter = 0
        self.frameCounter = 0

        # Configure interpolation factors for fractional interpolation
        if self.interpolate and isinstance(self.interpolateFactor, float):
            factor = Fraction(self.interpolateFactor).limit_denominator(100)
            self.factorNum = factor.numerator
            self.factorDen = factor.denominator

            increment = self.factorNum / self.factorDen
            if increment.is_integer():
                increment = int(increment)
        else:
            self.factorNum = self.interpolateFactor if self.interpolate else 1
            self.factorDen = 1
            increment = int(self.interpolateFactor) if self.interpolate else 1

        self.timesteps = None
        self.framesToInsert = self.interpolateFactor - 1 if self.interpolate else 0

        # Initialize interpolation queue if needed
        if self.interpolate and self.interpolateFirst:
            self.interpQueue = Queue(maxsize=round(self.interpolateFactor))

        try:
            with self.ProgressBarLogic(self.totalFrames * increment) as bar:
                for _ in range(self.totalFrames):
                    frame = self.readBuffer.read()
                    if frame is None:
                        break
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

        # Log processing statistics
        logging.info(f"Processed {frameCount} frames")
        if self.dedupCount > 0:
            logging.info(f"Deduplicated {self.dedupCount} frames")
        if self.scenechange:
            logging.info(f"Detected {self.sceneChangeCounter} scene changes")

    def start(self):
        """
        Initialize and start the video processing pipeline.
        
        Sets up input/output buffers, initializes AI models, and coordinates
        the multi-threaded processing workflow.
        """
        from src.utils.ffmpegSettings import BuildBuffer, WriteBuffer
        from src.utils.progressBarLogic import ProgressBarLogic

        self.ProgressBarLogic = ProgressBarLogic

        try:
            # Initialize AI models and get processing functions
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

            # Setup input buffer for reading video frames
            self.readBuffer = BuildBuffer(
                videoInput=self.input,
                inpoint=self.inpoint,
                outpoint=self.outpoint,
                half=self.half,
                resize=self.resize,
                width=self.width,
                height=self.height,
                bitDepth=self.bitDepth,
            )

            # Setup output buffer for writing processed frames
            self.writeBuffer = WriteBuffer(
                input=self.input,
                output=self.output,
                encode_method=self.encodeMethod,
                custom_encoder=self.customEncoder,
                width=self.new_width,
                height=self.new_height,
                fps=self.outputFPS,
                sharpen=self.sharpen,
                sharpen_sens=self.sharpenSens,
                grayscale=False,
                transparent=False,
                benchmark=self.benchmark,
                bitDepth=self.bitDepth,
                inpoint=self.inpoint,
                outpoint=self.outpoint,
                slowmo=self.slowmo,
            )

            # Initialize preview if enabled
            if self.preview:
                from src.utils.previewSettings import Preview
                self.preview = Preview()

            # Execute processing pipeline with thread pool
            with ThreadPoolExecutor(max_workers=4 if self.preview else 3) as executor:
                executor.submit(self.readBuffer)
                executor.submit(self.writeBuffer)
                executor.submit(self.process)
                if self.preview:
                    executor.submit(self.preview.start)

            # Calculate and log performance metrics
            elapsedTime: float = time() - starTime
            totalFPS: float = (
                self.totalFrames
                / elapsedTime
                * (1 if not self.interpolate else self.interpolateFactor)
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
    """
    Main entry point for The Anime Scripter application.
    
    Handles initialization, argument parsing, and coordinates video processing
    for single or multiple input files.
    """
    try:
        # Initialize system constants
        cs.SYSTEM = system()
        cs.MAINPATH = (
            os.path.join(os.getenv("APPDATA"), "TheAnimeScripter")
            if cs.SYSTEM == "Windows"
            else os.path.join(
                os.getenv("XDG_CONFIG_HOME", os.path.expanduser("~/.config")),
                "TheAnimeScripter",
            )
        )
        cs.WHEREAMIRUNFROM = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(cs.MAINPATH, exist_ok=True)

        # Determine if running from frozen executable
        isFrozen = hasattr(sys, "_MEIPASS")
        baseOutputPath = (
            os.path.dirname(sys.executable)
            if isFrozen
            else os.path.dirname(os.path.abspath(__file__))
        )

        # Configure logging system
        signal(SIGINT, SIG_DFL)
        logging.basicConfig(
            filename=os.path.join(cs.MAINPATH, "TAS-Log.log"),
            filemode="w",
            format="%(message)s",
            level=logging.INFO,
        )
        logging.info("============== Command Line Arguments ==============")
        logging.info(f"{' '.join(sys.argv)}\n")

        # Parse command line arguments and process input/output paths
        args = createParser(baseOutputPath)
        outputPath = os.path.join(baseOutputPath, "output")
        results = processInputOutputPaths(args, outputPath)

        # Validate that videos were found
        totalVideos = len(results)
        if totalVideos == 0:
            logAndPrint("No videos found to process", colorFunc="red")
            sys.exit(1)

        # Initialize batch processing timer if multiple videos
        if totalVideos > 1:
            logAndPrint(f"Total Videos found: {totalVideos}", colorFunc="green")
            folderTimer = time()

        # Process each video
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
                VideoProcessor(args, results=results[i])

            except Exception as e:
                logAndPrint(
                    f"Error processing video {results[i]['videoPath']}: {str(e)}",
                    colorFunc="red",
                )
                logging.exception(f"Error processing video {results[i]['videoPath']}")

        # Log batch processing statistics
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
