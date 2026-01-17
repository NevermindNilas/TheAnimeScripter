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
from src.utils.logAndPrint import (
    logAndPrint,
    logInfo,
    logSuccess,
    logWarning,
    logError,
    printSectionHeader,
    printSubsectionHeader,
)

from platform import system
from signal import signal, SIGINT, SIG_DFL
from time import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from fractions import Fraction
import src.constants as cs

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
        self.objDetectMethod: str = args.obj_detect_method
        self.objDetectDisableAnnotations: bool = args.obj_detect_disable_annotations

        # Quality and performance settings
        self.half: bool = args.half
        self.ensemble: bool = args.ensemble
        self.forceStatic: bool = args.static
        self.dynamicScale: bool = args.dynamic_scale
        self.staticStep: bool = args.static_step
        self.compileMode: str = args.compile_mode
        self.decodeMethod: str = args.decode_method

        # Video processing settings
        self.inpoint: float = args.inpoint
        self.outpoint: float = args.outpoint
        self.resize: bool = args.resize
        self.resizeFactor: float = args.resize_factor
        self.bitDepth: str = args.bit_depth
        self.slowmo: bool = args.slowmo
        self.interpolateFirst: bool = args.interpolate_first
        self.outputScaleWidth: int = args.output_scale_width
        self.outputScaleHeight: int = args.output_scale_height

        # Enhancement settings
        self.sharpenSens: float = args.sharpen_sens
        self.autoclipSens: float = args.autoclip_sens
        self.depthQuality: str = args.depth_quality

        # Utility settings
        self.customModel: str = args.custom_model
        self.benchmark: bool = args.benchmark
        self.preview: bool = args.preview
        self.profile: bool = args.profile

    def _initVideoMetadata(self, args) -> None:
        """
        Initialize video metadata by analyzing the input video file.

        Args:
            args: Command line arguments containing inpoint and outpoint
        """
        # Lazy import to speed up startup for non-processing paths
        from src.utils.getVideoMetadata import getVideoMetadata

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

        if self.slowmo:
            self.outputFPS = self.fps
        else:
            self.outputFPS = (
                self.fps * self.interpolateFactor if self.interpolate else self.fps
            )

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
            from src.initializeModels import autoClip

            autoClip(self)
        elif self.depth:
            logging.info("Depth Estimation")
            from src.initializeModels import depth

            depth(self)
        elif self.segment:
            logging.info("Segmenting video")
            from src.initializeModels import segment

            segment(self)
        elif self.objDetect:
            logging.info("Object Detection")
            from src.initializeModels import objectDetection

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

    def ifInterpolateFirst(self, frame: any) -> None:
        """
        Process frame with interpolation-first pipeline order.

        Args:
            frame: Input video frame tensor
        """
        if self.interpolate:
            if self.interpolateMethod.startswith("distildrba"):
                self.interpolate_process(
                    frame,
                    self.nextFrame,
                    self.interpQueue,
                    self.framesToInsert,
                    self.timesteps,
                )
            else:
                self.interpolate_process(
                    frame, self.interpQueue, self.framesToInsert, self.timesteps
                )

        if self.upscale:
            if self.interpolate:
                while not self.interpQueue.empty():
                    self.writeBuffer.write(
                        self.upscale_process(self.interpQueue.get(), self.nextFrame)
                    )

                self.writeBuffer.write(self.upscale_process(frame, self.nextFrame))

            else:
                self.writeBuffer.write(self.upscale_process(frame, self.nextFrame))

        else:
            if self.interpolate:
                while not self.interpQueue.empty():
                    self.writeBuffer.write(self.interpQueue.get())
            self.writeBuffer.write(frame)

    def ifInterpolateLast(self, frame: any) -> None:
        """
        Process frame with interpolation-last pipeline order.

        Args:
            frame: Input video frame tensor
        """
        if self.upscale:
            frame = self.upscale_process(frame, self.nextFrame)

        if self.interpolate:
            if self.interpolateMethod.startswith("distildrba"):
                self.interpolate_process(
                    frame,
                    self.nextFrame,
                    self.writeBuffer,
                    self.framesToInsert,
                    self.timesteps,
                )
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
        self.frameCounter = 0
        self.nextFrame = None

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

        if self.interpolate and self.interpolateFirst:
            self.interpQueue = Queue(maxsize=round(self.interpolateFactor))

        try:
            with self.ProgressBarLogic(
                self.totalFrames * increment, title=self.input
            ) as bar:
                for _ in range(self.totalFrames):
                    frame = self.readBuffer.read()
                    if frame is None:
                        break
                    if self.upscaleMethod == "animesr" or (
                        self.interpolate
                        and self.interpolateMethod.startswith("distildrba")
                    ):
                        self.nextFrame = self.readBuffer.peek()
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

    def start(self):
        """
        Initialize and start the video processing pipeline.

        Sets up input/output buffers, initializes AI models, and coordinates
        the multi-threaded processing workflow.
        """
        from src.utils.ffmpegSettings import BuildBuffer, WriteBuffer
        from src.utils.progressBarLogic import ProgressBarLogic
        from src.utils.aeComms import progressState

        self.ProgressBarLogic = ProgressBarLogic

        try:
            # Initialize AI models and get processing functions
            from src.initializeModels import initializeModels

            (
                self.new_width,
                self.new_height,
                self.upscale_process,
                self.interpolate_process,
                self.restore_process,
                self.dedup_process,
            ) = initializeModels(self)

            starTime: float = time()

            self.readBuffer = BuildBuffer(
                videoInput=self.input,
                inpoint=self.inpoint,
                outpoint=self.outpoint,
                half=self.half,
                resize=self.resize,
                width=self.width,
                height=self.height,
                bitDepth=self.bitDepth,
                decode_method=self.decodeMethod,
            )

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
                output_scale_width=self.outputScaleWidth,
                output_scale_height=self.outputScaleHeight,
                enablePreview=self.preview,
            )

            if self.preview:
                from src.utils.previewSettings import Preview

                self.preview = Preview(previewPath=self.writeBuffer.previewPath)
                self.preview.start()

            if self.profile:
                self._runWithProfiler()
            else:
                with ThreadPoolExecutor(max_workers=3) as executor:
                    executor.submit(self.readBuffer)
                    executor.submit(self.writeBuffer)
                    executor.submit(self.process)

            elapsedTime: float = time() - starTime
            totalFPS: float = (
                self.totalFrames
                / elapsedTime
                * (1 if not self.interpolate else self.interpolateFactor)
            )
            logAndPrint(
                f"Total Execution Time: {elapsedTime:.2f} seconds - FPS: {totalFPS:.2f}",
                colorFunc="green",
            )

            if cs.ADOBE:
                progressState.setCompleted(outputPath=self.output)

        except Exception as e:
            logging.exception(f"Something went wrong while starting the processes, {e}")
            if cs.ADOBE:
                progressState.setFailed(error=str(e))

    def _runWithProfiler(self):
        """
        Run the processing pipeline with torch.profiler enabled.
        Uses a simplified approach compatible with multi-threaded execution on Windows.
        """
        import torch
        from torch.profiler import profile, ProfilerActivity

        profilePath = os.path.join(cs.MAINPATH, "profiler_trace")
        os.makedirs(profilePath, exist_ok=True)

        logAndPrint(
            f"Profiling enabled. Trace will be saved to: {profilePath}",
            colorFunc="cyan",
        )

        activities = [ProfilerActivity.CPU]
        try:
            if torch.cuda.is_available():
                activities.append(ProfilerActivity.CUDA)
        except Exception:
            pass

        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as prof:
            with ThreadPoolExecutor(max_workers=3) as executor:
                executor.submit(self.readBuffer)
                executor.submit(self.writeBuffer)
                executor.submit(self.process)

        traceFile = os.path.join(profilePath, "trace.json")
        prof.export_chrome_trace(traceFile)

        logAndPrint(
            "\n=== Profiler Summary (Top 20 by CUDA time) ===", colorFunc="cyan"
        )
        try:
            sortKey = (
                "cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"
            )
            summary = prof.key_averages().table(sort_by=sortKey, row_limit=20)
            print(summary)
            logging.info(f"Profiler Summary:\n{summary}")
        except Exception as e:
            logging.warning(f"Could not print profiler summary: {e}")

        logAndPrint(
            f"\nTrace saved to: {traceFile}\n",
            colorFunc="green",
        )


def main():
    """
    Main entry point for The Anime Scripter application.

    Handles initialization, argument parsing, and coordinates video processing
    for single or multiple input files.
    """
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except (AttributeError, Exception):
            pass

    try:
        if any(flag in sys.argv for flag in ("-h", "--help", "-v", "--version")):
            from src.utils.argumentsChecker import createParser

            try:
                createParser(outputPath=os.getcwd())
            except SystemExit:
                return

        cs.SYSTEM = system()
        if cs.SYSTEM == "Windows":
            appdata = os.getenv("APPDATA") or os.getenv("LOCALAPPDATA")
            if not appdata:
                appdata = os.path.join(os.path.expanduser("~"), "AppData", "Roaming")
            cs.MAINPATH = os.path.join(appdata, "TheAnimeScripter")
        else:
            xdg_config = os.getenv("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
            cs.MAINPATH = os.path.join(xdg_config, "TheAnimeScripter")
        cs.WHEREAMIRUNFROM = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(cs.MAINPATH, exist_ok=True)

        baseOutputPath = os.path.dirname(os.path.abspath(__file__))

        signal(SIGINT, SIG_DFL)

        logging.basicConfig(
            filename=os.path.join(cs.MAINPATH, "TAS-Log.log"),
            filemode="w",
            format="[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.DEBUG,
        )

        logging.info("\n" + "=" * 80)
        logging.info("Command Line Arguments".center(80))
        logging.info("=" * 80)
        logging.info(f"{' '.join(sys.argv)}\n")

        from src.utils.argumentsChecker import createParser

        args = createParser(baseOutputPath)
        outputPath = os.path.join(baseOutputPath, "output")
        from src.utils.inputOutputHandler import processInputOutputPaths

        results = processInputOutputPaths(args, outputPath)

        totalVideos = len(results)
        if totalVideos == 0:
            logError("No videos found to process")
            sys.exit(1)

        if totalVideos > 1:
            printSectionHeader("Batch Processing")
            logSuccess(f"Found {totalVideos} videos to process")
            folderTimer = time()
        else:
            folderTimer = None

        for _, i in enumerate(results, 1):
            try:
                if totalVideos > 1:
                    printSubsectionHeader(f"Video {_} of {totalVideos}")
                logInfo(f"Input: {results[i]['videoPath']}")

                VideoProcessor(
                    args,
                    results=results[i],
                )

            except Exception as e:
                logError(f"Error processing video {results[i]['videoPath']}: {str(e)}")
                logging.exception(f"Error processing video {results[i]['videoPath']}")

        if totalVideos > 1 and folderTimer is not None:
            totalTime = time() - folderTimer
            printSectionHeader("Batch Processing Summary")
            logSuccess(f"Total Execution Time: {totalTime:.2f} seconds")
            logSuccess(f"Average per video: {totalTime / totalVideos:.2f} seconds")
            logSuccess(f"Videos processed: {totalVideos}")

    except KeyboardInterrupt:
        logWarning("Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        logError(f"An unexpected error occurred: {str(e)}")
        logging.exception("Fatal error in main execution")
        sys.exit(1)


if __name__ == "__main__":
    main()
