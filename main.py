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

os.environ.setdefault("FOR_DISABLE_CONSOLE_CTRL_HANDLER", "1")

import logging
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from queue import Empty, Queue
from time import time

import src.constants as cs

warnings.filterwarnings("ignore")


def _setTerminalTitle(title: str) -> None:
    if cs.ADOBE or not sys.stdout.isatty():
        return
    try:
        sys.stdout.write(f"\033]0;{title}\007")
        sys.stdout.flush()
    except Exception:
        pass


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

        # Cleared at start(), set by process() if the frame loop raises.
        # start() reads this to decide whether FPS/Output-Size stats are
        # meaningful or should be suppressed (they are computed from a tiny
        # sample that includes startup + the failing frame, so reporting them
        # is misleading).
        self.processingError: Exception | None = None

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
        self.autoclip: bool = args.autoclip
        self.depth: bool = args.depth
        self.segment: bool = args.segment
        self.objDetect: bool = args.obj_detect
        self.stabilize: bool = args.stabilize

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
        # Normalize: argparse choices=["cpu","nvdec"] enforces lowercase on the
        # CLI path, but --json passes the value through verbatim and the AE
        # bridge emits uppercase "NVDEC". nelux normalizes internally via
        # .lower(), but our cache-invalidation guard in BuildBuffer does not,
        # so normalize once here at the source.
        self.decodeMethod: str = (args.decode_method or "cpu").lower()

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
        self.autoclipSens: float = args.autoclip_sens
        self.autoclipMethod: str = args.autoclip_method
        self.depthQuality: str = args.depth_quality
        self.depthNorm: bool = args.depth_norm

        self.moblur: bool = args.moblur
        self.moblurMethod: str = args.moblur_method
        self.moblurFactor: int = args.moblur_factor
        self.moblurStrength: str = args.moblur_strength
        self.moblurShutterAngle: float = args.moblur_shutter_angle
        self.moblurLinearBlend: bool = not args.moblur_no_linear_blend
        self.moblurMask: str = args.moblur_mask

        # Utility settings
        self.customModel: str = args.custom_model
        self.benchmark: bool = args.benchmark
        self.preview: bool = args.preview
        self.profile: bool = args.profile
        self.singleImageInput: bool = getattr(args, "single_image_input", False)

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

        # NVDEC's cuvid decoders only handle compressed codecs (H.264/HEVC/VP9/
        # AV1/...) with YUV-family pix_fmts. AE-bridge prerender AVIs are
        # typically `codec=rawvideo, pix_fmt=bgr24` (uncompressed RGB), which
        # has no cuvid decoder: nelux.VideoReader(decode_accelerator="nvdec")
        # then deadlocks or fails opaquely at construction, and the CPU
        # fallback in BuildBuffer.__call__ never fires because the constructor
        # hangs rather than raising. Detect it up front and downgrade.
        if self.decodeMethod == "nvdec":
            from src.io.getVideoMetadata import isNvdecCompatible

            codec = videoMetadata["Codec"]
            pixFmt = videoMetadata["ColorFormat"]
            if not isNvdecCompatible(codec, pixFmt):
                logging.warning(
                    f"NVDEC cannot decode this source (codec='{codec}', "
                    f"pix_fmt='{pixFmt}'); NVDEC only handles H.264/HEVC/VP9/AV1 "
                    f"with YUV pix_fmts. Falling back to CPU decode."
                )
                self.decodeMethod = "cpu"

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
        elif self.stabilize:
            logging.info("Stabilizing video")
            from src.initializeModels import stabilize

            stabilize(self)

        elif self.moblur:
            logging.info("Applying motion blur")
            from src.initializeModels import motionBlur

            motionBlur(self)

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

    def _drainInterpQueue(self) -> None:
        while True:
            try:
                item = self.interpQueue.get_nowait()
            except Empty:
                break
            self.writeBuffer.write(item)

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
                while True:
                    try:
                        item = self.interpQueue.get_nowait()
                    except Empty:
                        break
                    self.writeBuffer.write(self.upscale_process(item, self.nextFrame))

                self.writeBuffer.write(self.upscale_process(frame, self.nextFrame))

            else:
                self.writeBuffer.write(self.upscale_process(frame, self.nextFrame))

        else:
            if self.interpolate:
                self._drainInterpQueue()
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
            currentFrame = self.readBuffer.read()
            nextFrame = self.readBuffer.read() if currentFrame is not None else None

            with self.ProgressBarLogic(
                self.totalFrames * increment,
                outputPath=self.output,
                videoFps=self.outputFPS,
            ) as bar:
                while currentFrame is not None:
                    if self.upscaleMethod == "animesr" or (
                        self.interpolate
                        and self.interpolateMethod.startswith("distildrba")
                    ):
                        self.nextFrame = nextFrame
                    self.processFrame(currentFrame)
                    frameCount += 1
                    bar(increment)

                    currentFrame = nextFrame
                    if currentFrame is not None:
                        nextFrame = self.readBuffer.read()

                if frameCount != self.totalFrames:
                    bar.updateTotal(frameCount * increment)

        except Exception as e:
            self.processingError = e
            logging.exception(f"Something went wrong while processing the frames, {e}")
        finally:
            # Always enqueue the writer's None sentinel (and close preview) even
            # when a frame raises, otherwise the writer/reader threads block
            # forever and the ThreadPoolExecutor join deadlocks the process.
            if self.preview:
                self.preview.close()
            self.writeBuffer.close()
            # Drain the decode buffer so the producer's blocking put() returns
            # and the reader thread can reach its sentinel-enqueue finally.
            # Without this, an exception in processFrame leaves the read thread
            # blocked on put() to a full queue -> ThreadPoolExecutor.__exit__
            # hangs (only KeyboardInterrupt escapes today, via os._exit).
            while not self.readBuffer.isReadFinished():
                try:
                    self.readBuffer.decodeBuffer.get(timeout=0.1)
                except Empty:
                    continue

        logging.info(f"Processed {frameCount} frames")
        if self.dedupCount > 0:
            logging.info(f"Deduplicated {self.dedupCount} frames")

    def start(self):
        """
        Initialize and start the video processing pipeline.

        Sets up input/output buffers, initializes AI models, and coordinates
        the multi-threaded processing workflow.
        """
        from src.utils.aeComms import progressState
        from src.utils.ffmpegSettings import BuildBuffer, createWriteBuffer
        from src.utils.logAndPrint import logAndPrint
        from src.utils.progressBarLogic import ProgressBarLogic

        self.ProgressBarLogic = ProgressBarLogic

        try:
            # Initialize AI models and get processing functions
            from src.initializeModels import initializeModels

            progressState.update({"status": "Initializing AI models..."})

            (
                self.new_width,
                self.new_height,
                self.upscale_process,
                self.interpolate_process,
                self.restore_process,
                self.dedup_process,
            ) = initializeModels(self)

            starTime: float = time()

            progressState.update({"status": "Building input/output buffers..."})

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

            self.writeBuffer = createWriteBuffer(
                input=self.input,
                output=self.output,
                encode_method=self.encodeMethod,
                custom_encoder=self.customEncoder,
                width=self.new_width,
                height=self.new_height,
                fps=self.outputFPS,
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
                single_image_output=self.singleImageInput,
            )

            if self.preview and self.writeBuffer.previewPath is not None:
                from src.utils.previewSettings import Preview

                self.preview = Preview(previewPath=self.writeBuffer.previewPath)
                self.preview.start()
            else:
                # No preview surface (e.g. nelux encoders emit no preview image),
                # so there is nothing to start and nothing for process() to close.
                self.preview = None

            if self.profile:
                self._runWithProfiler()
            else:
                with ThreadPoolExecutor(max_workers=3) as executor:
                    executor.submit(self.readBuffer)
                    executor.submit(self.writeBuffer)
                    executor.submit(self.process)

            elapsedTime: float = time() - starTime
            if self.processingError is not None:
                # process() caught a per-frame exception. Computing FPS from
                # frameCount over elapsedTime is meaningless (frameCount is
                # usually 0 or 1, elapsedTime includes startup + model init),
                # and the output file is typically missing/empty because the
                # encoder pipe was starved. Report failure clearly instead.
                logAndPrint(
                    f"Processing FAILED after {elapsedTime:.2f}s: {self.processingError}",
                    colorFunc="red",
                    level="ERROR",
                )
            else:
                totalFPS: float = (
                    self.totalFrames
                    / elapsedTime
                    * (1 if not self.interpolate else self.interpolateFactor)
                )
                logAndPrint(
                    f"Total Execution Time: {elapsedTime:.2f} seconds - FPS: {totalFPS:.2f}",
                    colorFunc="green",
                )
                # The bar's last filesize sample lands before the encoder
                # drains its queue and writes the container trailer; this is
                # the real, final number.
                try:
                    finalSize = os.path.getsize(self.output)
                except OSError, TypeError:
                    finalSize = None
                if finalSize is not None and finalSize > 0:
                    sz = (
                        f"{finalSize / 1024**3:.2f} GB"
                        if finalSize >= 1024**3
                        else f"{finalSize / (1024 * 1024):.1f} MB"
                    )
                    logAndPrint(f"Output Size: {sz}", colorFunc="green")
                else:
                    logAndPrint(
                        "Output Size: 0 bytes (encoder produced no frames)",
                        colorFunc="yellow",
                        level="WARNING",
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
        from torch.profiler import ProfilerActivity, profile

        from src.utils.logAndPrint import logAndPrint

        profilePath = os.path.join(cs.WHEREAMIRUNFROM, "profiler_trace")
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


def _runPngPassthrough(inputPath: str, outputPath: str) -> None:
    """Decode an image (OpenCV → Pillow fallback), round-trip through torch, write PNG."""
    outputDir = os.path.dirname(outputPath)
    if outputDir:
        os.makedirs(outputDir, exist_ok=True)

    tensorFrame = None
    cv2Module = None

    try:
        import cv2
        import torch

        cv2Module = cv2

        image = cv2.imread(inputPath, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"Failed to decode image with OpenCV: {inputPath}")

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        tensorFrame = torch.from_numpy(image)

    except Exception as cvError:
        logging.warning(f"OpenCV decode failed, trying Pillow fallback: {cvError}")
        import numpy as np
        import torch
        from PIL import Image

        pilImage = Image.open(inputPath).convert("RGB")
        tensorFrame = torch.from_numpy(np.array(pilImage))

    outputFrame = tensorFrame.cpu().numpy()
    if cv2Module is not None:
        if outputFrame.ndim == 2:
            writeOk = cv2Module.imwrite(outputPath, outputFrame)
        elif outputFrame.shape[2] == 4:
            writeOk = cv2Module.imwrite(
                outputPath,
                cv2Module.cvtColor(outputFrame, cv2Module.COLOR_RGBA2BGRA),
            )
        else:
            writeOk = cv2Module.imwrite(
                outputPath,
                cv2Module.cvtColor(outputFrame, cv2Module.COLOR_RGB2BGR),
            )

        if not writeOk:
            raise RuntimeError(f"Failed to write output PNG: {outputPath}")
    else:
        from PIL import Image

        Image.fromarray(outputFrame).save(outputPath)


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
        except AttributeError, Exception:
            pass

    try:
        from platform import system

        cs.SYSTEM = system()
        cs.WHEREAMIRUNFROM = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(cs.WHEREAMIRUNFROM, exist_ok=True)

        if any(flag in sys.argv for flag in ("-h", "--help", "-v", "--version")):
            from src.utils.argumentsChecker import createParser

            try:
                createParser(outputPath=os.getcwd())
            except SystemExit:
                return

        from src.utils.logAndPrint import (
            logError,
            logInfo,
            logSuccess,
            logWarning,
            printSectionHeader,
            printSubsectionHeader,
        )

        baseOutputPath = os.path.dirname(os.path.abspath(__file__))

        # Per-run log file keyed by PID. The AE bridge (and batch usage) can
        # launch multiple TAS workers against the same cs.WHEREAMIRUNFROM
        # directory; with the old fixed name + filemode="w", each worker
        # truncated the same TAS-Log.log and their outputs interleaved at the
        # line level (init block + "Processed/Encoded N frames" appearing
        # twice per run). A PID-suffixed name gives each worker its own file
        # so concurrent runs no longer pollute each other. The AE frontend
        # receives progress over Socket.IO, not by tailing this file, so the
        # rename is transparent to it.
        cs.LOG_PATH = os.path.join(cs.WHEREAMIRUNFROM, f"TAS-Log-{os.getpid()}.log")
        logging.basicConfig(
            filename=cs.LOG_PATH,
            filemode="w",
            format="[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.DEBUG,
        )

        logging.info("\n" + "=" * 80)
        logging.info("Command Line Arguments".center(80))
        logging.info("=" * 80)
        logging.info(f"{' '.join(sys.argv)}\n")
        logging.info(f"Log file: {cs.LOG_PATH}")

        from src.utils.argumentsChecker import (
            createParser,
            isAnyOtherProcessingMethodEnabled,
        )

        args = createParser(baseOutputPath)
        processingEnabled = isAnyOtherProcessingMethodEnabled(args)
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

        for idx, entry in enumerate(results, 1):
            try:
                _videoName = os.path.basename(entry["videoPath"])
                if len(_videoName) > 60:
                    _videoName = _videoName[:57] + "..."
                if totalVideos > 1:
                    _setTerminalTitle(
                        f"\u2605 Processing - {_videoName} [{idx}/{totalVideos}]"
                    )
                else:
                    _setTerminalTitle(f"\u2605 Processing - {_videoName}")
                if totalVideos > 1:
                    printSubsectionHeader(f"Video {idx} of {totalVideos}")
                logInfo(f"Input: {entry['videoPath']}")

                if getattr(args, "png_passthrough", False) and not processingEnabled:
                    inputPath = entry["videoPath"]
                    outputPath = entry["outputPath"]

                    _runPngPassthrough(inputPath, outputPath)

                    logSuccess(f"PNG passthrough completed: {outputPath}")

                    if cs.ADOBE:
                        from src.utils.aeComms import progressState

                        progressState.update(
                            {
                                "currentFrame": 1,
                                "totalFrames": 1,
                                "status": "Processing PNG preview...",
                            }
                        )
                        progressState.setCompleted(outputPath=outputPath)

                    continue

                VideoProcessor(
                    args,
                    results=entry,
                )

            except Exception as e:
                logError(f"Error processing video {entry['videoPath']}: {str(e)}")
                logging.exception(f"Error processing video {entry['videoPath']}")

        _setTerminalTitle("TAS")

        if totalVideos > 1 and folderTimer is not None:
            totalTime = time() - folderTimer
            printSectionHeader("Batch Processing Summary")
            logSuccess(f"Total Execution Time: {totalTime:.2f} seconds")
            logSuccess(f"Average per video: {totalTime / totalVideos:.2f} seconds")
            logSuccess(f"Videos processed: {totalVideos}")

    except KeyboardInterrupt:
        logWarning("Process interrupted by user")
        # Force-exit: bypass blocked thread joins (TRT inference, ffmpeg subprocess
        # wait, nelux decoder) which sys.exit() would hang on.
        os._exit(130)
    except Exception as e:
        logError(f"An unexpected error occurred: {str(e)}")
        logging.exception("Fatal error in main execution")
        sys.exit(1)


if __name__ == "__main__":
    main()
