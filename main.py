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

import gc
import logging
import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from time import time

os.environ.setdefault("FOR_DISABLE_CONSOLE_CTRL_HANDLER", "1")
os.environ.setdefault("PYTHONNOUSERSITE", "1")


def _disableUserSitePackages() -> None:
    """Keep bundled/runtime dependencies ahead of per-user Python packages."""
    try:
        import site

        userSite = site.getusersitepackages()
        if isinstance(userSite, str):
            userSitePaths = {userSite}
        else:
            userSitePaths = set(userSite or [])
        normalizedUserSites = {
            os.path.normcase(os.path.abspath(path)) for path in userSitePaths if path
        }
        if normalizedUserSites:
            sys.path[:] = [
                path
                for path in sys.path
                if os.path.normcase(os.path.abspath(path)) not in normalizedUserSites
            ]
        site.ENABLE_USER_SITE = False
    except Exception:
        pass


_disableUserSitePackages()

import src.constants as cs  # noqa: E402
from src.io.runOutcome import (  # noqa: E402
    outputWasWritten,
    truncatedDecodeError,
)

warnings.filterwarnings("ignore")


# The shell's own title, saved by the first _setTerminalTitle so it can be put
# back on exit instead of leaving the tab named after the last video.
_savedTerminalTitle = None


def _setTerminalTitle(title: str) -> None:
    """Name the terminal tab/window. Costs one syscall or one write: ctypes is
    already imported by the time the batch loop runs.

    Windows goes through SetConsoleTitleW rather than an OSC sequence, which
    conhost prints literally when the console has VT processing off; ConPTY
    forwards it to Windows Terminal. Elsewhere it is OSC 0, after pushing the
    shell's title onto the xterm title stack (ignored where unsupported)."""
    global _savedTerminalTitle
    if cs.ADOBE:
        return
    try:
        if sys.platform == "win32":
            import ctypes

            kernel32 = ctypes.windll.kernel32
            if _savedTerminalTitle is None:
                buf = ctypes.create_unicode_buffer(1024)
                if not kernel32.GetConsoleTitleW(buf, len(buf)):
                    return  # no console attached
                _savedTerminalTitle = buf.value
            kernel32.SetConsoleTitleW(title)
        elif sys.stdout.isatty():
            if _savedTerminalTitle is None:
                sys.stdout.write("\033[22;0t")
                _savedTerminalTitle = ""
            sys.stdout.write(f"\033]0;{title}\007")
            sys.stdout.flush()
    except Exception:
        pass


def _restoreTerminalTitle() -> None:
    global _savedTerminalTitle
    if _savedTerminalTitle is None:
        return
    try:
        if sys.platform == "win32":
            import ctypes

            ctypes.windll.kernel32.SetConsoleTitleW(_savedTerminalTitle)
        else:
            sys.stdout.write("\033[23;0t")
            sys.stdout.flush()
    except Exception:
        pass
    _savedTerminalTitle = None


def _notifyAdobeOfFatalError(error: Exception) -> None:
    """Tell the panel a run died on a path that never reaches _notifyAdobe.

    Every terminal status otherwise comes from ``VideoProcessor.start()``, so
    anything that raises before or outside it left After Effects showing the
    last progress string with no run behind it.
    """
    if not cs.ADOBE:
        return
    try:
        from src.server.aeComms import progressState

        # Several drivers report for themselves before re-raising, and a failed
        # video reports through _notifyAdobe before main() exits 1. One run
        # gets one terminal status.
        if progressState.get().get("status") in ("failed", "completed"):
            return
        progressState.setFailed(error=str(error))
    except Exception:
        logging.exception("Failed to report the error to After Effects")


def _videoFailed(
    processingError: Exception | None,
    outputPath: str,
    benchmark: bool = False,
    producesVideoFile: bool = True,
) -> bool:
    """Decide whether a processed video should be counted as a failure.

    A run failed if the frame loop stored an exception, or (for non-benchmark
    runs that write a video) the encoder produced no output file (missing or 0
    bytes). Benchmark runs write no output by design, so their output size is
    not checked -- and neither is a run whose artifact is not a video file:
    ``--autoclip`` writes its cut list to ``outputPath``, but zero cuts is a
    legitimate 0-byte result, so the non-empty check would report every
    cut-free scene-detection run as a failure and exit 1.
    """
    if processingError is not None:
        return True
    if benchmark or not producesVideoFile:
        return False
    return not outputWasWritten(outputPath)


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
        # Set by process() if the frame loop raises. start() reads this to
        # decide whether success stats are meaningful.
        self.processingError: Exception | None = None
        # --autoclip writes its cut list to self.output, but a clean run with
        # zero detected cuts is legitimately a 0-byte file, so the "non-empty
        # file at outputPath" success check does not apply.
        self.producesVideoFile: bool = not getattr(args, "autoclip", False)

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
        self.autoclip: bool = args.autoclip
        self.depth: bool = args.depth
        self.segment: bool = args.segment
        self.objDetect: bool = args.obj_detect
        self.stabilize: bool = args.stabilize
        self.stabilizeMethod: str = args.stabilize_method

        # The processing chain, one StageSpec per typed stage, in order (see
        # src/pipeline/chain.py). Every per-stage setting (methods, factors,
        # sensitivities, custom model) lives on these specs; the run-wide
        # args.<stage>_method-style attributes are not read here.
        self.chain: tuple = tuple(getattr(args, "chain", None) or ())
        self.smoothDedup: bool = getattr(args, "smooth_dedup", False)
        self.depthMethod: str = args.depth_method
        self.segmentMethod: str = args.segment_method
        self.segmentBatch: int = getattr(args, "segment_batch", 1)
        self.objDetectMethod: str = args.obj_detect_method
        self.objDetectDisableAnnotations: bool = args.obj_detect_disable_annotations

        # Quality and performance settings
        self.half: bool = args.half
        # Motion blur's own RIFE takes these run-wide; the interpolate stages
        # read theirs from the chain.
        self.ensemble: bool = getattr(args, "ensemble", False)
        self.forceStatic: bool = args.static
        self.dynamicScale: bool = getattr(args, "dynamic_scale", False)
        self.staticStep: bool = getattr(args, "static_step", False)
        self.compileMode: str = args.compile_mode
        # Normalize: argparse choices=["cpu","nvdec"] enforces lowercase on the
        # CLI path, but --json passes the value through verbatim and the AE
        # bridge emits uppercase "NVDEC". nelux normalizes internally via
        # .lower(), but our cache-invalidation guard in BuildBuffer does not,
        # so normalize once here at the source.
        self.decodeMethod: str = (args.decode_method or "cpu").lower()

        # Video processing settings
        self.inpoint: float | str = args.inpoint
        self.outpoint: float | str = args.outpoint
        self.bitDepth: str = args.bit_depth
        self.slowmo: bool = args.slowmo

        # Enhancement settings
        self.autoclipSens: float = args.autoclip_sens
        self.autoclipMethod: str = args.autoclip_method

        # Streaming scene-cut skip for the interpolation path (opt-in).
        self.sceneChange: bool = getattr(args, "scenechange", False)
        self.sceneChangeMethod: str = getattr(args, "scenechange_method", "ssim-cuda")
        self.sceneChangeThreshold = getattr(args, "scenechange_threshold", None)
        self.depthQuality: str = args.depth_quality
        self.depthWindow: int = getattr(args, "depth_window", 32)
        self.depthBatch: int = getattr(args, "depth_batch", 1)

        self.moblur: bool = args.moblur
        self.moblurMethod: str = args.moblur_method
        self.moblurFactor: int = args.moblur_factor
        self.moblurStrength: str = args.moblur_strength
        self.moblurShutterAngle: float = args.moblur_shutter_angle
        self.moblurLinearBlend: bool = not args.moblur_no_linear_blend

        # Protection mask, shared by motion blur and interpolation.
        self.maskPath: str = getattr(args, "mask", "")

        # Utility settings
        self.benchmark: bool = args.benchmark
        self.preview: bool = args.preview
        requestedPort = getattr(args, "preview_port", None)
        # An explicitly given port is honoured or reported busy; an unset one
        # may fall back to a free port. The parser default is None precisely so
        # these two are distinguishable.
        self.previewPortPinned: bool = requestedPort is not None
        self.previewPort: int | None = requestedPort
        self.profile: bool = args.profile
        self.singleImageInput: bool = getattr(args, "single_image_input", False)

    def _initVideoMetadata(self, args) -> None:
        """
        Initialize video metadata by analyzing the input video file.

        Args:
            args: Command line arguments containing inpoint and outpoint
        """
        # Lazy import to speed up startup for non-processing paths
        from src.io.getVideoMetadata import getVideoMetadata

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
                    f"pix_fmt='{pixFmt}'); NVDEC supports common compressed "
                    f"codecs (H.264, HEVC, VP9, AV1, MPEG-2/4, VC1, VP8, MJPEG) "
                    f"with YUV-family pixel formats. Falling back to CPU decode."
                )
                self.decodeMethod = "cpu"

    def _configureProcessingOptions(self, args) -> None:
        """
        Configure processing options based on the selected operations.

        Args:
            args: Command line arguments
        """
        logging.info("\n============== Processing Outputs ==============")

        # Every interpolate stage multiplies the frame rate. The product is of
        # the typed factors, so a single stage gives exactly fps * factor.
        self.fpsMultiplier = 1
        for spec in self.chain:
            if spec.kind == "interpolate":
                self.fpsMultiplier = self.fpsMultiplier * spec["factor"]

        if self.slowmo:
            self.outputFPS = self.fps
        else:
            self.outputFPS = (
                self.fps * self.fpsMultiplier if self.interpolate else self.fps
            )
        if self.outputFPS > 1000:
            from src.infra.logAndPrint import logAndPrint

            logAndPrint(
                f"The output frame rate is {self.outputFPS:.2f} fps (the source "
                f"{self.fps:.2f} fps times every interpolate factor); most players "
                f"and encoders handle 1000 fps or less.",
                colorFunc="yellow",
            )

        if self._isStandalone():
            # A standalone capability bypasses the stage chain: its one --resize
            # (the CLI allows no other stage alongside) is applied at decode,
            # the way the removed boolean --resize was.
            resizes = [spec for spec in self.chain if spec.kind == "resize"]
            if len(resizes) > 1:
                raise ValueError(
                    "a standalone capability (depth, segment, autoclip, stabilize, "
                    "obj_detect, moblur) accepts at most one --resize"
                )
            if resizes:
                from src.pipeline.planner import resizeDims

                self.width, self.height = resizeDims(
                    resizes[0]["target"], self.width, self.height
                )
                logging.info(f"Resizing to {self.width}x{self.height} at decode.")

    def _isStandalone(self) -> bool:
        """Whether _selectProcessingMethod hands this run to a standalone driver."""
        return bool(
            self.autoclip
            or self.depth
            or self.segment
            or self.objDetect
            or self.stabilize
            or self.moblur
        )

    def _selectProcessingMethod(self) -> None:
        """
        Select and execute the appropriate processing method based on user options.

        Prioritizes specialized operations (autoclip, depth, segment, object detection)
        over standard video processing.
        """
        if self.autoclip:
            logging.info("Detecting scene changes")
            from src.factories.standalone import autoClip

            autoClip(self)
        elif self.depth:
            logging.info("Depth Estimation")
            from src.factories.standalone import depth

            depth(self)
        elif self.segment:
            logging.info("Segmenting video")
            from src.factories.standalone import segment

            segment(self)
        elif self.objDetect:
            logging.info("Object Detection")
            from src.factories.standalone import objectDetection

            objectDetection(self)
        elif self.stabilize:
            logging.info("Stabilizing video")
            from src.factories.standalone import stabilize

            stabilize(self)

        elif self.moblur:
            logging.info("Applying motion blur")
            from src.factories.standalone import motionBlur

            motionBlur(self)

        else:
            self.start()

    def _checkRingMemory(self, pipeline) -> None:
        """Refuse a chain whose frame rings do not fit in the GPU memory left over.

        Runs after the models are built, so the weights, TensorRT engines and CUDA
        graph pools are already counted. Only frames a ring keeps beyond the one
        in flight are counted, at the ring's input size, and memory PyTorch has
        cached but not handed out counts as free: the allocator reuses it.
        """
        import torch

        # A run whose backends never touched CUDA keeps no frames on it, and
        # must not create a CUDA context just to ask.
        if not torch.cuda.is_initialized():
            return
        need = pipeline.ringBytes()
        if need == 0:
            return
        free, _total = torch.cuda.mem_get_info()
        free += torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
        if need > free:
            raise RuntimeError(
                f"the frame pipeline keeps {need / 1024**3:.2f} GiB of frames alive "
                f"but only {free / 1024**3:.2f} GiB of GPU memory is free after the "
                f"models loaded"
            )

    def process(self):
        """
        Main processing loop: decoded frames run through the planned stage chain.

        ``self.plan`` is the chain the typed flag order resolved to (see start()).
        The pipeline keeps its own counters, so a partial run still reports what
        it actually decoded and dropped.
        """
        from src.io.ffmpegSettings import adaptiveQueueDepth, closeWriterAndDrainReader
        from src.pipeline.executor import Pipeline

        frameCount = 0
        self.dedupCount = 0
        pipeline = None

        maskedSinkFactory = None
        if self.maskPath and self.plan.ofKind("interpolate"):
            from src.masking import MaskedSink, ProtectionMask

            # One mask (its weight tensors are cached per frame size, device
            # and dtype), one sink per interpolate stage.
            mask = ProtectionMask(self.maskPath)

            def maskedSinkFactory():
                return MaskedSink(mask)

        # Output frames per decoded frame: the product of every interpolate
        # stage's factor, each resolved the way the gap planner runs it.
        factor = Fraction(self.plan.factor)
        increment = factor.numerator / factor.denominator
        if increment.is_integer():
            increment = int(increment)

        try:
            # The bar counts output frames and only accepts integers, so a
            # fractional factor is projected onto the output grid rather than
            # multiplied per step -- `bar(2.5)` raised "an integer is required"
            # and killed every fractional-factor run before the first frame.
            def outputPosition(sourceFrames):
                return int(sourceFrames * increment)

            pipeline = Pipeline(
                self.plan,
                self.drivers,
                self.writeBuffer,
                maskedSinkFactory=maskedSinkFactory,
                half=self.half,
                # An interpolation gap is collected up to the writer's own queue
                # depth before it is handed on; past that the writer is the
                # backpressure, and a long --smooth_dedup hold stays bounded.
                collectLimit=adaptiveQueueDepth(self.new_width, self.new_height),
            )
            self.pipeline = pipeline
            self._checkRingMemory(pipeline)

            with self.ProgressBarLogic(
                outputPosition(self.totalFrames),
                title=os.path.splitext(os.path.basename(self.input))[0],
                titleDetail=self._barTitleDetail(),
                outputPath=self.output,
                videoFps=self.outputFPS,
                outputPerSource=increment,
            ) as bar:
                emitted = 0
                shownStats = (0, 0)
                read = self.readBuffer.read
                while pipeline.pull(read):
                    # The bar tracks decoded frames, not kept ones, so dedup'd
                    # frames still advance it.
                    position = outputPosition(pipeline.consumed)
                    if position != emitted:
                        bar(position - emitted)
                        emitted = position
                    stats = (pipeline.dropped, pipeline.cuts)
                    if stats != shownStats:
                        bar.setStats(*self._barStats(*stats))
                        shownStats = stats

                # End of stream: every stage releases what it still holds, so the
                # last gap is interpolated with no next frame.
                pipeline.drain()
                position = outputPosition(pipeline.consumed)
                if position != emitted:
                    bar(position - emitted)
                    emitted = position
                stats = (pipeline.dropped, pipeline.cuts)
                if stats != shownStats:
                    bar.setStats(*self._barStats(*stats))

                # Trailing duplicates are consumed after the last frame, so the bar
                # stopped short (598/600) unless they are counted here.
                pipeline.payoutTrailing()

                if pipeline.consumed != self.totalFrames:
                    bar.updateTotal(outputPosition(pipeline.consumed))

        except Exception as e:
            self.processingError = e
            logging.exception(f"Something went wrong while processing the frames, {e}")
        finally:
            # Counters live on the pipeline, so a partial run still reports what it
            # actually decoded and dropped.
            if pipeline is not None:
                frameCount = pipeline.consumed
                self.dedupCount = pipeline.dropped

            decodeError = truncatedDecodeError(
                self.readBuffer, self.totalFrames, self.writeBuffer
            )
            if self.processingError is None and decodeError is not None:
                self.processingError = decodeError
            # Always enqueue the writer's None sentinel (and close preview) even
            # when a frame raises, otherwise the writer/reader threads block
            # forever and the ThreadPoolExecutor join deadlocks the process.
            if self.preview:
                self.preview.close()
            # Closes the writer and drains the decode buffer, so the producer's
            # blocking put() returns and the reader thread can reach its
            # sentinel-enqueue finally. Without it, an exception in the frame loop
            # leaves the read thread blocked on put() to a full queue ->
            # ThreadPoolExecutor.__exit__ hangs (only KeyboardInterrupt escapes
            # today, via os._exit).
            closeWriterAndDrainReader(self.writeBuffer, self.readBuffer)

        logging.info(f"Processed {frameCount} frames")
        if self.dedupCount > 0:
            # Under --smooth_dedup nothing is removed from the output: the
            # duplicates are folded into wider interpolation gaps, so reporting
            # them as deduplicated would contradict the frame count.
            if self.smoothDedup:
                logging.info(f"Interpolated across {self.dedupCount} duplicate frames")
            else:
                logging.info(f"Deduplicated {self.dedupCount} frames")

    def _barTitleDetail(self) -> str:
        """The encoded output's resolution and frame rate, e.g. 2160p/60fps."""
        width, height = self.new_width, self.new_height
        fps = f"{self.outputFPS:.2f}".rstrip("0").rstrip(".")
        # "p" names the shorter side, so ultrawide 2560x1080 and portrait
        # 1080x1920 both read 1080p.
        return f"{min(width, height)}p/{fps}fps"

    def _barStats(self, dropped: int, cuts: int) -> tuple[str, str]:
        """Bar counters, full and short (for narrow terminals)."""
        full, short = [], []
        if dropped:
            # --smooth_dedup folds duplicates into wider gaps instead of
            # removing them, so "removed" would contradict the frame count.
            verb = "smoothed" if self.smoothDedup else "removed"
            full.append(f"{dropped} dupes {verb}")
            short.append(f"{dropped} dup")
        if cuts:
            full.append(f"{cuts} cut{'s' if cuts != 1 else ''}")
            short.append(f"{cuts} cut")
        return " · ".join(full), " · ".join(short)

    def start(self):
        """
        Initialize and start the video processing pipeline.

        Sets up input/output buffers, initializes AI models, and coordinates
        the multi-threaded processing workflow.
        """
        from src.infra.logAndPrint import logAndPrint
        from src.infra.progressBarLogic import ProgressBarLogic
        from src.io.ffmpegSettings import BuildBuffer, createWriteBuffer
        from src.server.aeComms import progressState

        self.ProgressBarLogic = ProgressBarLogic

        try:
            # Resolve the processing plan from the typed chain, then build every
            # driver at the frame size its stage receives.
            from src.initializeModels import initializeModels
            from src.pipeline.planner import planChain

            self.plan = planChain(
                self.chain,
                width=self.width,
                height=self.height,
                sceneChange=bool(self.sceneChange and self.interpolate),
                sceneChangeMethod=self.sceneChangeMethod,
            )
            # A leading resize is done by the decoder: from here on the frames
            # are that size.
            if self.plan.decodeResize:
                logging.info(
                    f"Resizing {self.width}x{self.height} to "
                    f"{self.plan.decodeWidth}x{self.plan.decodeHeight} at decode."
                )
            self.width = self.plan.decodeWidth
            self.height = self.plan.decodeHeight
            self.new_width = self.plan.outWidth
            self.new_height = self.plan.outHeight
            for note in self.plan.skipped:
                logging.info(f"Resize skipped, {note}")
            if self.plan.line:
                logAndPrint(self.plan.line, colorFunc="cyan")
            for hint in self.plan.hints:
                logAndPrint(hint, colorFunc="yellow")

            progressState.update({"status": "Initializing AI models..."})
            self.drivers = initializeModels(self, self.plan)

            starTime: float = time()

            progressState.update({"status": "Building input/output buffers..."})

            self.readBuffer = BuildBuffer(
                videoInput=self.input,
                inpoint=self.inpoint,
                outpoint=self.outpoint,
                half=self.half,
                resize=self.plan.decodeResize,
                width=self.width,
                height=self.height,
                bitDepth=self.bitDepth,
                decode_method=self.decodeMethod,
            )

            # Preview is sampled from the frames the writer already holds: the
            # writer's PreviewSampler pushes JPEGs into this sink and the HTTP
            # server reads the latest from it. No disk, and no encoder involvement.
            previewSink = None
            if self.preview:
                from src.server.previewSettings import PreviewSink

                previewSink = PreviewSink()

            self.writeBuffer = createWriteBuffer(
                input=self.input,
                output=self.output,
                encode_method=self.encodeMethod,
                width=self.new_width,
                height=self.new_height,
                fps=self.outputFPS,
                grayscale=False,
                transparent=False,
                benchmark=self.benchmark,
                bitDepth=self.bitDepth,
                inpoint=self.inpoint,
                outpoint=self.outpoint,
                sourceFps=self.fps,
                slowmo=self.slowmo,
                enablePreview=self.preview,
                previewSink=previewSink,
                single_image_output=self.singleImageInput,
            )

            if self.preview and getattr(self.writeBuffer, "enablePreview", False):
                from src.server.previewSettings import DEFAULT_PREVIEW_PORT, Preview

                self.preview = Preview(
                    previewSink=previewSink,
                    port=(
                        self.previewPort
                        if self.previewPort is not None
                        else DEFAULT_PREVIEW_PORT
                    ),
                    # A pinned port is a request, not a suggestion: report it
                    # busy rather than silently serving somewhere else.
                    allowPortFallback=not self.previewPortPinned,
                )
                self.preview.start()
            else:
                # Nothing to start, and nothing for process() to close.
                self.preview = None

            if self.profile:
                self._runWithProfiler()
            else:
                with ThreadPoolExecutor(max_workers=3) as executor:
                    executor.submit(self.readBuffer)
                    executor.submit(self.writeBuffer)
                    executor.submit(self.process)

            elapsedTime: float = time() - starTime

            # The bar's last filesize sample lands before the encoder drains its
            # queue and writes the container trailer, so read the real final
            # number here. A missing/0-byte output means the encoder never ran.
            try:
                finalSize = os.path.getsize(self.output)
            except (OSError, TypeError) as _e:
                finalSize = None

            if _videoFailed(
                self.processingError,
                self.output,
                self.benchmark,
                self.producesVideoFile,
            ):
                # Either process() caught a per-frame exception, or the encoder
                # produced no output file. Computing FPS from frameCount over
                # elapsedTime is meaningless in that case (frameCount is usually
                # 0 or 1, elapsedTime includes startup + model init). Report
                # failure clearly instead.
                reason = (
                    str(self.processingError)
                    if self.processingError is not None
                    else "encoder produced no output file (0 bytes / missing)"
                )
                logAndPrint(
                    f"Processing FAILED after {elapsedTime:.2f}s: {reason}",
                    colorFunc="red",
                    level="ERROR",
                )
            else:
                totalFPS: float = (
                    self.totalFrames
                    / elapsedTime
                    * (1 if not self.interpolate else self.fpsMultiplier)
                )
                logAndPrint(
                    f"Total Execution Time: {elapsedTime:.2f} seconds - FPS: {totalFPS:.2f}",
                    colorFunc="green",
                )
                if finalSize is not None and finalSize > 0:
                    sz = (
                        f"{finalSize / 1024**3:.2f} GB"
                        if finalSize >= 1024**3
                        else f"{finalSize / (1024 * 1024):.1f} MB"
                    )
                    logAndPrint(f"Output Size: {sz}", colorFunc="green")

            self._notifyAdobe(progressState)

        except Exception as e:
            logging.exception(f"Something went wrong while starting the processes, {e}")
            if cs.ADOBE:
                progressState.setFailed(error=str(e))

    def _notifyAdobe(self, progressState) -> None:
        """Emit the correct terminal status to the After Effects panel.

        A failed run (frame-loop exception or missing/0-byte output) must send
        ``setFailed``, not ``setCompleted``; the latter tells the panel the
        render succeeded and it then looks for an output file that is not there
        (issues #269, #236).
        """
        if not cs.ADOBE:
            return
        if _videoFailed(
            self.processingError, self.output, self.benchmark, self.producesVideoFile
        ):
            progressState.setFailed(
                error=str(self.processingError)
                if self.processingError is not None
                else "Output file not found after processing"
            )
        else:
            progressState.setCompleted(outputPath=self.output)

    def didFail(self) -> bool:
        """Whether this processed video should be counted as a batch failure."""
        return _videoFailed(
            self.processingError, self.output, self.benchmark, self.producesVideoFile
        )

    def _runWithProfiler(self):
        """
        Run the processing pipeline with torch.profiler enabled.
        Uses a simplified approach compatible with multi-threaded execution on Windows.
        """
        import torch
        from torch.profiler import ProfilerActivity, profile

        from src.infra.logAndPrint import logAndPrint

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
        except Exception:
            pass

    # Imported before the try, not inside it: the handlers at the bottom call
    # logWarning/logError, so a Ctrl-C or a failure during startup (the -h
    # short-circuit, argument parsing) used to die with NameError inside the
    # except block and lose the 130 exit code. The module has no import-time
    # dependency on anything set below.
    from src.infra.logAndPrint import (
        logError,
        logInfo,
        logSuccess,
        logWarning,
        printSectionHeader,
        printSubsectionHeader,
    )

    try:
        from platform import system

        cs.SYSTEM = system()
        cs.WHEREAMIRUNFROM = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(cs.WHEREAMIRUNFROM, exist_ok=True)

        if any(flag in sys.argv for flag in ("-h", "--help", "-v", "--version")):
            from src.cli.parser import createParser

            try:
                createParser(outputPath=os.getcwd())
            except SystemExit:
                return

        baseOutputPath = os.path.dirname(os.path.abspath(__file__))

        # Keep a single backend log in the runtime directory. Adobe Edition can
        # launch TAS repeatedly while rendering, so PID-suffixed logs quickly
        # accumulate and make support archives noisy. filemode="w" below keeps
        # the latest backend launch only.
        for filename in os.listdir(cs.WHEREAMIRUNFROM):
            suffix = filename.removeprefix("TAS-Log-").removesuffix(".log")
            if (
                filename.startswith("TAS-Log-")
                and filename.endswith(".log")
                and suffix.isdigit()
            ):
                try:
                    os.remove(os.path.join(cs.WHEREAMIRUNFROM, filename))
                except OSError:
                    pass

        cs.LOG_PATH = os.path.join(cs.WHEREAMIRUNFROM, "TAS-Log.log")
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

        from src.cli.parser import createParser
        from src.cli.validator import isAnyOtherProcessingMethodEnabled

        args = createParser(baseOutputPath)
        processingEnabled = isAnyOtherProcessingMethodEnabled(args)
        outputPath = os.path.join(baseOutputPath, "output")
        from src.io.inputOutputHandler import processInputOutputPaths

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

        succeededVideos = 0
        failedVideos = 0
        for idx, entry in enumerate(results, 1):
            processor = None
            try:
                _videoName = os.path.splitext(os.path.basename(entry["videoPath"]))[0]
                if len(_videoName) > 60:
                    _videoName = _videoName[:57] + "..."
                if totalVideos > 1:
                    _setTerminalTitle(f"TAS: {_videoName} [{idx}/{totalVideos}]")
                else:
                    _setTerminalTitle(f"TAS: {_videoName}")
                if totalVideos > 1:
                    printSubsectionHeader(f"Video {idx} of {totalVideos}")
                logInfo(f"Input: {entry['videoPath']}")

                if getattr(args, "png_passthrough", False) and not processingEnabled:
                    inputPath = entry["videoPath"]
                    outputPath = entry["outputPath"]

                    _runPngPassthrough(inputPath, outputPath)

                    logSuccess(f"PNG passthrough completed: {outputPath}")

                    if cs.ADOBE:
                        from src.server.aeComms import progressState

                        progressState.update(
                            {
                                "currentFrame": 1,
                                "totalFrames": 1,
                                "status": "Processing PNG preview...",
                            }
                        )
                        progressState.setCompleted(outputPath=outputPath)

                    succeededVideos += 1
                    continue

                processor = VideoProcessor(
                    args,
                    results=entry,
                )

                # The constructor runs the whole pipeline (start() / the
                # specialized factory paths) inline, so by the time it returns
                # the outcome is decided. start() swallows its own errors, so a
                # failed frame loop or a missing output surfaces only here.
                if processor.didFail():
                    failedVideos += 1
                    logError(f"Processing failed for video: {entry['videoPath']}")
                else:
                    succeededVideos += 1

            except Exception as e:
                failedVideos += 1
                logError(f"Error processing video {entry['videoPath']}: {str(e)}")
                logging.exception(f"Error processing video {entry['videoPath']}")
                # Anything raised before or outside start() -- unreadable
                # metadata, an unwired method, a download that ran out of
                # retries -- never reached _notifyAdobe, so the panel sat on
                # the last progress string with the run already over.
                _notifyAdobeOfFatalError(e)
                if totalVideos > 1 and processor is not None:
                    processor = None
                    tb = e.__traceback__
                    while tb is not None:
                        nextTb = tb.tb_next
                        try:
                            tb.tb_frame.clear()
                        except RuntimeError:
                            tb = nextTb
                            continue
                        tb = nextTb
                    gc.collect()
            finally:
                if totalVideos > 1 and processor is not None:
                    processor = None
                    gc.collect()

        _restoreTerminalTitle()

        if totalVideos > 1 and folderTimer is not None:
            totalTime = time() - folderTimer
            printSectionHeader("Batch Processing Summary")
            logSuccess(f"Total Execution Time: {totalTime:.2f} seconds")
            logSuccess(f"Average per video: {totalTime / totalVideos:.2f} seconds")
            logSuccess(
                f"Videos processed: {succeededVideos} succeeded, {failedVideos} failed"
            )

        # Make failure visible to shells and CI. KeyboardInterrupt keeps its own
        # exit path (130) below; this only covers processing/output failures.
        if failedVideos > 0:
            sys.exit(1)

    except KeyboardInterrupt:
        # A worker thread may still have a bar open, and os._exit below skips
        # its __exit__: end it first so barflow restores the cursor it hides
        # while drawing, and the warning lands on its own line instead of
        # over the bar. Looked up, not imported: no bar if it never loaded.
        try:
            bars = sys.modules.get("src.infra.progressBarLogic")
            if bars is not None:
                bars.closeLiveBars()
        except BaseException:
            pass
        logWarning("Process interrupted by user")
        _notifyAdobeOfFatalError(KeyboardInterrupt("Cancelled"))
        # The encoder runs in this process and writes its container
        # trailer in a worker thread's finally, which os._exit skips -- so a
        # cancelled render used to leave a large file with no moov atom that no
        # player could open. Bounded and best-effort: it never joins, never
        # raises, and the force-exit below happens either way.
        try:
            from src.io.ffmpegSettings import finalizeLiveWriters

            if not finalizeLiveWriters(timeout=5.0):
                logWarning(
                    "The in-process encoder did not finish writing the output "
                    "container in time; the partial file may not be playable."
                )
        except BaseException:
            # Must not preempt the force-exit below.
            pass
        _restoreTerminalTitle()
        # Force-exit: bypass blocked thread joins (TRT inference, nelux
        # decoder/encoder) which sys.exit() would hang on.
        os._exit(130)
    except SystemExit as e:
        # validator.py exits directly on unrunnable input, and the batch loop
        # exits 1 on a failed video. SystemExit is not an Exception, so the
        # handler below never saw either and the panel kept showing the last
        # progress string for a run that was already over.
        if e.code not in (0, None):
            _notifyAdobeOfFatalError(
                RuntimeError(f"TAS exited with code {e.code}; see TAS-Log.log")
            )
        raise
    except Exception as e:
        logError(f"An unexpected error occurred: {str(e)}")
        logging.exception("Fatal error in main execution")
        _notifyAdobeOfFatalError(e)
        sys.exit(1)
    finally:
        _restoreTerminalTitle()


if __name__ == "__main__":
    main()
