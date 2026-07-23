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

import logging
import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from queue import Empty
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
from src.io.frameWindow import FrameSlot  # noqa: E402

warnings.filterwarnings("ignore")


class _FrameCollector:
    """Small same-thread sink for interpolation outputs."""

    __slots__ = ("frames",)

    def __init__(self) -> None:
        self.frames = []

    def put(self, frame) -> None:
        self.frames.append(frame)

    def clear(self) -> None:
        self.frames.clear()


def _setTerminalTitle(title: str) -> None:
    if cs.ADOBE or not sys.stdout.isatty():
        return
    try:
        sys.stdout.write(f"\033]0;{title}\007")
        sys.stdout.flush()
    except Exception:
        pass


def _videoFailed(
    processingError: Exception | None, outputPath: str, benchmark: bool = False
) -> bool:
    """Decide whether a processed video should be counted as a failure.

    A run failed if the frame loop stored an exception, or (for non-benchmark
    runs) the encoder produced no output file (missing or 0 bytes). Benchmark
    runs write no output by design, so their output size is not checked.
    """
    if processingError is not None:
        return True
    if benchmark:
        return False
    try:
        return os.path.getsize(outputPath) <= 0
    except OSError, TypeError:
        return True


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
        # Set by process() if the frame loop raises. start() reads this to
        # decide whether success stats are meaningful.
        self.processingError: Exception | None = None

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
        self.segmentBatch: int = getattr(args, "segment_batch", 1)
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

        # Streaming scene-cut skip for the interpolation path (opt-in).
        self.sceneChange: bool = getattr(args, "scenechange", False)
        self.sceneChangeMethod: str = getattr(args, "scenechange_method", "ssim-cuda")
        self.sceneChangeThreshold = getattr(args, "scenechange_threshold", None)
        self.depthQuality: str = args.depth_quality
        self.depthNorm: bool = args.depth_norm
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

    def _enterFrame(self, rawFrame: any):
        """Admit a decoded frame into the window, or drop it.

        This is the pipeline's *entry* stage, run once per decoded frame in
        decode order by ``FrameWindow``. It performs the work that must happen
        before a frame can serve as anyone's neighbour:

        - dedup, on the raw frame, so duplicates never enter the window at all
          (a driver's "next frame" is the next frame that survives to the
          output, not a duplicate of the one it already has, and restore is not
          spent on frames that are about to be dropped);
        - restore, so every slot in the window shares one domain;
        - hard-cut detection, on the post-restore frame.

        Both detectors are stateful and see exactly the sequence of frames they
        saw when they ran inline, in the same order.
        """
        if self.dedup and self.dedup_process(rawFrame):
            return None

        frame = self.restore_process(rawFrame) if self.restore else rawFrame

        isCut = bool(
            self.interpolate
            and self.sceneChange_process is not None
            and self.sceneChange_process(frame)
        )
        return FrameSlot(frame, isCut)

    def _upscaledAt(self, offset: int):
        """Upscale of the slot at ``offset``, computed once and cached on it.

        The interpolation driver in ``ifInterpolateLast`` consumes upscaled
        frames, so a temporal driver's neighbour has to be upscaled too. Doing it
        through the window means each frame is upscaled exactly once even though
        two stages read it, and always in increasing frame order, which keeps
        AnimeSR's ``prevFrame``/``state`` recurrence in sequence.
        """
        return self.frameWindow.staged(
            offset,
            "upscaled",
            lambda index: self.upscale_process(
                self.frameWindow.at(index).frame,
                self.frameWindow.successorFrame(index),
            ),
        )

    def processFrame(self, slot: any) -> None:
        """
        Process a single video frame through the configured enhancement pipeline.

        Args:
            slot: The window slot being processed; dedup and restore already ran.
        """
        frame = slot.frame
        self._isCut = slot.isCut

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

    def _interpolateOrHold(self, frame: any, sink: any, nextFrame: any) -> None:
        """
        Feed the interpolation driver, or on a detected scene cut emit
        ``framesToInsert`` duplicates of the current frame into ``sink`` and
        reset the driver's frame/feature cache (via ``cacheFrameReset``) so the
        next interpolation anchors on this frame with no bleed from the previous
        scene. ``sink`` is ``self.interpQueue`` (interpolate-first) or
        ``self.writeBuffer`` (interpolate-last); both expose ``put``.

        With ``--mask``, the sink is wrapped so every emitted intermediate frame
        has its protected pixels restored from the segment's anchor frame. Held
        frames on a scene cut are already pristine copies, so they bypass the
        mask. Anchoring differs by driver: distildrba receives both endpoints and
        interpolates between ``frame`` and the window's next frame, so ``frame``
        is the anchor; every other driver keeps the previous frame internally and
        gets fed the segment's *end*, so the anchor is the previously fed frame.
        That keeps protected regions on the source frame that precedes them, so a
        subtitle or HUD swaps exactly on its own output slot instead of early.

        ``nextFrame`` is the driver's future context, in the same domain as
        ``frame``, or ``None`` at the end of the stream or across a scene cut.
        """
        if self._isCut:
            for _ in range(self.framesToInsert):
                sink.put(frame.clone())
            self.interpolate_process.cacheFrameReset(frame)
        elif self.interpolateMethod.startswith("distildrba"):
            if self.maskedSink is not None:
                sink = self.maskedSink.bind(sink, frame)
            self.interpolate_process(
                frame,
                nextFrame,
                sink,
                self.framesToInsert,
                self.timesteps,
            )
        else:
            if self.maskedSink is not None:
                sink = self.maskedSink.bind(sink, self.maskAnchor)
            self.interpolate_process(frame, sink, self.framesToInsert, self.timesteps)

        self.maskAnchor = frame

    def _drainInterpQueue(self) -> None:
        for item in self.interpQueue.frames:
            self.writeBuffer.write(item)
        self.interpQueue.clear()

    def ifInterpolateFirst(self, frame: any) -> None:
        """
        Process frame with interpolation-first pipeline order.

        Interpolation and upscaling both read the restore-domain source stream
        here -- interpolation consumes it directly, upscaling consumes its
        output -- so both take their neighbour straight from the window. A
        temporal upscaler applied to a generated intermediate is handed the
        source frame that closes the interval, which is the nearest real future
        frame that exists at that point in the pipeline.

        Args:
            frame: Restore-domain frame at the window's centre
        """
        nextFrame = self.frameWindow.successorFrame()

        if self.interpolate:
            self.interpQueue.clear()
            self._interpolateOrHold(frame, self.interpQueue, nextFrame)

        if self.upscale:
            if self.interpolate:
                for item in self.interpQueue.frames:
                    self.writeBuffer.write(self.upscale_process(item, nextFrame))
                self.interpQueue.clear()

                self.writeBuffer.write(self.upscale_process(frame, nextFrame))

            else:
                self.writeBuffer.write(self.upscale_process(frame, nextFrame))

        else:
            if self.interpolate:
                self._drainInterpQueue()
            self.writeBuffer.write(frame)

    def ifInterpolateLast(self, frame: any) -> None:
        """
        Process frame with interpolation-last pipeline order.

        Upscaling comes first, so the interpolation driver's stream *is* the
        upscaled stream and a temporal driver's neighbour must be upscaled too.
        Both come from the window's memoized upscale, so each frame is upscaled
        exactly once despite being read by two stages.

        Args:
            frame: Restore-domain frame at the window's centre
        """
        if self.upscale:
            frame = self._upscaledAt(0)

        nextFrame = None
        if self.interpolate and self._interpFuture:
            if self.upscale:
                # Guard first: a cut or the stream's end means no future context,
                # and upscaling the successor early would be pure waste.
                if self.frameWindow.successor() is not None:
                    nextFrame = self._upscaledAt(1)
            else:
                nextFrame = self.frameWindow.successorFrame()

        if self.interpolate:
            self._interpolateOrHold(frame, self.writeBuffer, nextFrame)

        self.writeBuffer.write(frame)

    def process(self):
        """
        Main processing loop that handles frame-by-frame video processing.

        Processes all frames through the configured enhancement pipeline and
        tracks processing statistics.
        """
        from src.io.frameWindow import FrameWindow, temporalDemand

        frameCount = 0
        self.dedupCount = 0
        self.frameCounter = 0

        self.maskedSink = None
        self.maskAnchor = None
        if self.interpolate and self.maskPath:
            from src.masking import MaskedSink, ProtectionMask

            self.maskedSink = MaskedSink(ProtectionMask(self.maskPath))

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
            self.interpQueue = _FrameCollector()

        # Drivers declare how many neighbouring frames they need handed to them.
        # Restore runs at window entry, so every slot already shares its domain
        # and it contributes no lookahead of its own.
        restorePast, restoreFuture = temporalDemand(self.restore_process)
        if restoreFuture or restorePast:
            raise NotImplementedError(
                "a temporal restore driver would need lookahead inside the window's "
                "entry pipeline, which is not wired up"
            )

        upscalePast, upscaleFuture = temporalDemand(
            self.upscale_process if self.upscale else None
        )
        interpPast, self._interpFuture = temporalDemand(
            self.interpolate_process if self.interpolate else None
        )

        if self.interpolateFirst:
            # Interpolation and upscaling both read the source stream, so their
            # demands overlap rather than stack.
            future = max(upscaleFuture, self._interpFuture)
        else:
            # Chained: interpolation's neighbour is an upscaled frame, whose own
            # neighbour is a source frame one further out. The demands compose.
            future = upscaleFuture + self._interpFuture
        past = max(upscalePast, interpPast)

        # Constructed before the try so the finally can always read its counters,
        # but this cannot raise (bounds are non-negative here).
        self.frameWindow = FrameWindow(
            self.readBuffer.read,
            past=past,
            future=future,
            enter=self._enterFrame,
        )

        try:
            with self.ProgressBarLogic(
                self.totalFrames * increment,
                outputPath=self.output,
                videoFps=self.outputFPS,
            ) as bar:
                consumed = 0
                while self.frameWindow.advance():
                    self.processFrame(self.frameWindow.centre)
                    # The bar tracks decoded frames, not kept ones, so dedup'd
                    # frames still advance it. Lookahead means the window may
                    # have consumed several by the time this centre is reached.
                    advanced = self.frameWindow.consumed - consumed
                    consumed = self.frameWindow.consumed
                    if advanced:
                        bar(increment * advanced)

                if self.frameWindow.consumed != self.totalFrames:
                    bar.updateTotal(self.frameWindow.consumed * increment)

        except Exception as e:
            self.processingError = e
            logging.exception(f"Something went wrong while processing the frames, {e}")
        finally:
            # Counters live on the window, so a partial run still reports what it
            # actually decoded and dropped.
            frameCount = self.frameWindow.consumed
            self.dedupCount = self.frameWindow.dropped
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
        from src.infra.logAndPrint import logAndPrint
        from src.infra.progressBarLogic import ProgressBarLogic
        from src.io.ffmpegSettings import BuildBuffer, createWriteBuffer
        from src.server.aeComms import progressState

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
                self.sceneChange_process,
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
                previewSink=previewSink,
                single_image_output=self.singleImageInput,
            )

            if self.preview and getattr(self.writeBuffer, "enablePreview", False):
                from src.server.previewSettings import Preview

                self.preview = Preview(previewSink=previewSink)
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

            if _videoFailed(self.processingError, self.output, self.benchmark):
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
                    * (1 if not self.interpolate else self.interpolateFactor)
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
        if _videoFailed(self.processingError, self.output, self.benchmark):
            progressState.setFailed(
                error=str(self.processingError)
                if self.processingError is not None
                else "Output file not found after processing"
            )
        else:
            progressState.setCompleted(outputPath=self.output)

    def didFail(self) -> bool:
        """Whether this processed video should be counted as a batch failure."""
        return _videoFailed(self.processingError, self.output, self.benchmark)

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

        from src.infra.logAndPrint import (
            logError,
            logInfo,
            logSuccess,
            logWarning,
            printSectionHeader,
            printSubsectionHeader,
        )

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

        _setTerminalTitle("TAS")

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
