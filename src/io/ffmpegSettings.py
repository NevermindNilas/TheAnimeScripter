import logging
import os
import subprocess
import threading
import time
import weakref
from queue import Empty, Queue

import torch  # this has to be always before nelux!
from torch.nn import functional as F


def _repair_nelux_macos_ffmpeg_links() -> None:
    from src.infra.dependencyHandler import repairNeluxMacosFfmpegLinks

    repairNeluxMacosFfmpegLinks()


try:
    import nelux
except ImportError:
    _repair_nelux_macos_ffmpeg_links()
    import nelux
except OSError:
    _repair_nelux_macos_ffmpeg_links()
    import nelux

import src.constants as cs
from src.io.encodingSettings import (
    colorSpaceFilter,
    colorSpaceOptions,
    getPixFMT,
    matchEncoder,
    matchNeluxEncoder,
)
from src.io.getVideoMetadata import trimFrameRange
from src.io.inputOutputHandler import SEQUENCE_EXTENSIONS
from src.infra.isCudaInit import CudaChecker
from src.infra.logAndPrint import logWarning

checker = CudaChecker()

# Debugging flag
neluxLog = False

CachedReader = None
CachedReaderMethod = None
CachedReaderResize = None


def _drainQueueUntilSentinel(queue) -> None:
    """Consume queued frames until the producer's None sentinel arrives.

    Used by the write buffers on early-exit paths (encoder startup failure or
    a raised exception): the producer keeps put()-ing frames and always
    close()es once its own loop ends, so without this it blocks forever on a
    full queue and the run deadlocks. The 5s idle escape covers a producer
    that itself died before calling close().
    """
    while True:
        try:
            item = queue.get(timeout=5.0)
        except Empty:
            logging.warning(
                "Writer drain: no sentinel within 5s; producer likely dead."
            )
            return
        if item is None:
            return


# Writers that finalize their container in-process and are currently running.
# Only NeluxWriteBuffer belongs here: the FFmpeg WriteBuffer's close() waits on
# a subprocess, which is exactly the hang os._exit exists to dodge, and its
# child finalizes itself on stdin EOF anyway.
_LIVE_NELUX_WRITERS = weakref.WeakSet()


def finalizeLiveWriters(timeout: float = 5.0) -> bool:
    """Let every in-flight in-process encoder write its container trailer.

    ``NeluxWriteBuffer`` encodes in this process and finalizes the container
    only in its worker thread's ``finally``. ``main.py``'s KeyboardInterrupt
    handler ends the process with ``os._exit(130)``, which runs no ``finally``
    on any thread, so cancelling a ``*_nelux`` render left a multi-hundred-MB
    file with no moov atom that no player or tool can open -- where the
    FFmpeg-subprocess writer leaves a valid, playable partial, because its
    child sees stdin EOF and finalizes itself.

    Bounded and best-effort by design: a writer can be parked inside a native
    ``encode_frame`` where no Python-level flag preempts it, so the caller must
    still ``os._exit`` afterwards rather than joining. Never raises -- an
    exception escaping here would skip that exit and change the exit code.

    Returns True if every live writer finalized within the deadline.
    """
    try:
        writers = list(_LIVE_NELUX_WRITERS)
        if not writers:
            return True

        finished = True
        # Ask every writer to stop before waiting on any of them, and isolate
        # each one: a single bad entry must not leave the healthy writers
        # un-asked and then time out waiting for them.
        for writer in writers:
            try:
                writer.requestStop()
            except BaseException:
                finished = False

        deadline = time.monotonic() + timeout
        for writer in writers:
            remaining = max(deadline - time.monotonic(), 0.0)
            try:
                if not writer.waitFinalized(remaining):
                    finished = False
            except BaseException:
                finished = False
        return finished
    except BaseException:
        # Must never escape: an exception here would skip the caller's
        # os._exit(130) and change the process exit code.
        return False


def _resolveOutputScale(
    width: int | None, height: int | None
) -> tuple[int | None, int | None]:
    """The `--output_scale` target for a writer, explicit value or run default.

    Only ``main.py`` ever passed these; every standalone driver (depth,
    segment, stabilize, moblur, obj_detect) builds its writer from a long
    positional argument list that never carried them, so `--output_scale` was
    silently discarded for half the toolkit while `src/cli/validator.py` logged
    that it had been applied. The flag is run-wide, so falling back to
    ``cs.OUTPUT_SCALE_*`` here fixes every driver at once and keeps a future
    one from inheriting the same omission. An explicit argument still wins, and
    a partial pair (one dimension only) is treated as unset, matching the
    ``and`` guards at both consumers.
    """
    if width is None and height is None:
        return cs.OUTPUT_SCALE_WIDTH, cs.OUTPUT_SCALE_HEIGHT
    return width, height


def _probedMetadata() -> dict:
    """The current input's probed metadata dict, or {} when unavailable.

    getVideoMetadata.saveMetadata records it in-process and also writes the
    file, once per input in a batch, before any writer is built. Both writers
    read it for their colour-space decision: WriteBuffer via colorSpaceFilter,
    NeluxWriteBuffer via colorSpaceOptions.

    The in-process copy wins. metadata.json is one fixed install-dir path
    shared by every TAS process on the machine, so a second run overwriting it
    mid-render handed this run the other run's colour space. The file read
    stays as the fallback for callers that set cs.METADATAPATH directly.
    """
    import json

    if cs.PROBED_METADATA is not None:
        return cs.PROBED_METADATA

    if cs.METADATAPATH and os.path.exists(cs.METADATAPATH):
        try:
            with open(cs.METADATAPATH, encoding="utf-8") as f:
                return json.load(f).get("metadata", {}) or {}
        except Exception as e:
            logging.warning(f"Failed to read metadata for color space: {e}")
    return {}


def drainReader(readBuffer) -> None:
    """Consume decoded frames until the reader thread reaches its sentinel.

    The reader blocks on ``put()`` into the 32-deep decode queue, so a consumer
    that stops early (because it raised) pins it there and every join on it
    hangs. Draining lets it run out and set ``isFinished``.
    """
    while not readBuffer.isReadFinished():
        try:
            readBuffer.decodeBuffer.get(timeout=0.1)
        except Empty:
            continue


def closeWriterAndDrainReader(writeBuffer, readBuffer) -> None:
    """Release both pipeline threads once a frame loop ends, however it ends.

    Mirrors the ``finally`` in ``main.py:process``. Neither thread can escape on
    its own: the writer's main loop swallows the queue timeout and spins until
    it sees the ``None`` sentinel, and the reader blocks on ``put()``. So a
    ``process()`` that raises before its own ``close()`` leaves
    ``ThreadPoolExecutor.__exit__`` joining forever, with the original exception
    buried in a future nobody reads.
    """
    writeBuffer.close()
    drainReader(readBuffer)


class BuildBuffer:
    def __init__(
        self,
        videoInput: str = "",
        inpoint: float = 0.0,
        outpoint: float = 0.0,
        half: bool = True,
        resize: bool = False,
        width: int = 1920,
        height: int = 1080,
        bitDepth: str = "8bit",
        toTorch: bool = True,
        decode_method: str = "cpu",
    ):
        """
        Initializes the BuildBuffer class.

        Args:
            videoInput (str): Path to the input video file.
            inpoint (float): Start time of the segment to decode, in seconds.
            outpoint (float): End time of the segment to decode, in seconds.
            half (bool): Whether to use half precision (float16) for tensors.
            resize (bool): Whether to resize the frames.
            width (int): Width of the output frames.
            height (int): Height of the output frames.
            bitDepth (str): Bit depth of the output frames, e.g., "8bit" or "10bit".
            toTorch (bool): Whether to convert frames to torch tensors.
            decode_method (str): The backend to use for decoding, e.g., "cpu" or "nvdec".

        Note:
            Nelux returns HWC format [H, W, 3] with native dtype (uint8/int16).
            processFrameToTorch converts this to BCHW float format for processing.
        """
        self.decodeMethod = decode_method
        self.half = half
        self.decodeBuffer = Queue(maxsize=32)
        self.width = width
        self.height = height
        self.resize = resize
        self.isFinished = False
        self.bitDepth = bitDepth
        self.videoInput = os.path.normpath(videoInput)
        # Set when decoding dies with no usable fallback. Consumers only ever
        # see the end-of-stream sentinel, so without this a truncated decode is
        # indistinguishable from a clean EOF.
        self.decodeError: Exception | None = None
        self.toTorch = toTorch
        self.inpoint = inpoint
        self.outpoint = outpoint
        self._didDecoderResize = False

        # USED FOR DEBUGGING neLux
        global neluxLog
        if neluxLog:
            try:
                nelux.set_log_level(nelux.LogLevel.info)
                neluxLog = True
                logging.info("neLux logging enabled (level: info)")
            except Exception as e:
                logging.warning(f"Failed to enable neLux logging: {e}")

        if "%" not in videoInput and not os.path.exists(videoInput):
            raise FileNotFoundError(f"Video file not found: {videoInput}")

        self.cudaEnabled = False
        if checker.cudaAvailable and toTorch:
            try:
                self.normStream = torch.cuda.Stream()
                self.deviceType = "cuda"
                self.cudaEnabled = True
            except Exception as e:
                logging.warning(
                    f"CUDA stream init failed, falling back to CPU. Reason: {e}"
                )
                self.deviceType = "cpu"
                self.cudaEnabled = False
        else:
            self.deviceType = "cpu"
            self.cudaEnabled = False

        self.backend = "pytorch" if toTorch else "numpy"

    def __call__(self):
        """
        Decodes frames from the video and stores them in the decodeBuffer.
        """
        decodedFrames = 0
        # Number of frames already pushed into the (concurrently-consumed)
        # decodeBuffer this run; used to forbid a fallback re-decode that would
        # duplicate frames after a mid-stream failure.
        self._emittedFrames = 0
        global CachedReader
        global CachedReaderMethod

        try:
            decodedFrames += self._decodeWithnelux(self.decodeMethod)

        except Exception as e:
            logging.error(f"nelux decoding error: {e}")
            # The error worth surfacing is the last one, not the first: an
            # nvdec failure that the CPU retry then died part-way through
            # should report the retry's error, since that is the one that
            # truncated the stream.
            lastError = e

            # A fallback re-decode restarts from frame 0. If the failed attempt
            # already emitted frames, re-decoding duplicates them, so only fall
            # back when nothing has been queued yet.
            if self._emittedFrames == 0 and self.decodeMethod != "cpu":
                try:
                    logging.warning(
                        "nelux decode failed with non-CPU method; retrying with cpu."
                    )
                    decodedFrames += self._decodeWithnelux("cpu")
                    self.decodeMethod = "cpu"
                    return
                except Exception as retry_e:
                    logging.error(f"nelux CPU retry failed: {retry_e}")
                    lastError = retry_e

            if self._emittedFrames == 0:
                logging.info("Attempting fallback to OpenCV decoder...")
                try:
                    decodedFrames += self.decodeWithOpenCV()
                except Exception as fallback_e:
                    logging.error(f"OpenCV fallback failed: {fallback_e}")
                    self.decodeError = fallback_e
            else:
                logging.error(
                    f"Decode failed after {self._emittedFrames} frame(s) were already "
                    f"queued; skipping fallback to avoid duplicating frames."
                )
                # Every consumer sees only the sentinel below, which is
                # indistinguishable from a clean EOF, so a decode that died
                # mid-stream used to finish as a "successful" short video.
                # src/io/runOutcome.py:truncatedDecodeError promotes this into
                # the run's processingError, but only when frames are missing.
                self.decodeError = lastError

        finally:
            self.decodeBuffer.put(None)

            self.isFinished = True
            logging.info(f"Decoded {decodedFrames} frames")

    def _decodeWithnelux(self, decodeMethod: str) -> int:
        """
        Decode using nelux. Returns number of frames decoded.
        """
        global CachedReader
        global CachedReaderMethod
        global CachedReaderResize

        resizeTarget = (self.width, self.height) if self.resize else None

        if decodeMethod.lower() == "nvdec":
            CachedReader = None
            CachedReaderMethod = None
            CachedReaderResize = None

        if CachedReader is not None and CachedReaderMethod != decodeMethod:
            CachedReader = None
            CachedReaderMethod = None
            CachedReaderResize = None

        if CachedReader is not None and CachedReaderResize != resizeTarget:
            CachedReader = None
            CachedReaderMethod = None
            CachedReaderResize = None

        if CachedReader is not None:
            try:
                logging.info(f"Reconfiguring cached VideoReader for {self.videoInput}")
                CachedReader.reconfigure(self.videoInput)
            except Exception as e:
                logging.warning(
                    f"Failed to reconfigure VideoReader: {e}. Creating new instance."
                )
                CachedReader = None
                CachedReaderMethod = None
                CachedReaderResize = None

        if CachedReader is None:
            logging.info(
                f"Initializing new VideoReader for {self.videoInput} ({decodeMethod})"
                + (f" resize={resizeTarget}" if resizeTarget else "")
            )
            CachedReader = nelux.VideoReader(
                self.videoInput,
                decode_accelerator=decodeMethod,
                backend=self.backend,
                resize=resizeTarget,
            )
            CachedReaderMethod = decodeMethod
            CachedReaderResize = resizeTarget

        self._didDecoderResize = resizeTarget is not None

        fps = CachedReader.fps or 0.0
        # Shared with getVideoMetadata so the decode and the reported frame
        # count cannot drift apart -- see trimFrameRange. endFrame is None when
        # no --outpoint was requested; testing that instead of `endFrame > 0`
        # is what keeps a sub-frame --outpoint from reading as "no limit".
        startFrame, endFrame = trimFrameRange(fps, self.inpoint, self.outpoint)

        decodedFrames = 0
        frameIdx = 0
        for frame in CachedReader:
            if frameIdx < startFrame:
                frameIdx += 1
                continue
            if endFrame is not None and frameIdx >= endFrame:
                break
            if self.toTorch:
                frame = self.processFrameToTorch(
                    frame, self.normStream if self.cudaEnabled else None
                )
            self.decodeBuffer.put(frame)
            decodedFrames += 1
            self._emittedFrames += 1
            frameIdx += 1

        return decodedFrames

    def decodeWithOpenCV(self):
        """
        Helper method to decode using OpenCV when nelux fails.
        Returns the number of frames decoded.
        """
        logging.info(f"Initializing OpenCV VideoCapture for {self.videoInput}")

        import cv2  # lazy: cv2 (~13ms) is only needed on the OpenCV decode fallback

        cap = cv2.VideoCapture(self.videoInput)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video with OpenCV: {self.videoInput}")

        totalFramesDecoded = 0
        startTime = float(self.inpoint)
        endTime = float(self.outpoint)

        try:
            if startTime > 0:
                cap.set(cv2.CAP_PROP_POS_MSEC, startTime * 1000.0)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                ptsMillis = cap.get(cv2.CAP_PROP_POS_MSEC)
                ptsSeconds = (
                    (ptsMillis / 1000.0) if ptsMillis and ptsMillis > 0 else None
                )

                if ptsSeconds is not None and endTime > 0 and ptsSeconds >= endTime:
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = torch.from_numpy(frame)

                if self.toTorch:
                    frame = self.processFrameToTorch(
                        frame,
                        self.normStream if self.cudaEnabled else None,
                    )
                else:
                    frame = frame.numpy()

                self.decodeBuffer.put(frame)
                totalFramesDecoded += 1
                self._emittedFrames += 1

        except Exception as e:
            logging.error(f"Error during OpenCV decoding loop: {e}")
            raise
        finally:
            cap.release()

        return totalFramesDecoded

    def processFrameToTorch(self, frame, normStream=None):
        """
        Processes a single frame with optimized memory handling.

        Args:
            frame: The frame to process as nelux frame (HWC format).
            normStream: The CUDA stream for normalization (if applicable).

        Returns:
            The processed frame as a torch tensor in BCHW format.
        """
        norm = 1 / 255.0 if frame.dtype == torch.uint8 else 1 / 65535.0
        if self.cudaEnabled:
            with torch.cuda.stream(normStream):
                frame = frame.to(
                    device="cuda",
                    non_blocking=True,
                    dtype=torch.float16 if self.half else torch.float32,
                )

                frame = frame.permute(2, 0, 1)
                frame.mul_(norm)
                frame.clamp_(0, 1)

                if self.resize and not self._didDecoderResize:
                    frame = F.interpolate(
                        frame.unsqueeze(0),
                        size=(self.height, self.width),
                        mode="bicubic",
                        align_corners=False,
                    )
                else:
                    frame = frame.unsqueeze(0)

            if normStream is not None:
                normStream.synchronize()
            return frame
        else:
            frame = frame.to(
                device="cpu",
                non_blocking=False,
                dtype=torch.float16 if self.half else torch.float32,
            )

            frame = frame.permute(2, 0, 1)

            frame.mul_(norm)
            frame.clamp_(0, 1)

            if self.resize and not self._didDecoderResize:
                frame = F.interpolate(
                    frame.unsqueeze(0),
                    size=(self.height, self.width),
                    mode="bicubic",
                    align_corners=False,
                )
            else:
                frame = frame.unsqueeze(0)

            return frame

    def read(self):
        """
        Reads a frame from the decodeBuffer.

        Returns:
            The next frame from the decodeBuffer.
        """
        return self.decodeBuffer.get()

    def isReadFinished(self) -> bool:
        """
        Returns:
            Whether the decoding process is finished.
        """
        return self.isFinished

    def isQueueEmpty(self) -> bool:
        """
        Returns:
            Whether the decoding buffer is empty.
        """

        if self.decodeBuffer.empty() and self.isFinished:
            return True
        else:
            return False


class WriteBuffer:
    def __init__(
        self,
        input: str = "",
        output: str = "",
        encode_method: str = "x264",
        custom_encoder: str = "",
        width: int = 1920,
        height: int = 1080,
        fps: float = 60.0,
        grayscale: bool = False,
        transparent: bool = False,
        benchmark: bool = False,
        bitDepth: str = "8bit",
        inpoint: float = 0.0,
        outpoint: float = 0.0,
        slowmo: bool = False,
        output_scale_width: int | None = None,
        output_scale_height: int | None = None,
        enablePreview: bool = False,
        previewSink=None,
        single_image_output: bool = False,
    ):
        """
        A class meant to Pipe the input to FFMPEG from a queue.

        output: str - The path to the output video file.
        encode_method: str - The method to use for encoding the video. Options include "x264", "x264_animation", "nvenc_h264", etc.
        custom_encoder: str - A custom encoder string to use for encoding the video.
        grayscale: bool - Whether to encode the video in grayscale.
        width: int - The width of the output video in pixels.
        height: int - The height of the output video in pixels.
        fps: float - The frames per second of the output video.
        transparent: bool - Whether to encode the video with transparency.
        audio: bool - Whether to include audio in the output video.
        benchmark: bool - Whether to benchmark the encoding process, this will not output any video.
        bitDepth: str - The bit depth of the output video. Options include "8bit" and "10bit".
        inpoint: float - The start time of the segment to encode, in seconds.
        outpoint: float - The end time of the segment to encode, in seconds.
        output_scale_width: int - The target width for output scaling (optional).
        output_scale_height: int - The target height for output scaling (optional).
        enablePreview: bool - Whether to sample preview frames for the preview server.
        previewSink: PreviewSink - In-memory holder the writer pushes preview JPEGs
            to. Preview is only active when a sink is supplied.
        """
        self.input = input
        self.output = os.path.normpath(output)
        outputDir = os.path.dirname(self.output)
        if outputDir:
            os.makedirs(outputDir, exist_ok=True)
        # A *_nelux method reaching this writer means a caller that builds
        # WriteBuffer directly instead of going through createWriteBuffer --
        # every depth backend does. matchEncoder has no arm for those names, so
        # it returned an EMPTY command: no -c:v at all, and FFmpeg quietly
        # encoded with the container default at its own CRF (23), discarding
        # the requested encoder and its quality target. The pix_fmt was never
        # affected -- getPixFMT does not consult matchEncoder. Each *_nelux
        # method mirrors an FFmpeg twin one-for-one (x264_nelux -> x264,
        # nvenc_av1_nelux -> nvenc_av1, ...), so fall back to that twin.
        #
        # animeSegment also builds WriteBuffer directly but was never affected:
        # it passes transparent=True, and getPixFMT rewrites the method to
        # prores_segment before matchEncoder sees it.
        #
        # No warning here: src/cli/validator.py:_resolveNeluxEncoder already
        # resolves this once per run, where a batch of 200 inputs gets one
        # message instead of 200. This is the last-resort net for a caller that
        # builds a writer without going through the CLI.
        if encode_method.endswith("_nelux"):
            fallback = encode_method[: -len("_nelux")]
            logging.info(
                f"{encode_method} needs the Nelux writer, which this operation "
                f"does not use; encoding with the equivalent {fallback}."
            )
            encode_method = fallback
        self.encode_method = encode_method
        self.single_image_output = single_image_output

        sequenceExt = SEQUENCE_EXTENSIONS.get(self.encode_method)
        if (
            sequenceExt is not None
            and not self.single_image_output
            and "%" not in self.output
        ):
            _, ext = os.path.splitext(self.output)
            if not ext:
                self.output = os.path.join(self.output, f"%08d{sequenceExt}")
            else:
                base, _ = os.path.splitext(self.output)
                self.output = f"{base}_%08d{sequenceExt}"

        if sequenceExt is not None and self.single_image_output:
            _, ext = os.path.splitext(self.output)
            if not ext:
                self.output = f"{self.output}{sequenceExt}"

        self.custom_encoder = custom_encoder
        self.grayscale = grayscale
        self.width = width
        self.height = height
        self.fps = fps
        self.transparent = transparent
        self.benchmark = benchmark
        self.bitDepth = bitDepth
        self.inpoint = inpoint
        self.outpoint = outpoint
        self.slowmo = slowmo
        self.output_scale_width, self.output_scale_height = _resolveOutputScale(
            output_scale_width, output_scale_height
        )
        # Preview is sampled from the frames this writer already holds, never
        # produced by the encoder: the ffmpeg command below is identical whether
        # preview is on or off, so preview cannot perturb encoding. The sampler
        # is demand-gated and wall-clock throttled -- see PreviewSampler.
        self.previewSink = previewSink
        self.enablePreview = bool(enablePreview and previewSink is not None)
        self.previewSampler = None
        if self.enablePreview:
            from src.server.previewSettings import PreviewSampler

            self.previewSampler = PreviewSampler(previewSink)

        self.writtenFrames = 0
        self.writeBuffer = Queue(maxsize=32)
        # True once __call__ consumed the producer's None sentinel. If the
        # encoder dies early this stays False and the finally-drain below keeps
        # emptying the queue so the producer's blocking put() can't deadlock
        # the whole run (e.g. prores into an .mp4 container).
        self._sawSentinel = False
        # Set when encoding fails. Read through
        # src/io/runOutcome.py:truncatedDecodeError, the same way the decoder's
        # own failure is.
        self.encodeError: Exception | None = None

    def _shouldUseDirectImageSingleFrame(self) -> bool:
        return (
            self.encode_method in SEQUENCE_EXTENSIONS
            and self.single_image_output
            and not self.custom_encoder
            and not self.benchmark
        )

    @staticmethod
    def _toOutputFrame(frame, multiplier: int, dtype):
        """CHW float frame -> contiguous HWC integer frame.

        fp16's largest finite value is 65504, so a 16-bit frame cannot be
        scaled in half precision: ``mul(65535)`` saturates to inf and
        ``clamp(0, 65535)`` raises "value cannot be converted to type
        c10::Half without overflow". ``--half`` is on by default, so
        ``--bit_depth 16bit`` hit that on the very first frame and the
        encoder's ``except Exception`` turned it into a 0-byte output.
        Promote to fp32 for 16-bit only; 8-bit keeps its existing fp16 math.
        """
        frame = frame.squeeze(0).permute(1, 2, 0)
        if multiplier > 255 and frame.dtype == torch.float16:
            frame = frame.float()
        return frame.mul(multiplier).clamp(0, multiplier).to(dtype).contiguous()

    def _writeSingleImageDirect(
        self, frameTensor, needsResize: bool, multiplier: int, dtype
    ):
        from torch.nn import functional as F

        if self.encode_method == "jpeg" and multiplier > 255:
            # JPEG is 8-bit only; a --bit_depth 16bit single-image run would
            # otherwise hand cv2/PIL a uint16 array neither can encode. Warn
            # like getPixFMT does on the sequence path.
            logWarning("JPEG only supports 8-bit encoding. Falling back to 8-bit.")
            multiplier, dtype = 255, torch.uint8

        if needsResize:
            frameTensor = F.interpolate(
                frameTensor,
                size=(self.height, self.width),
                mode="bicubic",
                align_corners=False,
            )

        frameArray = self._toOutputFrame(frameTensor, multiplier, dtype).cpu().numpy()

        if frameArray.ndim == 3 and frameArray.shape[2] == 1:
            frameArray = frameArray[:, :, 0]

        import cv2  # lazy: only needed for single-image output

        cvWriteFrame = frameArray
        if frameArray.ndim == 3 and frameArray.shape[2] == 3:
            cvWriteFrame = cv2.cvtColor(frameArray, cv2.COLOR_RGB2BGR)
        elif frameArray.ndim == 3 and frameArray.shape[2] == 4:
            cvWriteFrame = cv2.cvtColor(frameArray, cv2.COLOR_RGBA2BGRA)

        if self.encode_method == "jpeg":
            writeParams = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
        else:
            writeParams = []
        writeOk = cv2.imwrite(self.output, cvWriteFrame, writeParams)
        if not writeOk:
            from PIL import Image

            if self.encode_method == "jpeg":
                Image.fromarray(frameArray).save(
                    self.output, format="JPEG", quality=100
                )
            else:
                Image.fromarray(frameArray).save(self.output, format="PNG")

    def encodeSettings(self) -> list:
        """
        Simplified structure for setting input/output pix formats
        and building FFMPEG command.
        """
        # Set environment variables
        os.environ["FFREPORT"] = "file=FFmpeg-Log.log:level=32"
        if any(m and "av1" in m for m in (self.encode_method, self.custom_encoder)):
            os.environ["SVT_LOG"] = "0"

        self.inputPixFmt, outputPixFmt, self.encode_method = getPixFMT(
            self.encode_method, self.bitDepth, self.grayscale, self.transparent
        )

        if self.benchmark:
            return self._buildBenchmarkCommand()
        else:
            return self._buildEncodingCommand(outputPixFmt)

    def _buildBenchmarkCommand(self):
        """Build FFmpeg command for benchmarking"""
        return [
            cs.FFMPEGPATH,
            "-y",
            "-hide_banner",
            "-v",
            "warning",
            "-nostats",
            "-f",
            "rawvideo",
            "-video_size",
            f"{self.width}x{self.height}",
            "-pix_fmt",
            self.inputPixFmt,
            "-r",
            str(self.fps),
            "-i",
            "-",
            "-benchmark",
            "-f",
            "null",
            "-",
        ]

    def _isNvencEncoder(self):
        """Check if the current encode method uses NVENC"""
        nvenc_methods = [
            "nvenc_h264",
            "slow_nvenc_h264",
            "nvenc_h265",
            "slow_nvenc_h265",
            "nvenc_h265_10bit",
            "nvenc_av1",
            "slow_nvenc_av1",
            "lossless_nvenc",
            "lossless_nvenc_h264",
        ]
        return self.encode_method in nvenc_methods

    def _buildEncodingCommand(self, outputPixFmt):
        """Build FFmpeg command for encoding"""
        useHwUpload = self._isNvencEncoder() and not self.custom_encoder

        command = [
            cs.FFMPEGPATH,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostats",
            "-threads",
            "0",
            "-filter_threads",
            "0",
        ]

        # Initialize CUDA device for hwupload when using NVENC
        if useHwUpload:
            command.extend(["-init_hw_device", "cuda=cu:0", "-filter_hw_device", "cu"])

        command.extend(
            [
                "-f",
                "rawvideo",
                "-pix_fmt",
                self.inputPixFmt,
                "-s",
                f"{self.width}x{self.height}",
                "-r",
                str(self.fps),
            ]
        )

        # The piped frames are ALREADY the [inpoint, outpoint] range -- BuildBuffer
        # trimmed the decode -- so the video input takes no -itsoffset and no
        # trim. It used to take both: "-itsoffset in -i pipe:0 -ss in -to out"
        # shifted the piped video to start at `inpoint` and then trimmed it, so
        # --inpoint 1 --outpoint 6 lost a full second of video off a 5s payload
        # and --inpoint 5 --outpoint 10 produced a 261-byte streamless file,
        # both with ffmpeg exiting 0.
        command.extend(["-i", "pipe:0"])

        if cs.AUDIO:
            command.extend(["-thread_queue_size", "1024"])
            # Only the source input is trimmed, and only to feed audio/subtitles
            # the matching range. Input-side -ss rebases to 0, so the audio lines
            # up with the piped video's own zero.
            #
            # -ss is gated on inpoint alone, NOT on outpoint: BuildBuffer decodes
            # [inpoint, EOF] when only --inpoint is given, so tying the audio
            # offset to --outpoint left the audio ahead of the video by inpoint
            # seconds whenever --outpoint was omitted. NeluxWriteBuffer's
            # add_passthrough already gates them independently; this matches it.
            if not self.slowmo:
                if self.inpoint:
                    command.extend(["-ss", str(self.inpoint)])
                if self.outpoint != 0:
                    command.extend(["-to", str(self.outpoint)])
            command.extend(["-i", self.input])

        filterList = self._buildFilterList()

        command.extend(["-map", "0:v"])

        if not self.custom_encoder:
            command.extend(matchEncoder(self.encode_method))

            if useHwUpload:
                # hwupload_cuda uploads whatever SW format precedes it, so
                # pinning nv12 handed 8-bit 4:2:0 surfaces to nvenc_h265_10bit
                # -- which matchEncoder already tags "-profile:v main10" -- and
                # the "10bit" preset encoded 8-bit data. p010le is the CUDA
                # 10-bit format.
                #
                # CUDA has no yuv444p10le hwframe format, so a 4:4:4 request
                # subsamples here either way -- but it used to also fall to
                # nv12 and lose the two extra bits, which is the half that
                # actually matters (--depth --bit_depth 16bit on nvenc_h265
                # wrote an 8-bit depth map: visible banding on the smooth
                # gradients the 16-bit path exists for). Keep the depth
                # wherever the encoder can carry it. h264_nvenc cannot encode
                # 10-bit at all -- getPixFMT already forces "nvenc_h264" back
                # to 8-bit, but "lossless_nvenc" also runs on h264_nvenc and
                # does not go through that branch, so gate on the encoder.
                tenBitNvenc = self.encode_method in (
                    "nvenc_h265",
                    "slow_nvenc_h265",
                    "nvenc_h265_10bit",
                    "nvenc_av1",
                    "slow_nvenc_av1",
                )
                wantsTenBit = outputPixFmt.endswith(("10le", "12le", "16le"))
                hwFormat = "p010le" if (tenBitNvenc and wantsTenBit) else "nv12"
                hwFilters = filterList.copy() if filterList else []
                hwFilters.append(f"format={hwFormat}")
                hwFilters.append("hwupload_cuda")
                command.extend(["-vf", ",".join(hwFilters)])
            else:
                if filterList:
                    command.extend(["-vf", ",".join(filterList)])
                command.extend(["-pix_fmt", outputPixFmt])
        else:
            command.extend(self._buildCustomEncoder(filterList, outputPixFmt))

        if cs.AUDIO:
            command.extend(self._buildAudioSettings())

        if self.single_image_output:
            command.extend(["-frames:v", "1"])

        command.append(self.output)

        return command

    def _buildFilterList(self):
        """Build list of video filters based on settings"""
        filterList = []

        if self.output_scale_width and self.output_scale_height:
            filterList.append(
                f"scale={self.output_scale_width}:{self.output_scale_height}:flags=bilinear"
            )

        if self.grayscale:
            filterList.append(
                "format=gray" if self.bitDepth == "8bit" else "format=gray16be"
            )
        if self.transparent:
            # NOT yuva420p: getPixFMT already selects yuva444p10le for this
            # path and matchEncoder pairs it with prores_ks -profile:v 4
            # (4444). Routing rgba through 8-bit 2x2-subsampled yuva420p first
            # threw away half the chroma resolution and two bits of precision
            # before the 4:4:4 conversion, which cannot recover either -- the
            # colour fringing showed up along the hard matte edges --segment
            # exists to produce.
            filterList.append("format=yuva444p10le")

        if not self.grayscale and not self.transparent:
            # RGB->YUV colorspace conversion + dithering.
            #
            # BT.709 (the common path; unknown/SD sources also default here):
            # use swscale (`scale`) instead of `zscale`. zscale/zimg runs a
            # float pipeline and costs ~3x the CPU of swscale's SIMD-integer
            # path for a matrix-identical result (verified bit-for-bit on luma
            # mean, black level, and banding across every test source/pixfmt).
            # swscale only dithers on a bit-DEPTH reduction, so we route through
            # a 16-bit working format (yuv444p16le); the downstream -pix_fmt /
            # hwupload reduction then error-diffusion-dithers down to the target
            # depth, matching zscale's banding exactly. `setparams` stamps the
            # full color metadata (matrix + primaries + transfer + range) --
            # swscale alone only tags the matrix.
            #
            # BT.2020 stays on zscale: swscale has no bt2020nc matrix.
            #
            # This arm used to read `matrix=bt2020:norm=bt2020`, and neither
            # token exists: zscale's matrix constants are 2020_ncl/2020_cl
            # (aliases bt2020nc/bt2020c) and there is no `norm` option at all.
            # So the filtergraph could never parse -- FFmpeg died with
            # "Undefined constant or missing '(' in 'bt2020'" and every
            # BT.2020/HDR10 source failed the whole render for a 0-byte output.
            # setparams mirrors the bt709 arm, which already stamps the full
            # colour metadata rather than the matrix alone.
            filterList.append(colorSpaceFilter(_probedMetadata()))

        return filterList

    def _buildCustomEncoder(self, filterList, outputPixFmt):
        """Apply custom encoder settings with filters"""
        customEncoderArgs = self.custom_encoder.split()

        if "-vf" in customEncoderArgs:
            vfIndex = customEncoderArgs.index("-vf")
            filterString = customEncoderArgs[vfIndex + 1]
            for filterItem in filterList:
                filterString += f",{filterItem}"
            customEncoderArgs[vfIndex + 1] = filterString
        elif filterList:
            customEncoderArgs.extend(["-vf", ",".join(filterList)])

        if "-pix_fmt" not in customEncoderArgs:
            logging.info(f"-pix_fmt was not found, adding {outputPixFmt}.")
            customEncoderArgs.extend(["-pix_fmt", outputPixFmt])

        return customEncoderArgs

    # Subtitle codecs an ISO-BMFF container carries natively, so a stream copy
    # is both valid and lossless. Everything else text-based (subrip, ass, ssa)
    # has to become mov_text or the mux aborts at header-write with "Could not
    # find tag for codec subrip ... not currently supported in container".
    _ISO_BMFF_NATIVE_SUBTITLES = frozenset({"mov_text", "ttml", "stpp", "wvtt"})

    def _isoBmffSubtitleCodec(self) -> str:
        """`copy` or `mov_text` for an .mp4/.m4v/.mov output, by source codec.

        Forcing mov_text unconditionally breaks the containers that already
        agreed: FFmpeg ships a TTML *encoder* but no TTML decoder, so a
        TTML-in-MP4 source could not be transcoded at all and the render died
        with "no decoder found for: ttml" for a 0-byte output -- where a plain
        copy had worked. Ask the source what it has; a probe failure keeps the
        transcode, which is right for the common case (an MKV of subrip/ass).
        """
        # An image sequence needs no test of its own: the pattern is not a path
        # on disk, so it fails the existence check below. Testing for a bare
        # percent instead would divert an ordinary ``50%_off.mkv`` here and
        # force mov_text on it -- the TTML case this method exists to avoid.
        if not self.input or not os.path.exists(self.input):
            return "mov_text"
        try:
            probe = subprocess.run(
                [
                    cs.FFPROBEPATH,
                    "-v",
                    "error",
                    "-select_streams",
                    "s",
                    "-show_entries",
                    "stream=codec_name",
                    "-of",
                    "csv=p=0",
                    self.input,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except Exception as e:
            logging.warning(f"Subtitle probe failed ({e}); transcoding to mov_text.")
            return "mov_text"

        codecs = [line.strip() for line in probe.stdout.splitlines() if line.strip()]
        if codecs and all(c in self._ISO_BMFF_NATIVE_SUBTITLES for c in codecs):
            logging.info(f"Subtitles {codecs} are ISO-BMFF native; stream-copying.")
            return "copy"
        return "mov_text"

    def _buildAudioSettings(self):
        """Build audio encoding settings"""
        audioSettings = ["-map", "1:a"]

        # Lowercased: inputOutputHandler matches extensions case-insensitively
        # and copies the input's extension verbatim, so `clip.MP4` from a phone
        # or drone is an ordinary output name here.
        output = self.output.lower()

        audioCodec = "copy"
        subCodec = "copy"
        if output.endswith(".webm"):
            audioCodec = "libopus"
            subCodec = "webvtt"
        elif output.endswith((".mov", ".mp4", ".m4v")):
            subCodec = self._isoBmffSubtitleCodec()
            if output.endswith(".mov"):
                # MOV cannot stream-copy opus/vorbis (FFmpeg aborts with "opus
                # only supported in MP4"). The transparent/segment path always
                # lands here as prores in a .mov. MP4/M4V accept opus natively,
                # so they keep `copy`.
                audioCodec = "aac"
        audioSettings.extend(["-c:a", audioCodec, "-map", "1:s?", "-c:s", subCodec])

        # No -ss/-to here. These land immediately before the output file, so they
        # trimmed the OUTPUT timeline, not the audio: combined with the old
        # -itsoffset on the piped video they cut real frames off the front of the
        # render. The source input is trimmed in _buildEncodingCommand instead.
        return audioSettings

    def __call__(self):
        writtenFrames = 0

        # Wait for at least one frame to be queued before starting encoding
        while self.writeBuffer.empty():
            try:
                time.sleep(0.001)
            except KeyboardInterrupt:
                logging.warning("Encoding interrupted by user")
                return

        ffmpegProc = None
        try:
            import torch
            from torch.nn import functional as F

            initialFrame = self.writeBuffer.queue[0]
            if initialFrame is None:
                self.writeBuffer.get()
                self._sawSentinel = True
                logging.info("Encoded 0 frames")
                return

            self.channels = 1 if self.grayscale else 4 if self.transparent else 3

            isEightBit = self.bitDepth == "8bit"
            multiplier = 255 if isEightBit else 65535
            dtype = torch.uint8 if isEightBit else torch.uint16

            needsResize = (
                initialFrame.shape[2] != self.height
                or initialFrame.shape[3] != self.width
            )

            if needsResize:
                logging.info(
                    f"Frame size mismatch. Frame: {initialFrame.shape[3]}x{initialFrame.shape[2]}, Output: {self.width}x{self.height}"
                )

            if self._shouldUseDirectImageSingleFrame():
                logging.info("Using direct single-frame image encoder (ffmpeg bypass)")

                firstFrame = None
                while firstFrame is None:
                    try:
                        firstFrame = self.writeBuffer.get(timeout=1.0)
                    except Exception:
                        time.sleep(0.001)
                        continue

                if firstFrame is not None:
                    self._writeSingleImageDirect(
                        firstFrame,
                        needsResize=needsResize,
                        multiplier=multiplier,
                        dtype=dtype,
                    )
                    writtenFrames = 1

                while True:
                    try:
                        frame = self.writeBuffer.get(timeout=1.0)
                    except Exception:
                        time.sleep(0.001)
                        continue
                    if frame is None:
                        self._sawSentinel = True
                        break

                logging.info(f"Encoded {writtenFrames} frames")
                return

            command = self.encodeSettings()
            logging.info(f"Encoding with: {' '.join(map(str, command))}")

            sampler = self.previewSampler
            if sampler is not None:
                logging.info("Preview enabled, sampling frames from the writer")

            useCuda = False
            transferStream = None
            if checker.cudaAvailable:
                try:
                    transferStream = torch.cuda.Stream()
                    useCuda = True
                except Exception as e:
                    logging.warning(
                        f"CUDA init failed in writer, using CPU path. Reason: {e}"
                    )
                    useCuda = False

            ffmpegProc = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                cwd=cs.WHEREAMIRUNFROM,
            )

            if ffmpegProc.poll() is not None:
                stderr_out = (
                    ffmpegProc.stderr.read().decode(errors="replace")
                    if ffmpegProc.stderr
                    else ""
                )
                logging.error(
                    f"FFmpeg exited immediately with code {ffmpegProc.returncode}: {stderr_out}"
                )
                self.encodeError = RuntimeError(
                    f"FFmpeg exited immediately with code {ffmpegProc.returncode}: "
                    f"{stderr_out.strip()[:300]}"
                )
                return

            logging.info(f"Encoding path: {'CUDA pinned' if useCuda else 'CPU'}")

            if useCuda:
                frameShape = (self.height, self.width, self.channels)
                pinnedBuffers = [
                    torch.empty(frameShape, dtype=dtype, pin_memory=True),
                    torch.empty(frameShape, dtype=dtype, pin_memory=True),
                ]
                transferEvents = [torch.cuda.Event(), torch.cuda.Event()]
                bufferIdx = 0
                pendingBuffer = None
                pendingEvent = None

                while True:
                    try:
                        frame = self.writeBuffer.get(timeout=1.0)
                    except Exception:
                        time.sleep(0.001)
                        continue
                    if frame is None:
                        self._sawSentinel = True
                        if pendingBuffer is not None:
                            pendingEvent.synchronize()
                            ffmpegProc.stdin.write(pendingBuffer.numpy())
                            writtenFrames += 1
                        break

                    with torch.cuda.stream(transferStream):
                        if needsResize:
                            frame = F.interpolate(
                                frame,
                                size=(self.height, self.width),
                                mode="bicubic",
                                align_corners=False,
                            )

                        gpuTensor = self._toOutputFrame(frame, multiplier, dtype)

                        currentBuffer = pinnedBuffers[bufferIdx]
                        currentBuffer.copy_(gpuTensor, non_blocking=True)
                        currentEvent = transferEvents[bufferIdx]
                        currentEvent.record(transferStream)

                    if pendingBuffer is not None:
                        pendingEvent.synchronize()
                        ffmpegProc.stdin.write(pendingBuffer.numpy())
                        writtenFrames += 1
                        if sampler is not None and sampler.wants():
                            # The pinned buffer is host-side and already synced;
                            # sampling after the encoder write keeps it off the
                            # critical path. submit() copies (the buffer is
                            # reused two frames from now).
                            sampler.submit(pendingBuffer.numpy())

                    pendingBuffer = currentBuffer
                    pendingEvent = currentEvent
                    bufferIdx = 1 - bufferIdx

            else:
                while True:
                    try:
                        frame = self.writeBuffer.get(timeout=1.0)
                    except Exception:
                        time.sleep(0.001)
                        continue
                    if frame is None:
                        self._sawSentinel = True
                        break

                    if needsResize:
                        frame = F.interpolate(
                            frame,
                            size=(self.height, self.width),
                            mode="bicubic",
                            align_corners=False,
                        )
                    frameTensor = self._toOutputFrame(frame, multiplier, dtype)

                    ffmpegProc.stdin.write(frameTensor.numpy())
                    writtenFrames += 1
                    if sampler is not None and sampler.wants():
                        sampler.submit(frameTensor.numpy())

            logging.info(f"Encoded {writtenFrames} frames")

        except Exception as e:
            # Recorded, not just logged: for a single output file a dead
            # encoder shows up as 0 bytes, but an image sequence that died
            # after one frame is indistinguishable from a complete one.
            self.encodeError = e
            logging.error(f"Encoding error: {e}")
            if ffmpegProc is not None:
                rc = ffmpegProc.poll()
                logging.error(f"FFmpeg exit code: {rc}")
                if ffmpegProc.stderr:
                    try:
                        stderr_out = (
                            ffmpegProc.stderr.read().decode(errors="replace").strip()
                        )
                        if stderr_out:
                            logging.error(f"FFmpeg stderr: {stderr_out}")
                    except Exception:
                        pass
        finally:
            # Early exit (encoder startup failure / raised exception) leaves the
            # producer blocked on put() to a full queue with nobody consuming;
            # keep draining until its close() sentinel lands so the run fails
            # cleanly instead of deadlocking (segment->mp4 prores case).
            if not self._sawSentinel:
                _drainQueueUntilSentinel(self.writeBuffer)
            try:
                if ffmpegProc is not None and ffmpegProc.stderr:
                    ffmpegProc.stderr.close()
                if ffmpegProc is not None and ffmpegProc.stdin:
                    ffmpegProc.stdin.close()
                if ffmpegProc is not None:
                    # 3s was too short: when the worker died early the encoder
                    # pipe still had buffered data + an audio input file to
                    # finalize, routinely blowing past 3s and spamming a
                    # "Cleanup error" warning that looked like a bug. 30s is a
                    # safe upper bound for any real finalize; if we still hit
                    # it, force-kill and log at DEBUG (this is expected when
                    # the upstream processing already errored).
                    try:
                        ffmpegProc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        logging.debug(
                            "FFmpeg did not exit within 30s after stdin close, killing"
                        )
                        ffmpegProc.kill()
                        ffmpegProc.wait()

                if self.previewSampler is not None:
                    self.previewSampler.close()

            except Exception as e:
                logging.warning(f"Cleanup error: {e}")

    def write(self, frame):
        """
        Add a frame to the queue. Must be in [B, C, H, W] format (BCHW).
        Frame type is torch.Tensor when using PyTorch backend.
        """
        self.writeBuffer.put(frame)

    def put(self, frame):
        """
        Equivalent to write()
        Add a frame to the queue. Must be in [B, C, H, W] format (BCHW).
        Frame type is torch.Tensor when using PyTorch backend.
        """
        self.writeBuffer.put(frame)

    def close(self):
        self.writeBuffer.put(None)


class NeluxWriteBuffer:
    """
    Write buffer that uses Nelux VideoEncoder for NVENC (hardware) or
    libx264/libx265/libsvtav1/libvpx-vp9/prores_ks/gif (software) encoding.
    More efficient than FFmpeg pipe for GPU-resident frames.
    """

    def __init__(
        self,
        input: str = "",
        output: str = "",
        encode_method: str = "nvenc_h264_nelux",
        width: int = 1920,
        height: int = 1080,
        fps: float = 60.0,
        inpoint: float = 0.0,
        outpoint: float = 0.0,
        benchmark: bool = False,
        grayscale: bool = False,
        enablePreview: bool = False,
        previewSink=None,
        output_scale_width: int | None = None,
        output_scale_height: int | None = None,
        **kwargs,  # Accept and ignore other WriteBuffer params for compatibility
    ):
        """
        Initialize Nelux-based encoder.

        Args:
            input: Source video path (audio/subtitle passthrough; see _setupPassthrough).
            output: Output video path.
            encode_method: Any *_nelux method with a matchNeluxEncoder mapping
                (the NVENC families plus the libx264/libx265/libsvtav1/
                libvpx-vp9/prores_ks/gif software twins).
            width: Output width.
            height: Output height.
            fps: Output framerate.
            inpoint: Trim start in seconds (--ss); applied to passthrough streams.
            outpoint: Trim end in seconds (--to); applied to passthrough streams.
            enablePreview: Whether to sample preview frames for the preview server.
            previewSink: PreviewSink the sampler pushes preview JPEGs to.
            output_scale_width/output_scale_height: `--output_scale` target. The
                encoder is built at these dims with nelux's encoder-side
                `resize=True` (nelux >= 0.18.0), which scales incoming frames
                inside the libswscale convert pass it already runs.
        """
        self.input = input
        self.output = os.path.normpath(output)
        # Same as WriteBuffer.__init__. NOT a latent bug being fixed: nelux
        # creates the parent itself (verified on 0.17.0 -- VideoEncoder's
        # constructor makes the folder), so this is belt-and-braces at TAS's
        # own boundary rather than a dependency's undocumented courtesy. The
        # failure it guards against is the one FFmpeg reports as "Error
        # opening output <path>: No such file or directory", which kills the
        # run at the first frame with a broken pipe and a 0-byte output.
        outputDir = os.path.dirname(self.output)
        if outputDir:
            os.makedirs(outputDir, exist_ok=True)
        self.width = width
        self.height = height
        self.fps = fps
        self.inpoint = inpoint
        self.outpoint = outpoint
        self.benchmark = benchmark
        self.writeBuffer = Queue(maxsize=32)
        self.writtenFrames = 0
        self.CudaStream = None
        # Cancel handshake, driven by finalizeLiveWriters: _stopNow asks the
        # encode loop to stop now rather than drain 32 queued frames first, and
        # _finalized says the container trailer has been written. Both exist so
        # a Ctrl-C leaves a playable file instead of a headerless one.
        self._stopNow = threading.Event()
        self._finalized = threading.Event()
        # Same early-death drain contract as WriteBuffer (see its __init__).
        self._sawSentinel = False
        self.encodeError: Exception | None = None
        self.acceptsHwcUint8 = True
        # Preview is sampled from the frames handed to the encoder, so it works
        # here exactly as it does for WriteBuffer -- nelux needs no preview
        # stream of its own.
        self.previewSink = previewSink
        self.enablePreview = bool(enablePreview and previewSink is not None)
        self.previewSampler = None
        if self.enablePreview:
            from src.server.previewSettings import PreviewSampler

            self.previewSampler = PreviewSampler(previewSink)

        # The Nelux mirror of encodingSettings.matchEncoder, one mapping per
        # FFmpeg twin; matchNeluxEncoder documents the knob translation and
        # returns a fresh dict, so the mutations below stay per-instance.
        self.encoderKwargs = matchNeluxEncoder(encode_method)
        if self.encoderKwargs is None:
            # Silently encoding H.264 when the caller asked for something else
            # is how the FFmpeg writer's *_nelux gap stayed invisible for so
            # long; do not repeat it on this side.
            logWarning(
                f"'{encode_method}' has no Nelux encoder mapping. "
                "Encoding with nvenc_h264_nelux."
            )
            self.encoderKwargs = matchNeluxEncoder("nvenc_h264_nelux")
        # The Nelux mirror of the FFmpeg writer's colorSpaceFilter: convert
        # and tag bt709 (or bt2020nc for a probed BT.2020 source) instead of
        # riding swscale's default bt470bg/601 matrix, which produced a
        # correctly-tagged file that still did not match the FFmpeg twin's
        # conversion. Mapping options stay on top only where keys collide
        # (they never do today -- colour keys and quality keys are disjoint).
        colourOptions = colorSpaceOptions(_probedMetadata())
        self.encoderKwargs["options"] = {
            **colourOptions,
            **(self.encoderKwargs.get("options") or {}),
        }
        # --output_scale: build the encoder at the target dims and let nelux
        # scale each incoming pipeline-res frame inside the libswscale convert
        # pass it already runs (encoder-side resize, nelux >= 0.18.0). bilinear
        # matches the FFmpeg twin's `scale=w:h:flags=bilinear` byte-identically.
        # NVENC caveat: a CUDA frame whose size differs from the output takes
        # nelux's CPU staging path (the fused NVENC kernel converts but does
        # not scale).
        output_scale_width, output_scale_height = _resolveOutputScale(
            output_scale_width, output_scale_height
        )
        self.outputWidth = width
        self.outputHeight = height
        if output_scale_width and output_scale_height:
            self.outputWidth = output_scale_width
            self.outputHeight = output_scale_height
            self.encoderKwargs["resize"] = True
            self.encoderKwargs["resize_filter"] = "bilinear"
        self.codec = self.encoderKwargs["codec"]
        # NVENC variants expect a hardware encoder; software (libx*/libsvtav1) do not.
        self.expectHardware = self.codec.endswith("_nvenc")
        self.encoder = None

        if checker.cudaAvailable:
            self.CudaStream = torch.cuda.Stream()

        scaleNote = (
            f" -> {self.outputWidth}x{self.outputHeight} (encoder-side resize)"
            if self.encoderKwargs.get("resize")
            else ""
        )
        logging.info(
            f"NeluxWriteBuffer initialized: {width}x{height}@{fps}fps{scaleNote}, "
            f"codec={self.codec}"
        )

    def _setupPassthrough(self):
        """
        Copy/transcode audio + subtitle streams from the source into the Nelux
        output via VideoEncoder.add_passthrough, replacing the FFmpeg-subprocess
        muxing the standard WriteBuffer uses (second `-i input`, `-map 1:a -c:a
        copy`, `-map 1:s? -c:s copy`, with the webm `libopus`/`webvtt` fallback).
        The --ss/--to trim (inpoint/outpoint) is honored: streams are packet-gated
        and rebased so inpoint maps to output t=0, aligning with the first video
        frame pushed (the decode side already trims the video to the same range).

        Must run AFTER the encoder is constructed but BEFORE the first
        encode_frame: Nelux writes the container header on the first frame and
        every stream must exist by then.
        """
        encoder = self.encoder
        if encoder is None or not cs.AUDIO:
            return
        # Same as _isoBmffSubtitleCodec: an image-sequence pattern is caught by
        # the existence check, while a literal percent in a real file's name is
        # not a reason to drop its audio.
        if not self.input or not os.path.exists(self.input):
            logging.info(
                f"Nelux passthrough skipped: no usable source file ('{self.input}')."
            )
            return

        end = self.outpoint if self.outpoint and self.outpoint > 0 else None
        try:
            # allow_transcode re-encodes streams the container cannot stream-copy
            # (e.g. AAC->webm) to the container default instead of dropping them,
            # mirroring the WriteBuffer webm path. The kwarg only exists in
            # nelux >= 0.12.0; older builds fall back to copy-only.
            try:
                encoder.add_passthrough(
                    self.input,
                    audio=True,
                    subtitles=True,
                    start=float(self.inpoint),
                    end=end,
                    allow_transcode=True,
                )
            except TypeError:
                encoder.add_passthrough(
                    self.input,
                    audio=True,
                    subtitles=True,
                    start=float(self.inpoint),
                    end=end,
                )
            logging.info(
                f"Nelux passthrough: audio+subtitles from '{self.input}' "
                f"(start={self.inpoint}, end={end})"
            )
        except Exception as e:
            logging.warning(f"Nelux passthrough failed, encoding video only: {e}")

    def __call__(self):
        """Process frames from writeBuffer and encode with Nelux."""
        import torch

        sampler = self.previewSampler
        _LIVE_NELUX_WRITERS.add(self)
        try:
            if self.benchmark:
                while not self._stopNow.is_set():
                    try:
                        frame = self.writeBuffer.get(timeout=0.5)
                    except Empty:
                        continue
                    if frame is None:
                        self._sawSentinel = True
                        break
                    self.writtenFrames += 1
                logging.info(f"Nelux benchmark consumed {self.writtenFrames} frames")
                return

            # Cancelling during model init/warmup parks the writer here, before
            # the encoder even exists, so this spin has to watch the flag too.
            while self.writeBuffer.empty() and not self._stopNow.is_set():
                time.sleep(0.001)
            if self._stopNow.is_set():
                return

            self.encoder = nelux.VideoEncoder(
                self.output,
                width=self.outputWidth,
                height=self.outputHeight,
                fps=self.fps,
                # codec, preset, cq, pixel_format, options [, resize]
                **self.encoderKwargs,
            )

            if hasattr(self.encoder, "is_hardware_encoder"):
                if self.encoder.is_hardware_encoder:
                    logging.info(
                        f"Nelux hardware encoder confirmed: {self.codec} -> {self.output}"
                    )
                elif self.expectHardware:
                    logging.warning(
                        f"Nelux encoder is NOT using hardware NVENC! Codec: {self.codec}"
                    )
                else:
                    logging.info(
                        f"Nelux software encoder: {self.codec} -> {self.output}"
                    )
            else:
                logging.info(f"Nelux encoder created: {self.codec} -> {self.output}")

            # Register audio/subtitle passthrough before the first encode_frame
            # (the container header is written on that first frame).
            self._setupPassthrough()

            while True:
                if self._stopNow.is_set():
                    logging.info(
                        f"Nelux encode cancelled after {self.writtenFrames} "
                        "frames; finalizing the container."
                    )
                    break
                try:
                    frame = self.writeBuffer.get(timeout=1.0)
                except Exception:
                    time.sleep(0.001)
                    continue

                if frame is None:
                    self._sawSentinel = True
                    break

                if (
                    isinstance(frame, torch.Tensor)
                    and frame.ndim == 3
                    and frame.dtype == torch.uint8
                    and frame.shape[2] == 3
                ):
                    if not frame.is_contiguous():
                        frame = frame.contiguous()
                    self.encoder.encode_frame(frame)
                    self.writtenFrames += 1
                    if sampler is not None and sampler.wants():
                        sampler.submit(frame)
                    continue

                if self.CudaStream is not None:
                    with torch.cuda.stream(self.CudaStream):
                        frame = frame.squeeze(0).permute(1, 2, 0)
                        frame = (
                            frame.mul(255.0)
                            .clamp(0, 255)
                            .to(dtype=torch.uint8, non_blocking=True)
                        )
                    self.CudaStream.synchronize()
                else:
                    # No CUDA stream on CPU/DirectML/MPS hosts; run the same
                    # conversion directly (frame is already a CPU tensor here).
                    frame = frame.squeeze(0).permute(1, 2, 0)
                    frame = frame.mul(255.0).clamp(0, 255).to(dtype=torch.uint8)

                if not frame.is_contiguous():
                    frame = frame.contiguous()

                self.encoder.encode_frame(frame)
                self.writtenFrames += 1
                if sampler is not None and sampler.wants():
                    sampler.submit(frame)

            logging.info(f"Nelux encoded {self.writtenFrames} frames")

        except Exception as e:
            self.encodeError = e
            logging.error(f"Nelux encoding error: {e}")
            import traceback

            traceback.print_exc()
        finally:
            # Keep the producer's blocking put() from deadlocking the run if
            # the encoder died before consuming the close() sentinel. Skipped
            # on the cancel path: the producer is still alive and producing, so
            # the drain would consume faster than it idles and never see a
            # sentinel -- burning the caller's whole deadline before the
            # encoder.close() below, which is the only thing that matters here.
            # Its blocked put() does not matter when os._exit follows.
            if not self._sawSentinel and not self._stopNow.is_set():
                _drainQueueUntilSentinel(self.writeBuffer)
            if sampler is not None:
                sampler.close()
            if self.encoder is not None:
                try:
                    self.encoder.close()
                except Exception as e:
                    logging.warning(f"Error closing Nelux encoder: {e}")
            self._finalized.set()
            _LIVE_NELUX_WRITERS.discard(self)

    def requestStop(self) -> None:
        """Ask the encode loop to stop and finalize the container now.

        Cancellation only: the loop breaks without draining the 32 frames still
        queued, because the caller is about to end the process.
        """
        self._stopNow.set()

    def waitFinalized(self, timeout: float) -> bool:
        """Block until the container trailer is written, or ``timeout`` passes."""
        return self._finalized.wait(timeout)

    def write(self, frame):
        """Add a frame to the queue. Must be in [H, W, 3] HWC format."""
        self.writeBuffer.put(frame)

    def put(self, frame):
        """Equivalent to write(). Add a frame to the queue."""
        self.writeBuffer.put(frame)

    def close(self):
        """Signal end of encoding."""
        self.writeBuffer.put(None)


def isNeluxEncoder(encode_method: str) -> bool:
    """Check if the encode method routes to the Nelux writer."""
    return encode_method.endswith("_nelux")


# The MOV muxer answers avformat_query_codec "yes" for these but then rejects
# the stream copy at header write ("opus only supported in MP4" -- both need
# strict=experimental), so nelux's allow_transcode never kicks in and the
# FAILURE poisons the whole encode: the header is written on the first frame,
# every encode_frame after it errors, and the run ends with a 0-byte .mov.
# The FFmpeg writer's _buildAudioSettings transcodes these to AAC for .mov.
_MOV_UNCOPYABLE_AUDIO = ("opus", "vorbis")


def _neluxMovAudioBlocker(input: str, output: str) -> str | None:
    """The source's audio codec when a .mov output cannot carry it through
    nelux's passthrough, else None.

    Probed with nelux.probe (metadata-only open, no subprocess); any probe
    failure returns None so an unreadable source degrades to the behaviour it
    had anyway (passthrough skips a source it cannot open).
    """
    if not str(output).lower().endswith(".mov") or not cs.AUDIO:
        return None
    if not input or not os.path.exists(input):
        return None
    try:
        audioCodec = str(nelux.probe(input).get("audio_codec", "") or "").lower()
    except Exception as e:
        logging.warning(f"nelux audio probe failed for '{input}': {e}")
        return None
    return audioCodec if audioCodec in _MOV_UNCOPYABLE_AUDIO else None


def createWriteBuffer(encode_method: str, **kwargs):
    """
    Factory function to create the appropriate write buffer.

    A `*_nelux` method only reaches here when it can actually be honored:
    NeluxWriteBuffer swallows `--custom_encoder` and `--bit_depth` through
    `**kwargs` without reading them, so `src/cli/validator.py:_resolveNeluxEncoder`
    swaps the method for its FFmpeg twin, once per run, before any writer is
    built. (`--output_scale` is honored natively via nelux's encoder-side
    resize since nelux 0.18.0; the validator only downgrades it on an older
    installed nelux.)

    One downgrade lives here instead: a .mov output whose source carries
    opus/vorbis audio runs the FFmpeg twin, because that combination breaks
    nelux's passthrough at header write (see _neluxMovAudioBlocker) and it
    depends on the individual input, which the once-per-run validator cannot
    see in a batch.

    Args:
        encode_method: The encoding method string.
        **kwargs: Arguments passed to the buffer constructor.

    Returns:
        WriteBuffer or NeluxWriteBuffer instance.

    Usage:
        buffer = createWriteBuffer(
            encode_method=args.encode_method,
            input=args.input,
            output=args.output,
            width=width,
            height=height,
            fps=fps,
            ...
        )
    """
    if isNeluxEncoder(encode_method):
        # Per-file, not in the validator: whether the audio survives a .mov
        # depends on THIS input's audio codec, and a batch mixes sources.
        blockedAudio = _neluxMovAudioBlocker(
            kwargs.get("input", ""), kwargs.get("output", "")
        )
        if blockedAudio:
            twin = encode_method[: -len("_nelux")]
            logWarning(
                f"{encode_method} cannot carry this source's {blockedAudio} "
                f"audio into a .mov (the muxer rejects the copy at header "
                f"write). Encoding with the equivalent {twin}, which "
                "transcodes the audio to AAC, so the output keeps its sound."
            )
            return WriteBuffer(encode_method=twin, **kwargs)
        logging.info(f"Using NeluxWriteBuffer for {encode_method}")
        return NeluxWriteBuffer(encode_method=encode_method, **kwargs)
    else:
        return WriteBuffer(encode_method=encode_method, **kwargs)
