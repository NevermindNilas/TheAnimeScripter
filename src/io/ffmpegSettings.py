import logging
import os
import threading
import time
import weakref
from contextlib import nullcontext
from queue import Empty, Queue

import torch  # this has to be always before nelux!
from torch.nn import functional as F

# Every nelux wheel (>= 0.17.0) carries its own FFmpeg libraries, so nothing
# has to be on the library search path first.
import nelux  # noqa: I001

import src.constants as cs
from src.io.encodingSettings import colorSpaceOptions, matchNeluxEncoder
from src.io.getVideoMetadata import (
    isFramePoint,
    isTrimUnset,
    trimFrameRange,
    trimPointToSeconds,
)
from src.io.inputOutputHandler import SEQUENCE_EXTENSIONS
from src.infra.isCudaInit import CudaChecker
from src.infra.logAndPrint import logWarning

checker = CudaChecker()

# Debugging flag
neluxLog = False

# I/O queue-depth tuning. 32 slots at or below 1080p; above that the depth
# scales down with frame area (floor 8) so 4K+ frames do not pin 4x the VRAM
# for ~0% throughput gain -- the writer is the backpressure past that point.
_BASE_QUEUE_DEPTH = 32
_MIN_QUEUE_DEPTH = 8
_REFERENCE_PIXELS = 1920 * 1080


def adaptiveQueueDepth(width=None, height=None) -> int:
    """Queue slots for a frame of ``width``x``height``.

    Returns ``_BASE_QUEUE_DEPTH`` at or below 1080p and scales down with
    frame area above it, never below ``_MIN_QUEUE_DEPTH``. Unknown or
    non-positive dims fall back to the base depth.
    """
    try:
        w = int(width) if width is not None else 0
        h = int(height) if height is not None else 0
    except TypeError, ValueError:
        return _BASE_QUEUE_DEPTH
    if w <= 0 or h <= 0:
        return _BASE_QUEUE_DEPTH
    pixels = w * h
    if pixels <= _REFERENCE_PIXELS:
        return _BASE_QUEUE_DEPTH
    return max(_MIN_QUEUE_DEPTH, round(_BASE_QUEUE_DEPTH * _REFERENCE_PIXELS / pixels))


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


# NeluxWriteBuffers currently running, which finalize their container
# in-process.
_LIVE_NELUX_WRITERS = weakref.WeakSet()


def finalizeLiveWriters(timeout: float = 5.0) -> bool:
    """Let every in-flight in-process encoder write its container trailer.

    ``NeluxWriteBuffer`` encodes in this process and finalizes the container
    only in its worker thread's ``finally``. ``main.py``'s KeyboardInterrupt
    handler ends the process with ``os._exit(130)``, which runs no ``finally``
    on any thread, so a cancelled render left a multi-hundred-MB file with no
    moov atom that no player or tool can open.

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


def _imageOutputPath(output: str, encodeMethod: str, singleImageOutput: bool) -> str:
    """Where a png/jpeg method writes.

    A sequence needs a numbered pattern (image2 and nelux both count from 1):
    a folder gets ``%08d.png`` inside it, a file name grows ``_%08d`` before
    its extension, and a path that already carries a pattern is kept. A
    single-image run writes one file and only gains the extension. Any other
    method keeps ``output`` as given.
    """
    sequenceExt = SEQUENCE_EXTENSIONS.get(encodeMethod)
    if sequenceExt is None:
        return output
    base, ext = os.path.splitext(output)
    if singleImageOutput:
        return output if ext else f"{output}{sequenceExt}"
    if "%" in output:
        return output
    if not ext:
        return os.path.join(output, f"%08d{sequenceExt}")
    return f"{base}_%08d{sequenceExt}"


def _probedMetadata() -> dict:
    """The current input's probed metadata dict, or {} when unavailable.

    getVideoMetadata.saveMetadata records it in-process and also writes the
    file, once per input in a batch, before any writer is built.
    NeluxWriteBuffer reads it for its colour-space decision (colorSpaceOptions).

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

    The reader blocks on ``put()`` into the bounded decode queue, so a consumer
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


# nelux 0.20.0's CPU decoder does not put a >8-bit source's nominal white at
# 65535 in the uint16 frames it returns, and its encoder expects the same
# limited-range white back (NeluxWriteBuffer._uint16White). Black is exact, so it is a pure gain:
# limited range lands on 65283 at every bit depth, and full range is
# bit-shifted rather than rescaled ((2**n - 1) << (16 - n), 65472 at 10-bit).
# NVDEC returns 65535. Dividing by 65535 dimmed every CPU-decoded HDR frame by
# 0.4% -- hidden on 16-bit output, where swscale's encode ran 0.4% bright.
_NELUX_CPU_LIMITED_WHITE = 65283.0

# 8-bit YUV formats into which nelux (>= 0.21.0) narrows a float [0, 1] RGB
# frame itself, by the writer's own rule: (x * 255).clamp(0, 255).to(uint8),
# in its CUDA kernels or CPU row loader, reading a CHW view through its
# strides. Deeper formats scale float to 0-65535 instead (the writer wants
# its own uint16 white there), and gray frames round, so those keep the
# writer's conversion.
_NELUX_FLOAT_NARROWED_FORMATS = frozenset(
    {"yuv420p", "yuvj420p", "nv12", "yuv422p", "yuvj422p", "yuv444p", "yuvj444p"}
)


def neluxUint16White(decodeMethod: str, bitDepth: int, colorRange: str) -> float:
    """The uint16 value nelux decodes a source's nominal white to."""
    if decodeMethod.lower() != "cpu" or not 8 < bitDepth <= 16:
        return 65535.0
    if colorRange in ("pc", "jpeg"):
        return float(((1 << bitDepth) - 1) << (16 - bitDepth))
    return _NELUX_CPU_LIMITED_WHITE


class BuildBuffer:
    def __init__(
        self,
        videoInput: str = "",
        inpoint: float | str = 0,
        outpoint: float | str = 0,
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
            inpoint (float | str): Start of the segment to decode, in seconds
                by default or as frames with an 'f' suffix (e.g. "100f").
            outpoint (float | str): End of the segment to decode, in seconds
                by default or as frames with an 'f' suffix. 0 means EOF.
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
        self.decodeBuffer = Queue(maxsize=adaptiveQueueDepth(width, height))
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
        # Set per reader in _decodeWithnelux; see neluxUint16White.
        self._uint16Norm = 1 / 65535.0

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
        # Read after any reconfigure(): a batch input can change depth/range.
        self._uint16Norm = 1 / neluxUint16White(
            decodeMethod,
            CachedReader.bit_depth,
            CachedReader.properties.get("color_range", "unknown"),
        )

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
        useFrameIndex = isFramePoint(self.inpoint) or isFramePoint(self.outpoint)

        if not useFrameIndex:
            startTime = float(self.inpoint or 0)
            endTime = float(self.outpoint or 0)

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

        # Frame-indexed trim: seek by frame number and count, so "<n>f"
        # stays exact instead of rounding through seconds.
        try:
            ocvFps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        except Exception:
            ocvFps = 0.0
        startFrame, endFrame = trimFrameRange(ocvFps, self.inpoint, self.outpoint)

        try:
            if startFrame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, startFrame)

            frameIdx = startFrame
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if endFrame is not None and frameIdx >= endFrame:
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
                frameIdx += 1

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
        norm = 1 / 255.0 if frame.dtype == torch.uint8 else self._uint16Norm
        if self.cudaEnabled:
            with torch.cuda.stream(normStream):
                frame = frame.to(
                    device="cuda",
                    non_blocking=True,
                    dtype=torch.float16 if self.half else torch.float32,
                )

                frame = frame.permute(2, 0, 1)
                # Out-of-place: permute() is a view and .to() above returns its
                # input unchanged when dtype/device already match, so in-place
                # math here would corrupt the caller's buffer. No perf claim --
                # the copy is the safety.
                frame = frame.mul(norm).clamp(0, 1)

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

            # Same out-of-place contract as the CUDA path above.
            frame = frame.mul(norm).clamp(0, 1)

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


class SingleImageWriteBuffer:
    """Writes the one frame of a single-image run (png/jpeg) with OpenCV.

    The png/jpeg encoders write a numbered sequence; a single-image input gets
    one file instead (`_imageOutputPath`), saved the way it always has been:
    cv2 (PIL as the fallback), PNG lossless, JPEG at quality 100. Frames past
    the first are consumed and dropped.
    """

    def __init__(
        self,
        output: str = "",
        encode_method: str = "png",
        width: int = 1920,
        height: int = 1080,
        bitDepth: str = "8bit",
        **kwargs,  # the rest of createWriteBuffer's arguments do not apply
    ):
        self.encode_method = encode_method
        self.output = _imageOutputPath(
            os.path.normpath(output), self.encode_method, True
        )
        outputDir = os.path.dirname(self.output)
        if outputDir:
            os.makedirs(outputDir, exist_ok=True)
        self.width = width
        self.height = height
        self.bitDepth = bitDepth
        self.writtenFrames = 0
        self.writeBuffer = Queue(maxsize=adaptiveQueueDepth(width, height))
        self.enablePreview = False
        self.previewSampler = None
        self._sawSentinel = False
        # Read through src/io/runOutcome.py:truncatedDecodeError.
        self.encodeError: Exception | None = None

    @staticmethod
    def _toOutputFrame(frame, multiplier: int, dtype):
        """CHW float frame -> contiguous HWC integer frame.

        fp16's largest finite value is 65504, so a 16-bit frame cannot be
        scaled in half precision: ``mul(65535)`` saturates to inf and
        ``clamp(0, 65535)`` raises "value cannot be converted to type
        c10::Half without overflow". Promote to fp32 for 16-bit only.
        """
        frame = frame.squeeze(0).permute(1, 2, 0)
        if multiplier > 255 and frame.dtype == torch.float16:
            frame = frame.float()
        return frame.mul(multiplier).clamp(0, multiplier).to(dtype).contiguous()

    def _writeImage(self, frameTensor):
        multiplier, dtype = (
            (255, torch.uint8) if self.bitDepth == "8bit" else (65535, torch.uint16)
        )
        if self.encode_method == "jpeg" and multiplier > 255:
            # JPEG is 8-bit only; cv2/PIL cannot encode a uint16 array as one.
            logWarning("JPEG only supports 8-bit encoding. Falling back to 8-bit.")
            multiplier, dtype = 255, torch.uint8

        if tuple(frameTensor.shape[-2:]) != (self.height, self.width):
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

    def __call__(self):
        try:
            frame = self.writeBuffer.get()
            if frame is None:
                self._sawSentinel = True
                logging.info("Encoded 0 frames")
                return
            self._writeImage(frame)
            self.writtenFrames = 1
            logging.info(f"Wrote single image {self.output}")
        except Exception as e:
            self.encodeError = e
            logging.error(f"Single image write error: {e}")
        finally:
            if not self._sawSentinel:
                _drainQueueUntilSentinel(self.writeBuffer)

    def write(self, frame):
        """Add a frame to the queue, [1, C, H, W] float in [0, 1]."""
        self.writeBuffer.put(frame)

    def put(self, frame):
        """Equivalent to write()."""
        self.writeBuffer.put(frame)

    def close(self):
        self.writeBuffer.put(None)


class NeluxWriteBuffer:
    """
    The video writer: encodes in-process with nelux's VideoEncoder, on NVENC/
    QSV/AMF (hardware) or libx264/libx265/libsvtav1/libvpx-vp9/prores_ks/gif/
    png/mjpeg (software), and muxes the source's audio and subtitles.
    """

    def __init__(
        self,
        input: str = "",
        output: str = "",
        encode_method: str = "x264",
        width: int = 1920,
        height: int = 1080,
        fps: float = 60.0,
        inpoint: float | str = 0,
        outpoint: float | str = 0,
        benchmark: bool = False,
        grayscale: bool = False,
        transparent: bool = False,
        bitDepth: str = "8bit",
        enablePreview: bool = False,
        previewSink=None,
        sourceFps: float | None = None,
        **kwargs,  # createWriteBuffer arguments this writer has no use for
    ):
        """
        Initialize Nelux-based encoder.

        Args:
            input: Source video path (audio/subtitle passthrough; see _setupPassthrough).
            output: Output video path.
            encode_method: An --encode_method; see matchNeluxEncoder.
            width: Output width.
            height: Output height.
            fps: Output framerate.
            inpoint: Trim start in seconds, or frames as "<n>f";
                applied to passthrough streams.
            outpoint: Trim end in seconds, or frames as "<n>f";
                applied to passthrough streams. 0 means EOF.
            sourceFps: Source fps used to resolve "<n>f" points; defaults to
                fps (correct for standalone drivers; main.py passes source).
            enablePreview: Whether to sample preview frames for the preview server.
            previewSink: PreviewSink the sampler pushes preview JPEGs to.
            grayscale: Frames are [1, 1, H, W] (depth maps); they reach nelux
                as [H, W, 1] in the gray pipe's pixel_format.
            transparent: Frames are [1, 4, H, W] RGBA (segment mattes); the
                method becomes prores_segment.
            bitDepth: "8bit" frames go out as uint8; "16bit" frames go out as
                uint16 at full precision (nelux >= 0.17.0) into a 10-bit
                pixel_format -- see matchNeluxEncoder's promotion table.
        """
        self.input = input
        # png/jpeg write a numbered sequence; every other method keeps
        # `output`. A single image never gets here (createWriteBuffer hands it
        # to SingleImageWriteBuffer).
        self.isImageOutput = encode_method in SEQUENCE_EXTENSIONS
        self.output = _imageOutputPath(os.path.normpath(output), encode_method, False)
        # nelux creates the parent itself (verified on 0.17.0 -- VideoEncoder's
        # constructor makes the folder), so this is belt-and-braces at TAS's
        # own boundary rather than a dependency's undocumented courtesy.
        outputDir = os.path.dirname(self.output)
        if outputDir:
            os.makedirs(outputDir, exist_ok=True)
        self.width = width
        self.height = height
        self.fps = fps
        self.inpoint = inpoint
        self.outpoint = outpoint
        self.sourceFps = sourceFps if sourceFps else fps
        self.benchmark = benchmark
        self.writeBuffer = Queue(maxsize=adaptiveQueueDepth(width, height))
        self.writtenFrames = 0
        self.CudaStream = None
        # Cancel handshake, driven by finalizeLiveWriters: _stopNow asks the
        # encode loop to stop now rather than drain queued frames first, and
        # _finalized says the container trailer has been written. Both exist so
        # a Ctrl-C leaves a playable file instead of a headerless one.
        self._stopNow = threading.Event()
        self._finalized = threading.Event()
        # True once __call__ consumed the producer's None sentinel. If the
        # encoder dies early this stays False and the finally-drain keeps
        # emptying the queue so the producer's blocking put() can't deadlock
        # the whole run.
        self._sawSentinel = False
        # Set when encoding fails. Read through
        # src/io/runOutcome.py:truncatedDecodeError, the same way the decoder's
        # own failure is.
        self.encodeError: Exception | None = None
        self.acceptsHwcUint8 = True
        # Preview is sampled from the frames handed to the encoder (demand-gated
        # and wall-clock throttled, see PreviewSampler), so it cannot perturb
        # the encode -- nelux needs no preview stream of its own.
        self.previewSink = previewSink
        self.enablePreview = bool(enablePreview and previewSink is not None)
        self.previewSampler = None
        if self.enablePreview:
            from src.server.previewSettings import PreviewSampler

            self.previewSampler = PreviewSampler(previewSink)

        # matchNeluxEncoder documents the knob translation and returns a fresh
        # dict, so the mutations below stay per-instance. bitDepth promotes the
        # pixel_format to 10-bit for --bit_depth 16bit (nelux >= 0.17.0 keeps
        # uint16 frames at full precision), and grayscale/transparent pick the
        # gray and alpha formats.
        self.bitDepth = bitDepth
        self.grayscale = grayscale
        self.transparent = transparent
        self.encoderKwargs = matchNeluxEncoder(
            encode_method, bitDepth, grayscale, transparent
        )
        if self.encoderKwargs is None:
            # Only a direct caller can get here (the CLI accepts mapped names
            # only); say so rather than silently encode something else.
            logWarning(f"'{encode_method}' has no encoder mapping. Encoding with x264.")
            self.encoderKwargs = matchNeluxEncoder(
                "x264", bitDepth, grayscale, transparent
            )
        # Convert and tag bt709 (or bt2020nc for a probed BT.2020 source)
        # instead of riding swscale's default bt470bg/601 matrix. Mapping
        # options stay on top only where keys collide (they never do today --
        # colour keys and quality keys are disjoint).
        #
        # Not for gray or alpha frames, which never had a colour conversion:
        # their pixels go through swscale's default matrix (bt709 options put
        # segment output 0.1 8-bit units further from the source frames).
        colourOptions = (
            {} if grayscale or transparent else colorSpaceOptions(_probedMetadata())
        )
        self.encoderKwargs["options"] = {
            **colourOptions,
            **(self.encoderKwargs.get("options") or {}),
        }
        # nelux's encoder takes uint16 RGB white to be the same 65283 its CPU
        # decoder produces (libswscale and its NVENC CUDA kernel alike), so a
        # frame scaled by 65535 came out 0.4% bright (10-bit white 943). Gray,
        # alpha and RGB destinations get no YUV range conversion; alpha must
        # not be scaled at all.
        yuvOutput = str(self.encoderKwargs.get("pixel_format", "")).startswith(
            ("yuv", "nv", "p0")
        )
        self._uint16White = (
            _NELUX_CPU_LIMITED_WHITE if colourOptions and yuvOutput else 65535.0
        )
        # Float RGB frames go to nelux as they are (a CHW view) where it
        # narrows them as this writer would; see _NELUX_FLOAT_NARROWED_FORMATS.
        self._floatToNelux = (
            bitDepth != "16bit"
            and not grayscale
            and not transparent
            and self.encoderKwargs.get("pixel_format") in _NELUX_FLOAT_NARROWED_FORMATS
        )
        self.outputWidth = width
        self.outputHeight = height
        self.codec = self.encoderKwargs["codec"]
        # NVENC variants expect a hardware encoder; software (libx*/libsvtav1) do not.
        self.expectHardware = self.codec.endswith("_nvenc")
        self.encoder = None

        if checker.cudaAvailable:
            self.CudaStream = torch.cuda.Stream()

        logging.info(
            f"NeluxWriteBuffer initialized: {width}x{height}@{fps}fps, "
            f"codec={self.codec}, "
            f"pix_fmt={self.encoderKwargs.get('pixel_format', 'default')}, "
            f"bitDepth={self.bitDepth}"
        )

    def _setupPassthrough(self):
        """
        Copy/transcode audio + subtitle streams from the source into the Nelux
        output via VideoEncoder.add_passthrough: stream copy where the
        container takes the codec, else a transcode to its default (opus into
        .mov -> aac, subrip into .mp4 -> mov_text, into .webm -> opus/webvtt).
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
        # An image sequence (or single image) has nowhere to put audio or
        # subtitles; the validator turns cs.AUDIO off for these methods, and
        # a standalone driver that never ran it must not depend on that.
        if self.isImageOutput:
            logging.info("Nelux passthrough skipped: image output has no streams.")
            return
        # Same as _isoBmffSubtitleCodec: an image-sequence pattern is caught by
        # the existence check, while a literal percent in a real file's name is
        # not a reason to drop its audio.
        if not self.input or not os.path.exists(self.input):
            logging.info(
                f"Nelux passthrough skipped: no usable source file ('{self.input}')."
            )
            return

        end = (
            trimPointToSeconds(self.sourceFps, self.outpoint)
            if not isTrimUnset(self.outpoint)
            else None
        )
        start = trimPointToSeconds(self.sourceFps, self.inpoint)
        try:
            # allow_transcode re-encodes streams the container cannot stream-copy
            # (e.g. AAC->webm) to the container default instead of dropping
            # them. The kwarg only exists in nelux >= 0.12.0; older builds fall
            # back to copy-only.
            try:
                encoder.add_passthrough(
                    self.input,
                    audio=True,
                    subtitles=True,
                    start=start,
                    end=end,
                    allow_transcode=True,
                )
            except TypeError:
                encoder.add_passthrough(
                    self.input,
                    audio=True,
                    subtitles=True,
                    start=start,
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
        from torch.nn import functional as F

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

                # nelux reads CPU and CUDA tensors only: any other device (the
                # MPS restore backend hands its output over as-is) reached
                # encode_frame as a device buffer nelux memcpy'd from as if it
                # were host memory, so everything else is moved to the CPU.
                if isinstance(frame, torch.Tensor) and frame.device.type not in (
                    "cpu",
                    "cuda",
                ):
                    frame = frame.cpu()

                # Fast path: a frame already in HWC integer form skips the
                # float-pipeline conversion below. Both dtypes ride straight
                # through: nelux carries uint16 at full precision into a
                # >8-bit destination, narrows it into an 8-bit one, and
                # upconverts uint8 into a 10-bit one. This also keeps the
                # standalone drivers working: they push HWC uint8 (their
                # pixels genuinely are 8-bit) even on a --bit_depth 16bit
                # run, which then encodes as 8-bit data in a 10-bit container
                # rather than scrambling through the BCHW branch below.
                if (
                    isinstance(frame, torch.Tensor)
                    and frame.ndim == 3
                    and (frame.dtype == torch.uint8 or frame.dtype == torch.uint16)
                    and frame.shape[2] == 3
                ):
                    # nelux (>= 0.21.0) reads a strided frame (CHW view,
                    # padded rows) where it lies, CPU or CUDA: no copy here.
                    self.encoder.encode_frame(frame)
                    self.writtenFrames += 1
                    if sampler is not None and sampler.wants():
                        sampler.submit(frame)
                    continue

                # BCHW float pipeline frames in [0, 1]: scale to the integer
                # range the encoder mapping expects (16-bit promotes fp16 to
                # fp32 first: fp16 tops out at 65504, so mul(65535) saturates
                # to inf and the clamp raises).
                multiplier = self._uint16White if self.bitDepth == "16bit" else 255
                outDtype = torch.uint16 if self.bitDepth == "16bit" else torch.uint8
                # Bicubic fallback for a frame that is not at the pipeline
                # size; depth backends lean on it.
                needsResize = tuple(frame.shape[-2:]) != (self.height, self.width)
                onStream = self.CudaStream is not None and frame.is_cuda
                # On a CUDA host everything below, nelux's own convert
                # included, is queued on this writer's stream: nothing waits
                # for the GPU here, and the convert does not queue behind the
                # inference on the default stream. CPU/DirectML/MPS hosts have
                # no stream and run the same steps directly.
                with torch.cuda.stream(self.CudaStream) if onStream else nullcontext():
                    converted = frame
                    if needsResize:
                        converted = F.interpolate(
                            converted,
                            size=(self.height, self.width),
                            mode="bicubic",
                            align_corners=False,
                        )
                    converted = converted.squeeze(0).permute(1, 2, 0)
                    if self._floatToNelux and converted.is_floating_point():
                        # nelux narrows it as the branch below would, reading
                        # the CHW view in place (fp16 is widened to fp32 before
                        # the multiply, so an fp16 frame can land 1 level away
                        # from the fp16 multiply below).
                        out = converted
                    else:
                        if multiplier > 255 and converted.dtype == torch.float16:
                            converted = converted.float()
                        out = (
                            converted.mul(multiplier)
                            .clamp(0, multiplier)
                            .to(dtype=outDtype, non_blocking=True)
                        )
                        # Gray and RGBA destinations take a packed frame; nelux
                        # reads an RGB CHW view where it lies.
                        if out.shape[-1] != 3 and not out.is_contiguous():
                            out = out.contiguous()
                    if onStream:
                        # This stream reads the producer's frame (the resize,
                        # the conversion, or nelux's convert of the view). It
                        # was made on the producer's stream, so without this,
                        # dropping it hands its block straight back to that
                        # stream's pool while the read is still queued, and the
                        # next frame is written over it mid-read (a partly
                        # stale frame every few hundred on 1080p NVENC).
                        frame.record_stream(self.CudaStream)
                    self.encoder.encode_frame(out)
                    self.writtenFrames += 1
                    if sampler is not None and sampler.wants():
                        if out.dtype not in (torch.uint8, torch.uint16):
                            out = (out.float() * 255).clamp(0, 255).to(torch.uint8)
                        sampler.submit(out)

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

        Cancellation only: the loop breaks without draining the frames still
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


def createWriteBuffer(encode_method: str, **kwargs):
    """
    Factory function to create the appropriate write buffer.

    Every writer in src/ is built here: NeluxWriteBuffer for every video and
    image sequence, honouring `--bit_depth 16bit` via matchNeluxEncoder's
    10-bit pixel_format promotion and the grayscale (depth) and transparent
    (segment) writers via its grayscale/transparent arms; SingleImageWriteBuffer
    for the one still of a single-image run.

    Args:
        encode_method: The encoding method string.
        **kwargs: Arguments passed to the buffer constructor.

    Returns:
        NeluxWriteBuffer or SingleImageWriteBuffer instance.

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
    if encode_method in SEQUENCE_EXTENSIONS and kwargs.get("single_image_output"):
        # A single still is written with cv2 (quality 100, 4:2:0, no colour
        # chunks); the nelux sequence mapping would give a different JPEG.
        return SingleImageWriteBuffer(encode_method=encode_method, **kwargs)
    return NeluxWriteBuffer(encode_method=encode_method, **kwargs)
