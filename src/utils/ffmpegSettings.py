import logging
import subprocess
import os
import src.constants as cs
import time
import celux
import threading

from queue import Queue
from src.utils.encodingSettings import matchEncoder, getPixFMT
from .isCudaInit import CudaChecker
 

# Lazy imports for heavy dependencies
# torch and torch.nn.functional are imported only when needed


checker = CudaChecker()


# Global cache for VideoReader to support reconfigure() across batch processing
_CACHED_READER = None


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

        """
        self.decodeMethod = decode_method
        self.half = half
        self.decodeBuffer = Queue(maxsize=64)
        self.useOpenCV = False
        self.width = width
        self.height = height
        self.resize = resize
        self.isFinished = False
        self._frameAvailable = threading.Event()
        self.bitDepth = bitDepth
        self.videoInput = os.path.normpath(videoInput)
        self.toTorch = toTorch
        self.inpoint = inpoint
        self.outpoint = outpoint

        if "%" not in videoInput and not os.path.exists(videoInput):
            raise FileNotFoundError(f"Video file not found: {videoInput}")

        self.cudaEnabled = False
        if checker.cudaAvailable and toTorch:
            try:
                import torch

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
        global _CACHED_READER

        try:
            # Try to reuse cached reader
            if _CACHED_READER is not None:
                try:
                    logging.info(
                        f"Reconfiguring cached VideoReader for {self.videoInput}"
                    )
                    _CACHED_READER.reconfigure(self.videoInput)
                except Exception as e:
                    logging.warning(
                        f"Failed to reconfigure VideoReader: {e}. Creating new instance."
                    )
                    _CACHED_READER = None

            if _CACHED_READER is None:
                logging.info(f"Initializing new VideoReader for {self.videoInput}")
                _CACHED_READER = celux.VideoReader(
                    self.videoInput,
                    decode_accelerator=self.decodeMethod,
                    backend=self.backend,
                )

            # Apply range slicing if needed
            if self.inpoint > 0 or self.outpoint > 0:
                reader = _CACHED_READER([float(self.inpoint), float(self.outpoint)])
            else:
                reader = _CACHED_READER
            for frame in reader:
                if self.toTorch:
                    frame = self.processFrameToTorch(
                        frame, self.normStream if self.cudaEnabled else None
                    )
                else:
                    pass

                self.decodeBuffer.put(frame)
                self._frameAvailable.set()
                decodedFrames += 1

        except Exception as e:
            logging.error(f"Celux decoding error: {e}")

            logging.info("Attempting fallback to TorchCodec...")
            try:
                decodedFrames += self.decodeWithTorchCodec()
            except Exception as fallback_e:
                logging.error(f"TorchCodec fallback failed: {fallback_e}")

        finally:
            self.decodeBuffer.put(None)
            self._frameAvailable.set()

            self.isFinished = True
            logging.info(f"Decoded {decodedFrames} frames")

    def decodeWithTorchCodec(self):
        """
        Helper method to decode using TorchCodec when Celux fails.
        Returns the number of frames decoded.
        """
        logging.info(f"Initializing TorchCodec VideoDecoder for {self.videoInput}")
        device = "cuda" if self.cudaEnabled else "cpu"

        try:
            from torchcodec.decoders import VideoDecoder as TorchCodecDecoder
            decoder = TorchCodecDecoder(self.videoInput, device=device)
        except Exception as e:
            logging.error(f"Failed to create TorchCodec decoder: {e}")
            raise

        totalFramesDecoded = 0

        try:
            startTime = float(self.inpoint)
            endTime = float(self.outpoint)
            totalFrames = len(decoder)

            if endTime > 0:
                logging.info(
                    "TorchCodec: Decoding frames by PTS in range "
                    f"[{startTime}, {endTime})"
                )
            else:
                logging.info(
                    "TorchCodec: Decoding frames by PTS starting at "
                    f"{startTime}"
                )

            chunkSize = 256
            stopDecoding = False

            for chunkStart in range(0, totalFrames, chunkSize):
                chunkEnd = min(chunkStart + chunkSize, totalFrames)
                indices = list(range(chunkStart, chunkEnd))
                frameBatch = decoder.get_frames_at(indices=indices)

                for idx, pts in enumerate(frameBatch.pts_seconds):
                    ptsSeconds = float(pts.item())

                    if ptsSeconds < startTime:
                        continue
                    if endTime > 0 and ptsSeconds >= endTime:
                        stopDecoding = True
                        break

                    frame = frameBatch.data[idx]

                    if self.toTorch:
                        frame = self.processFrameToTorch(
                            frame,
                            self.normStream if self.cudaEnabled else None,
                            channels_first=True,
                        )
                    else:
                        if self.cudaEnabled:
                            frame = frame.cpu()
                        frame = frame.permute(1, 2, 0).numpy()

                    self.decodeBuffer.put(frame)
                    self._frameAvailable.set()
                    totalFramesDecoded += 1

                if stopDecoding:
                    break

        except Exception as e:
            logging.error(f"Error during TorchCodec decoding loop: {e}")
            raise

        return totalFramesDecoded

    def processFrameToTorch(self, frame, normStream=None, channels_first=False):
        """
        Processes a single frame with optimized memory handling.

        Args:
            frame: The frame to process as Celux frame.
            normStream: The CUDA stream for normalization (if applicable).
            channels_first: If True, frame is (C, H, W). If False, (H, W, C).

        Returns:
            The processed frame as a torch tensor.
        """
        import torch
        from torch.nn import functional as F

        norm = 1 / 255.0 if frame.dtype == torch.uint8 else 1 / 65535.0
        if self.cudaEnabled:
            with torch.cuda.stream(normStream):
                try:
                    frame = frame.pin_memory()
                except Exception:
                    pass
                frame = frame.to(
                    device="cuda",
                    non_blocking=True,
                    dtype=torch.float16 if self.half else torch.float32,
                )

                if not channels_first:
                    frame = frame.permute(2, 0, 1)

                frame.mul_(norm)
                frame.clamp_(0, 1)

                if self.resize:
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
            try:
                frame = frame.pin_memory()
            except Exception:
                pass

            frame = frame.to(
                device="cpu",
                non_blocking=False,
                dtype=torch.float16 if self.half else torch.float32,
            )

            if not channels_first:
                frame = frame.permute(2, 0, 1)

            frame.mul_(norm)
            frame.clamp_(0, 1)

            if self.resize:
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

    def peek(self):
        """
        Peeks at the next frame in the decodeBuffer without removing it.

        Returns:
            The next frame from the decodeBuffer, or None if decoding is finished and queue is empty.
        """
        while True:
            with self.decodeBuffer.mutex:
                if len(self.decodeBuffer.queue) > 0:
                    return self.decodeBuffer.queue[0]

            if self.isFinished:
                return None

            self._frameAvailable.wait(timeout=0.1)
            self._frameAvailable.clear()

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
        sharpen: bool = False,
        sharpen_sens: float = 0.0,
        grayscale: bool = False,
        transparent: bool = False,
        benchmark: bool = False,
        bitDepth: str = "8bit",
        inpoint: float = 0.0,
        outpoint: float = 0.0,
        slowmo: bool = False,
        output_scale_width: int = None,
        output_scale_height: int = None,
        enablePreview: bool = False,
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
        sharpen: bool - Whether to apply a sharpening filter to the video.
        sharpen_sens: float - The sensitivity of the sharpening filter.
        transparent: bool - Whether to encode the video with transparency.
        audio: bool - Whether to include audio in the output video.
        benchmark: bool - Whether to benchmark the encoding process, this will not output any video.
        bitDepth: str - The bit depth of the output video. Options include "8bit" and "10bit".
        inpoint: float - The start time of the segment to encode, in seconds.
        outpoint: float - The end time of the segment to encode, in seconds.
        output_scale_width: int - The target width for output scaling (optional).
        output_scale_height: int - The target height for output scaling (optional).
        enablePreview: bool - Whether to enable FFmpeg-based preview output (optional).
        """
        self.input = input
        self.output = os.path.normpath(output)
        self.encode_method = encode_method

        if self.encode_method == "png" and "%" not in self.output:
            # Check if it has an extension
            _, ext = os.path.splitext(self.output)
            if not ext:
                # It's likely a directory
                self.output = os.path.join(self.output, "%08d.png")
            else:
                # It's a file path, assume we want a sequence in the same dir
                # If it's something like "video.mp4", switch to "video_%08d.png"
                base, _ = os.path.splitext(self.output)
                self.output = f"{base}_%08d.png"

        self.custom_encoder = custom_encoder
        self.grayscale = grayscale
        self.width = width
        self.height = height
        self.fps = fps
        self.sharpen = sharpen
        self.sharpen_sens = sharpen_sens
        self.transparent = transparent
        self.benchmark = benchmark
        self.bitDepth = bitDepth
        self.inpoint = inpoint
        self.outpoint = outpoint
        self.slowmo = slowmo
        self.output_scale_width = output_scale_width
        self.output_scale_height = output_scale_height
        self.enablePreview = enablePreview

        self.writtenFrames = 0
        self.writeBuffer = Queue(maxsize=64)

        self.previewPath = (
            os.path.join(cs.MAINPATH, "preview.jpg") if enablePreview else None
        )

    def encodeSettings(self) -> list:
        """
        Simplified structure for setting input/output pix formats
        and building FFMPEG command.
        """
        # Set environment variables
        os.environ["FFREPORT"] = "file=FFmpeg-Log.log:level=32"
        if "av1" in [self.encode_method, self.custom_encoder]:
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
            "quiet",
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

        if self.outpoint != 0 and not self.slowmo:
            command.extend(
                [
                    "-itsoffset",
                    str(self.inpoint),
                    "-i",
                    "pipe:0",
                    "-ss",
                    str(self.inpoint),
                    "-to",
                    str(self.outpoint),
                ]
            )
        else:
            command.extend(["-i", "pipe:0"])

        if cs.AUDIO:
            command.extend(["-thread_queue_size", "1024", "-i", self.input])

        filterList = self._buildFilterList()

        if self.enablePreview:
            filterComplexParts = []

            if filterList:
                baseFilters = ",".join(filterList)
                filterComplexParts.append(f"[0:v]{baseFilters},split=2[main][preview]")
            else:
                filterComplexParts.append("[0:v]split=2[main][preview]")

            filterComplexParts.append("[preview]fps=2[previewThrottled]")

            combinedFilter = ";".join(filterComplexParts)
            command.extend(["-filter_complex", combinedFilter, "-filter_complex_threads", "0"])

            command.extend(["-map", "[main]"])

            if not self.custom_encoder:
                command.extend(matchEncoder(self.encode_method))
                command.extend(["-pix_fmt", outputPixFmt])
            else:
                customArgs = self.custom_encoder.split()
                if "-vf" in customArgs:
                    vfIdx = customArgs.index("-vf")
                    customArgs.pop(vfIdx)
                    customArgs.pop(vfIdx)
                if "-pix_fmt" not in customArgs:
                    customArgs.extend(["-pix_fmt", outputPixFmt])
                command.extend(customArgs)

            if cs.AUDIO:
                command.extend(self._buildAudioSettings())

            command.append(self.output)

            command.extend(
                [
                    "-map",
                    "[previewThrottled]",
                    "-q:v",
                    "2",
                    "-update",
                    "1",
                    self.previewPath,
                ]
            )
        else:
            command.extend(["-map", "0:v"])

            if not self.custom_encoder:
                command.extend(matchEncoder(self.encode_method))

                if useHwUpload:
                    hwFilters = filterList.copy() if filterList else []
                    hwFilters.append("format=nv12")
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

            command.append(self.output)

        return command

    def _getOutputFormat(self):
        ext = os.path.splitext(self.output)[1].lower()
        formatMap = {
            ".mp4": "mp4",
            ".mkv": "matroska",
            ".webm": "webm",
            ".mov": "mov",
            ".avi": "avi",
        }
        return formatMap.get(ext, "mp4")

    def _buildFilterList(self):
        """Build list of video filters based on settings"""
        filterList = []

        if self.output_scale_width and self.output_scale_height:
            filterList.append(
                f"scale={self.output_scale_width}:{self.output_scale_height}:flags=bilinear"
            )

        if self.sharpen:
            filterList.append(f"cas={self.sharpen_sens}")
        if self.grayscale:
            filterList.append(
                "format=gray" if self.bitDepth == "8bit" else "format=gray16be"
            )
        if self.transparent:
            filterList.append("format=yuva420p")

        """
                "-vf",
            "zscale=matrix=709:dither=error_diffusion,format=yuv420p",
            """

        import json

        metadata = json.loads(open(cs.METADATAPATH, "r", encoding="utf-8").read())
        if not self.grayscale and not self.transparent:
            colorSPaceFilter = {
                "bt709": f"zscale=matrix=709:dither=error_diffusion,format={self.inputPixFmt}",
                "bt2020": "zscale=matrix=bt2020:norm=bt2020:dither=error_diffusion,format=yuv420p",
            }

            metadataFields = ["ColorSpace", "PixelFormat", "ColorTRT"]
            detectedColorSpace = None

            for field in metadataFields:
                colorValue = metadata["metadata"].get(field, "unknown")
                if colorValue in colorSPaceFilter:
                    detectedColorSpace = colorValue
                    break

            filterList.append(
                colorSPaceFilter.get(detectedColorSpace, colorSPaceFilter["bt709"])
            )

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

    def _buildAudioSettings(self):
        """Build audio encoding settings"""
        audioSettings = ["-map", "1:a"]

        audioCodec = "copy"
        subCodec = "copy"
        if self.output.endswith(".webm"):
            audioCodec = "libopus"
            subCodec = "webvtt"
        audioSettings.extend(["-c:a", audioCodec, "-map", "1:s?", "-c:s", subCodec])

        if self.outpoint != 0:
            audioSettings.extend(["-ss", str(self.inpoint), "-to", str(self.outpoint)])

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

            command = self.encodeSettings()
            logging.info(f"Encoding with: {' '.join(map(str, command))}")

            if self.enablePreview:
                logging.info(f"Preview enabled, writing to: {self.previewPath}")
                from src.utils.logAndPrint import logAndPrint

                logAndPrint(f"Preview will be saved to: {self.previewPath}", "cyan")

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
                stdout=None,
                stderr=subprocess.DEVNULL,
                shell=False,
                cwd=cs.MAINPATH,
            )

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
                        if pendingBuffer is not None:
                            pendingEvent.synchronize()
                            ffmpegProc.stdin.write(memoryview(pendingBuffer.numpy()))
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

                        gpuTensor = (
                            frame.squeeze(0)
                            .permute(1, 2, 0)
                            .mul(multiplier)
                            .clamp(0, multiplier)
                            .to(dtype)
                            .contiguous()
                        )

                        currentBuffer = pinnedBuffers[bufferIdx]
                        currentBuffer.copy_(gpuTensor, non_blocking=True)
                        currentEvent = transferEvents[bufferIdx]
                        currentEvent.record(transferStream)

                    if pendingBuffer is not None:
                        pendingEvent.synchronize()
                        ffmpegProc.stdin.write(memoryview(pendingBuffer.numpy()))
                        writtenFrames += 1

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
                        break

                    if needsResize:
                        frame = F.interpolate(
                            frame,
                            size=(self.height, self.width),
                            mode="bicubic",
                            align_corners=False,
                        )
                    frameTensor = (
                        frame.squeeze(0)
                        .permute(1, 2, 0)
                        .mul(multiplier)
                        .clamp(0, multiplier)
                        .to(dtype)
                        .contiguous()
                    )

                    ffmpegProc.stdin.write(memoryview(frameTensor.numpy()))
                    writtenFrames += 1

            logging.info(f"Encoded {writtenFrames} frames")

        except Exception as e:
            logging.error(f"Encoding error: {e}")
        finally:
            try:
                if ffmpegProc is not None and ffmpegProc.stdin:
                    ffmpegProc.stdin.close()
                if ffmpegProc is not None:
                    ffmpegProc.wait(timeout=3)

            except Exception as e:
                logging.warning(f"Cleanup error: {e}")

    def write(self, frame):
        """
        Add a frame to the queue. Must be in [B, C, H, W] format.
        Frame type is torch.Tensor when using PyTorch backend.
        """
        self.writeBuffer.put(frame)

    def put(self, frame):
        """
        Equivalent to write()
        Add a frame to the queue. Must be in [B, C, H, W] format.
        Frame type is torch.Tensor when using PyTorch backend.
        """
        self.writeBuffer.put(frame)

    def close(self):
        self.writeBuffer.put(None)

        if self.previewPath and os.path.exists(self.previewPath):
            try:
                os.remove(self.previewPath)
            except Exception as e:
                logging.warning(f"Could not remove preview file: {e}")
