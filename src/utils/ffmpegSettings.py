import logging
import subprocess
import os
import torch
import src.constants as cs
import time
import celux

from queue import Queue
from torch.nn import functional as F
from src.utils.encodingSettings import matchEncoder, getPixFMT
from .isCudaInit import CudaChecker


checker = CudaChecker()


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

        """
        self.half = half
        self.decodeBuffer = Queue(maxsize=20)
        self.useOpenCV = False
        self.width = width
        self.height = height
        self.resize = resize
        self.isFinished = False
        self.bitDepth = bitDepth
        self.videoInput = os.path.normpath(videoInput)
        self.toTorch = toTorch
        self.inpoint = inpoint
        self.outpoint = outpoint

        if not os.path.exists(videoInput):
            raise FileNotFoundError(f"Video file not found: {videoInput}")

        # Determine device and create CUDA stream if possible; gracefully fall back on failure
        self.cudaEnabled = False
        if checker.cudaAvailable:
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

    def __call__(self):
        """
        Decodes frames from the video and stores them in the decodeBuffer.
        """
        decodedFrames = 0

        try:
            if self.inpoint > 0 or self.outpoint > 0:
                reader = celux.VideoReader(self.videoInput)(
                    [float(self.inpoint), float(self.outpoint)]
                )
            else:
                reader = celux.VideoReader(
                    self.videoInput,
                )

            for frame in reader:
                if self.toTorch:
                    frame = self.processFrameToTorch(
                        frame, self.normStream if self.cudaEnabled else None
                    )
                else:
                    frame = self.processFrameToNumpy(frame)

                self.decodeBuffer.put(frame)
                decodedFrames += 1

        except Exception as e:
            logging.error(f"Decoding error: {e}")
        finally:
            self.decodeBuffer.put(None)

            self.isFinished = True
            logging.info(f"Decoded {decodedFrames} frames")

    def processFrameToNumpy(self, frame):
        """
        Processes a single frame and converts it to a numpy array.

        Args:
            frame: The frame to process as Celux frame.

        Returns:
            The processed frame as a numpy array.
        """
        norm = 1 / 255.0 if frame.dtype == torch.uint8 else 1 / 65535.0
        frame = frame.permute(2, 0, 1)
        frame = frame.mul(norm)
        frame = frame.clamp(0, 1)
        if self.resize:
            frame = F.interpolate(
                frame.unsqueeze(0),
                size=(self.height, self.width),
                mode="bicubic",
                align_corners=False,
            ).squeeze(0)
        else:
            frame = frame.unsqueeze(0)

        frame = frame.half() if self.half else frame.float()
        frame = frame.cpu().numpy()

        return frame

    def processFrameToTorch(self, frame, normStream=None):
        """
        Processes a single frame with optimized memory handling.

        Args:
            frame: The frame to process as Celux frame.
            normStream: The CUDA stream for normalization (if applicable).

        Returns:
            The processed frame as a torch tensor.
        """
        norm = 1 / 255.0 if frame.dtype == torch.uint8 else 1 / 65535.0
        if self.cudaEnabled:
            with torch.cuda.stream(normStream):
                try:
                    frame = frame.pin_memory()
                except Exception:
                    pass
                frame = (
                    frame.to(
                        device="cuda",
                        non_blocking=True,
                        dtype=torch.float16 if self.half else torch.float32,
                    )
                    .permute(2, 0, 1)
                    .mul(norm)
                    .clamp(0, 1)
                )

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

            frame = (
                frame.to(
                    device="cpu",
                    non_blocking=False,
                    dtype=torch.float16 if self.half else torch.float32,
                )
                .permute(2, 0, 1)
                .mul(norm)
                .clamp(0, 1)
            )

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
            The next frame from the decodeBuffer, or None if the queue is empty.
        """
        if self.decodeBuffer.empty():
            return None

        with self.decodeBuffer.mutex:
            if len(self.decodeBuffer.queue) > 0:
                return self.decodeBuffer.queue[0]
            if self.isFinished:
                return None
            else:
                while len(self.decodeBuffer.queue) == 0:
                    time.sleep(0.01)
                return self.decodeBuffer.queue[0]
            return None

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
        return self.decodeBuffer.empty() and self.isFinished


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
        """
        self.input = input
        self.output = os.path.normpath(output)
        self.encode_method = encode_method
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

        self.writtenFrames = 0
        self.writeBuffer = Queue(maxsize=10)

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

    def _buildEncodingCommand(self, outputPixFmt):
        """Build FFmpeg command for encoding"""
        command = [
            cs.FFMPEGPATH,
            "-y",
            "-hide_banner",
            "-loglevel",
            "quiet",
            "-nostats",
            "-f",
            "rawvideo",
            "-pix_fmt",
            self.inputPixFmt,
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            str(self.fps),
        ]

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
            command.extend(["-i", self.input])

        command.extend(["-map", "0:v"])

        filterList = self._buildFilterList()
        if not self.custom_encoder:
            command.extend(matchEncoder(self.encode_method))
            if filterList:
                command.extend(["-vf", ",".join(filterList)])
            command.extend(["-pix_fmt", outputPixFmt])
        else:
            command.extend(self._buildCustomEncoder(filterList, outputPixFmt))

        if cs.AUDIO:
            command.extend(self._buildAudioSettings())

        command.append(self.output)
        return command

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
                time.sleep(0.01)
            except KeyboardInterrupt:
                logging.warning("Encoding interrupted by user")
                return

        ffmpegProc = None
        try:
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

            hostBuffers = [
                torch.empty(
                    (self.height, self.width, self.channels),
                    dtype=dtype,
                )
                for _ in range(2)
            ]

            for buf in hostBuffers:
                try:
                    buf = buf.pin_memory()
                except Exception:
                    pass

            useCuda = False
            normStream = None
            events = None
            if checker.cudaAvailable:
                try:
                    normStream = torch.cuda.Stream()
                    events = [
                        torch.cuda.Event(enable_timing=False),
                        torch.cuda.Event(enable_timing=False),
                    ]
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

            bufIdx = 0
            prevIDX = None

            while True:
                frame = self.writeBuffer.get()
                if frame is None:
                    break

                if useCuda:
                    with torch.cuda.stream(normStream):
                        if needsResize:
                            frame = F.interpolate(
                                frame,
                                size=(self.height, self.width),
                                mode="bicubic",
                                align_corners=False,
                            )

                        GpuHWCInt = (
                            frame.mul(multiplier)
                            .clamp(0, multiplier)
                            .squeeze(0)
                            .permute(1, 2, 0)
                            .to(dtype)
                        )
                        hostBuffers[bufIdx].copy_(GpuHWCInt, non_blocking=True)
                        events[bufIdx].record(normStream)

                    if prevIDX is not None:
                        events[prevIDX].synchronize()
                        arr = hostBuffers[prevIDX].numpy()
                        ffmpegProc.stdin.write(memoryview(arr))
                        writtenFrames += 1

                    prevIDX = bufIdx
                    bufIdx ^= 1

                else:
                    if needsResize:
                        frame = F.interpolate(
                            frame,
                            size=(self.height, self.width),
                            mode="bicubic",
                            align_corners=False,
                        )
                    CpuHWCInt = (
                        frame.mul(multiplier)
                        .clamp(0, multiplier)
                        .squeeze(0)
                        .permute(1, 2, 0)
                        .to(dtype)
                    )
                    hostBuffers[0].copy_(CpuHWCInt, non_blocking=False)
                    arr = hostBuffers[0].numpy()
                    ffmpegProc.stdin.write(memoryview(arr))
                    writtenFrames += 1

            if useCuda and prevIDX is not None:
                events[prevIDX].synchronize()
                arr = hostBuffers[prevIDX].numpy()
                ffmpegProc.stdin.write(memoryview(arr))
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

    def write(self, frame: torch.Tensor):
        """
        Add a frame to the queue. Must be in [B, C, H, W] format.
        """
        self.writeBuffer.put(frame)

    def put(self, frame: torch.Tensor):
        """
        Equivalent to write()
        Add a frame to the queue. Must be in [B, C, H, W] format.
        """
        self.writeBuffer.put(frame)

    def close(self):
        """
        Close the queue.
        """
        self.writeBuffer.put(None)
