import logging
import subprocess
import os
import torch
import numpy as np
import cv2
import src.constants as cs
import bv
import time

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
        totalFrames: int = 0,
        fps: float = 0,
        half: bool = True,
        resize: bool = False,
        width: int = 1920,
        height: int = 1080,
        bitDepth: str = "8bit",
    ):
        """
        Initializes the BuildBuffer class.

        Args:
            videoInput (str): Path to the video input file.
            inpoint (float): The starting point in seconds for decoding.
            outpoint (float): The ending point in seconds for decoding.
            totalFrames (int): The total number of frames in the video.
            fps (float): Frames per second of the video.
            half (bool): Whether to use half precision for decoding.
            resize (bool): Whether to resize the frames.
            width (int): Width to resize frames to.
            height (int): Height to resize frames to.
            bitDepth (str): Bit depth of the video. Options are "8bit" or "16bit".
        """
        self.half = half
        self.decodeBuffer = Queue(maxsize=20)
        self.useOpenCV = False
        self.width = width
        self.height = height
        self.resize = resize
        self.isFinished = False
        self.bitDepth = bitDepth

        self.inputFramePoint = round(inpoint * fps)
        self.outputFramePoint = round(outpoint * fps)

        if not os.path.exists(videoInput):
            raise FileNotFoundError(f"Video file not found: {videoInput}")

        if self.inputFramePoint != 0 and self.outputFramePoint != 0:
            logging.info(
                f"Decoding frames from {self.inputFramePoint} to {self.outputFramePoint}"
            )
        try:
            self.reader = bv.open(
                videoInput,
            )

            if inpoint > 0 and outpoint != 0:
                self.reader.seek(
                    int(inpoint * 1000000),
                    stream=self.reader.streams.video[0],
                    backward=True,
                    any_frame=False,
                )

        except Exception as e:
            logging.error(f"Failed to open video: {e}")
            raise

        if checker.cudaAvailable:
            self.deviceType = "cuda"
            self.normStream = torch.cuda.Stream()
            self.preAllocFrame = None
        else:
            self.deviceType = "cpu"

    def __call__(self):
        """
        Decodes frames from the video and stores them in the decodeBuffer.
        """
        decodedFrames = 0

        try:
            stream = self.reader.streams.video[0]
            stream.thread_type = "AUTO"

            frameGen = self.reader.decode(video=0)

            for frameIdx, frame in enumerate(frameGen):
                if self.inputFramePoint != 0 or self.outputFramePoint != 0:
                    if self.inputFramePoint + frameIdx >= self.outputFramePoint:
                        break

                if self.resize:
                    frame = frame.reformat(self.width, self.height)

                # if self.bitDepth == "8bit":
                #    frameArray = frame.to_ndarray(format="rgb24")
                # elif self.bitDepth == "16bit":
                #    frameArray = frame.to_ndarray(format="rgb48")
                frameArray = frame.to_ndarray(format="rgb24")

                processedFrame = self.processFrame(
                    frameArray, self.normStream if checker.cudaAvailable else None
                )

                self.decodeBuffer.put(processedFrame)
                decodedFrames += 1

        except Exception as e:
            logging.error(f"Decoding error: {e}")
        finally:
            self.decodeBuffer.put(None)

            try:
                if hasattr(self, "reader") and self.reader:
                    self.reader.close()

                if (
                    checker.cudaAvailable
                    and hasattr(self, "preAllocFrame")
                    and self.preAllocFrame is not None
                ):
                    del self.preAllocFrame
                    torch.cuda.empty_cache()
            except Exception as e:
                logging.warning(f"Cleanup error: {e}")

            self.isFinished = True
            logging.info(f"Decoded {decodedFrames} frames")

    def processFrame(self, frame, normStream=None):
        """
        Processes a single frame with optimized memory handling.

        Args:
            frame: The frame to process as numpy array.
            normStream: The CUDA stream for normalization (if applicable).

        Returns:
            The processed frame as a torch tensor.
        """
        frameTensor = torch.from_numpy(frame)
        multiply = 1 / 255.0 if frameTensor.dtype == torch.uint8 else 1 / 65535.0
        dtype = torch.float16 if self.half else torch.float32

        if checker.cudaAvailable:
            with torch.cuda.stream(normStream):
                result = (
                    frameTensor.to(device="cuda", non_blocking=True, dtype=dtype)
                    .mul(multiply)
                    .clamp(0, 1)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .contiguous()
                )

            normStream.synchronize()
            return result
        else:
            return (
                frameTensor.mul(multiply)
                .clamp(0, 1)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(dtype=dtype)
                .contiguous()
            )

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
        isClosed = False
        if self.decodeBuffer.empty() and self.isFinished:
            try:
                # Safely handle queue shutdown
                self.decodeBuffer.shutdown()  # new method to close the queue introduced with python 3.13.
                isClosed = True
            except (AttributeError, Exception) as e:
                # Handle older Python versions or errors
                logging.debug(f"Queue shutdown failed: {e}")
                isClosed = True

        return isClosed


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
        realtime: bool = False,
        slowmo: bool = False,
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
        realtime: bool - Whether to preview the video in real-time using FFPLAY.
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
        self.realtime = realtime
        self.slowmo = slowmo

        self.writtenFrames = 0
        self.writeBuffer = Queue(maxsize=10)
        self.mpvPath = (
            os.path.join(cs.MAINPATH, "ffmpeg", "mpv.exe") if self.realtime else None
        )
        if self.realtime and not os.path.exists(self.mpvPath):
            logging.warning(
                f"MPV not found at {self.mpvPath}. Disabling realtime preview."
            )
            self.realtime = False

    def encodeSettings(self) -> list:
        """
        Simplified structure for setting input/output pix formats
        and building FFMPEG command.
        """
        # Set environment variables
        os.environ["FFREPORT"] = "file=FFmpeg-Log.log:level=32"
        if "av1" in [self.encode_method, self.custom_encoder]:
            os.environ["SVT_LOG"] = "0"

        inputPixFmt, outputPixFmt, self.encode_method = getPixFMT(
            self.encode_method, self.bitDepth, self.grayscale, self.transparent
        )

        if self.benchmark:
            return self._buildBenchmarkCommand(inputPixFmt)
        else:
            return self._buildEncodingCommand(inputPixFmt, outputPixFmt)

    def _buildBenchmarkCommand(self, inputPixFmt):
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
            inputPixFmt,
            "-r",
            str(self.fps),
            "-i",
            "-",
            "-benchmark",
            "-f",
            "null",
            "-",
        ]

    def _buildEncodingCommand(self, inputPixFmt, outputPixFmt):
        """Build FFmpeg command for encoding"""
        command = [
            cs.FFMPEGPATH,
            "-y",
            "-report",
            "-hide_banner",
            "-loglevel",
            "quiet",
            "-nostats",
            "-f",
            "rawvideo",
            "-pixel_format",
            inputPixFmt,
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

        if not self.realtime:
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

        if self.realtime:
            command.extend(["-f", "matroska", "-"])
        else:
            command.append(self.output)

        return command

    def _buildFilterList(self):
        """Build list of video filters based on settings"""
        filterList = []
        if self.sharpen:
            filterList.append(f"cas={self.sharpen_sens}")
        if self.grayscale:
            filterList.append(
                "format=gray" if self.bitDepth == "8bit" else "format=gray16be"
            )
        if self.transparent:
            filterList.append("format=yuva420p")

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
        self.frameQueue = Queue(maxsize=10)
        writtenFrames = 0

        # Wait for at least one frame to be queued before starting encoding
        while self.writeBuffer.empty():
            try:
                time.sleep(0.01)
            except KeyboardInterrupt:
                logging.warning("Encoding interrupted by user")
                return

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

            device = "cuda" if checker.cudaAvailable else "cpu"

            dummyTensor = torch.zeros(
                (self.height, self.width, self.channels), dtype=dtype, device=device
            )

            if checker.cudaAvailable:
                normStream = torch.cuda.Stream()

            # Start FFmpeg subprocess
            ffmpegProc = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE if self.realtime else None,
                stderr=subprocess.PIPE,
                shell=False,
                cwd=cs.MAINPATH,
            )

            mpvProc = None
            if self.realtime and ffmpegProc.stdout:
                mpvProc = subprocess.Popen(
                    [
                        self.mpvPath,
                        "-",
                        "--no-terminal",
                        "--force-window=yes",
                        "--keep-open=yes",
                        "--title=" + self.input,
                        "--cache=yes",
                        "--demuxer-max-bytes=5M",
                        "--demuxer-readahead-secs=5",
                        "--demuxer-seekable-cache=yes",
                        "--hr-seek-framedrop=no",
                        "--hwdec=auto",
                        "--border=no",
                        "--profile=high-quality",
                        "--vo=gpu-next",
                    ],
                    stdin=ffmpegProc.stdout,
                    shell=False,
                )
                ffmpegProc.stdout.close()

            while True:
                frame = self.writeBuffer.get()
                if frame is None:
                    break

                if checker.cudaAvailable:
                    with torch.cuda.stream(normStream):
                        if needsResize:
                            frame = F.interpolate(
                                frame,
                                size=(self.height, self.width),
                                mode="bicubic",
                                align_corners=False,
                            )

                        dummyTensor.copy_(
                            frame.mul(multiplier)
                            .clamp(0, multiplier)
                            .squeeze(0)
                            .permute(1, 2, 0),
                            non_blocking=True,
                        )
                    normStream.synchronize()
                else:
                    if needsResize:
                        frame = F.interpolate(
                            frame,
                            size=(self.height, self.width),
                            mode="bicubic",
                            align_corners=False,
                        )
                    dummyTensor.copy_(
                        frame.mul(multiplier)
                        .clamp(0, multiplier)
                        .squeeze(0)
                        .permute(1, 2, 0),
                        non_blocking=False,
                    )

                frameData = dummyTensor.cpu().numpy()
                ffmpegProc.stdin.write(np.ascontiguousarray(frameData))
                writtenFrames += 1

            self.frameQueue.put(None)
            logging.info(f"Encoded {writtenFrames} frames")

        except Exception as e:
            logging.error(f"Encoding error: {e}")
        finally:
            try:
                if "ffmpegProc" in locals() and ffmpegProc.stdin:
                    ffmpegProc.stdin.close()
                if "ffmpegProc" in locals() and ffmpegProc.poll() is None:
                    ffmpegProc.wait(timeout=5)
                if "mpvProc" in locals() and mpvProc and mpvProc.poll() is None:
                    mpvProc.wait(timeout=5)
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
