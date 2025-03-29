import logging
import subprocess
import os
import torch
import numpy as np
import cv2
import json
import src.constants as cs

from queue import Queue
from celux import VideoReader, Scale
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
        decodeThreads: int = 0,
        resize: bool = False,
        resizeMethod: str = "bicubic",
        width: int = 1920,
        height: int = 1080,
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
            decodeThreads (int): Amount of threads allocated to decoding.
            resize (bool): Whether to resize the frames.
            resizeFactor (float): The factor to resize the frames by.

        Attributes:
            half (bool): Whether to use half precision for decoding.
            decodeBuffer (Queue): A queue to store decoded frames.
            reader (celux.VideoReader or cv2.VideoCapture): Video reader object for decoding frames.
        """
        self.half = half
        self.decodeBuffer = Queue(maxsize=10)
        self.useOpenCV = False
        self.width = width
        self.height = height

        inputFramePoint = round(inpoint * fps)
        outputFramePoint = round(outpoint * fps) if outpoint != 0.0 else totalFrames
        if resize:
            filters = [Scale(width=str(width), height=str(height), flags=resizeMethod)]
        else:
            filters = []

        logging.info(f"Decoding frames from {inputFramePoint} to {outputFramePoint}")
        jsonMetadata = json.load(open(os.path.join(cs.MAINPATH, "metadata.json"), "r"))

        if jsonMetadata["Codec"] is None or jsonMetadata["Codec"] in [
            "av1",
            "rawvideo",
        ]:
            logging.info(
                "The video codec is unsupported by Celux, falling back to OpenCV for video decoding"
            )
            self.useOpenCV = True
            self.initializeOpenCV(videoInput, inputFramePoint, outputFramePoint)
        else:
            try:
                if outpoint != 0.0:
                    self.reader = VideoReader(
                        videoInput,
                        num_threads=decodeThreads,
                        filters=filters,
                        tensor_shape="HWC",
                    )([float(inpoint), float(outpoint)])
                else:
                    self.reader = VideoReader(
                        videoInput,
                        num_threads=decodeThreads,
                        filters=filters,
                        tensor_shape="HWC",
                    )
                logging.info("Using Celux pipeline for video decoding")
            except Exception as e:
                logging.error(f"Failed to initialize Celux pipeline: {e}")
                logging.info("Falling back to OpenCV for video decoding")
                self.useOpenCV = True
                self.initializeOpenCV(videoInput, inputFramePoint, outputFramePoint)

        # Delete from memory, can't trust the garbage collector
        del jsonMetadata

    def initializeOpenCV(
        self, videoInput: str, inputFramePoint: int = 0, outputFramePoint: int = 0
    ):
        """
        Initializes the OpenCV video reader.
        """
        self.reader = cv2.VideoCapture(videoInput)
        if inputFramePoint != 0.0 or outputFramePoint != 0.0:
            self.reader.set(cv2.CAP_PROP_POS_FRAMES, inputFramePoint)
            self.outputFramePoint = outputFramePoint

    def __call__(self):
        """
        Decodes frames from the video and stores them in the decodeBuffer.
        """
        decodedFrames = 0
        self.isFinished = False

        if checker.cudaAvailable:
            normStream = torch.cuda.Stream()

        # OpenCV fallback in case of issues
        if self.useOpenCV:
            print("Using OpenCV for video decoding")
            while self.reader.isOpened():
                ret, frame = self.reader.read()
                if not ret or decodedFrames >= self.outputFramePoint:
                    break
                frame = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame = self.processFrame(
                    frame, normStream if checker.cudaAvailable else None
                )
                self.decodeBuffer.put(frame)
                decodedFrames += 1
            self.reader.release()
        else:
            for frame in self.reader:
                frame = self.processFrame(
                    frame, normStream if checker.cudaAvailable else None
                )
                self.decodeBuffer.put(frame)
                decodedFrames += 1

        self.isFinished = True
        logging.info(f"Decoded {decodedFrames} frames")

    def processFrame(self, frame, normStream=None):
        """
        Processes a single frame.

        Args:
            frame: The frame to process.
            normStream: The CUDA stream for normalization (if applicable).

            Returns:
                The processed frame.
        """
        multiply = 1 / 255 if frame.dtype == torch.uint8 else 1 / 65535
        dtype = torch.float16 if self.half else torch.float32

        if checker.cudaAvailable:
            with torch.cuda.stream(normStream):
                frame = (
                    frame.to(device="cuda", non_blocking=True, dtype=dtype)
                    .mul(multiply)
                    .clamp(0, 1)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .contiguous()
                )
            normStream.synchronize()
            return frame
        else:
            return (
                frame.mul(multiply)
                .clamp(0, 1)
                .permute(2, 0, 1)
                .unsqueeze(0)
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
        return self.decodeBuffer.empty()


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

    def encodeSettings(self) -> list:
        """
        Simplified structure for setting input/output pix formats
        and building FFMPEG command.
        """
        # Set environment variables
        os.environ["FFREPORT"] = "file=FFmpeg-Log.log:level=32"
        if "av1" in [self.encode_method, self.custom_encoder]:
            os.environ["SVT_LOG"] = "0"

        inputPixFormat, outputPixFormat, self.encode_method = getPixFMT(
            self.encode_method, self.bitDepth, self.grayscale, self.transparent
        )

        if self.benchmark:
            return self._buildBenchmarkCommand(inputPixFormat)
        else:
            return self._buildEncodingCommand(inputPixFormat, outputPixFormat)

    def _buildBenchmarkCommand(self, inputPixFormat):
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
            inputPixFormat,
            "-r",
            str(self.fps),
            "-i",
            "-",
            "-benchmark",
            "-f",
            "null",
            "-",
        ]

    def _buildEncodingCommand(self, inputPixFormat, outputPixFormat):
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
            inputPixFormat,
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
            filters = self._buildFilterList()

            if not self.custom_encoder:
                command.extend(matchEncoder(self.encode_method))
                if filters:
                    command.extend(["-vf", ",".join(filters)])
                command.extend(["-pix_fmt", outputPixFormat])
            else:
                command.extend(self._buildCustomEncoder(filters, outputPixFormat))

        if cs.AUDIO:
            command.extend(self._buildAudioSettings())

        if self.realtime:
            command.extend(["-f", "matroska", "-"])
        else:
            command.append(self.output)

        return command

    def _buildFilterList(self):
        """Build list of video filters based on settings"""
        filters = []
        if self.sharpen:
            filters.append(f"cas={self.sharpen_sens}")
        if self.grayscale:
            filters.append(
                "format=gray" if self.bitDepth == "8bit" else "format=gray16be"
            )
        if self.transparent:
            filters.append("format=yuva420p")

        return filters

    def _buildCustomEncoder(self, filters, outputPixFormat):
        """Apply custom encoder settings with filters"""
        customEncoderList = self.custom_encoder.split()

        if "-vf" in customEncoderList:
            vfIndex = customEncoderList.index("-vf")
            filterString = customEncoderList[vfIndex + 1]
            for filter_item in filters:
                filterString += f",{filter_item}"
            customEncoderList[vfIndex + 1] = filterString
        elif filters:
            customEncoderList.extend(["-vf", ",".join(filters)])

        if "-pix_fmt" not in customEncoderList:
            logging.info(f"-pix_fmt was not found, adding {outputPixFormat}.")
            customEncoderList.extend(["-pix_fmt", outputPixFormat])

        return customEncoderList

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

        command = self.encodeSettings()
        logging.info(f"Encoding options: {' '.join(map(str, command))}")

        if self.grayscale:
            self.channels = 1
        elif self.transparent:
            self.channels = 4
        else:
            self.channels = 3

        dtype = torch.uint8 if self.bitDepth == "8bit" else torch.uint16
        mul = 255 if self.bitDepth == "8bit" else 65535

        dummyTensor = torch.zeros(
            (self.height, self.width, self.channels),
            dtype=dtype,
            device="cuda" if checker.cudaAvailable else "cpu",
        )

        if checker.cudaAvailable:
            normStream = torch.cuda.Stream()
            try:
                dummyTensor = dummyTensor.pin_memory()
            except Exception:
                pass

        while self.writeBuffer.empty():
            pass

        initialFrame = self.writeBuffer.queue[0]

        NEEDSRESIZE: bool = (
            initialFrame.shape[2] != self.height or initialFrame.shape[3] != self.width
        )
        if NEEDSRESIZE:
            logging.info(
                f"The frame size does not match the output size, resizing the frame. Frame size: {initialFrame.shape[3]}x{initialFrame.shape[2]}, Output size: {self.width}x{self.height}"
            )

        ffmpegSubprocess = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            shell=False,
            cwd=cs.MAINPATH,
        )

        if self.realtime:
            # glslPath = (
            #    r"C:\Users\nilas\AppData\Roaming\TheAnimeScripter\ffmpeg\FSR.glsl"
            # )
            mpvSubprocess = subprocess.Popen(
                [
                    self.mpvPath,
                    "-",
                    "--no-terminal",
                    "--force-window=yes",
                    "--keep-open=yes",
                    "--title=" + input,
                    "--cache=yes",
                    "--demuxer-max-bytes=5M",
                    "--demuxer-readahead-secs=5",
                    "--demuxer-seekable-cache=yes",
                    "--hr-seek-framedrop=no",
                    "--hwdec=auto",
                    "--border=no",
                    "--profile=high-quality",
                    "--vo=gpu-next",
                    # "--profile=gpu-hq",
                    # "--glsl-shader=" + glslPath,
                ],
                stdin=ffmpegSubprocess.stdout,
                shell=False,
            )
            ffmpegSubprocess.stdout.close()

        while True:
            frame = self.writeBuffer.get()
            if frame is None:
                break

            if checker.cudaAvailable:
                with torch.cuda.stream(normStream):
                    if NEEDSRESIZE:
                        frame = F.interpolate(
                            frame,
                            size=(self.height, self.width),
                            mode="bicubic",
                            align_corners=False,
                        )

                    dummyTensor.copy_(
                        frame.mul(mul).clamp(0, mul).squeeze(0).permute(1, 2, 0),
                        non_blocking=True,
                    )
                normStream.synchronize()
            else:
                if NEEDSRESIZE:
                    frame = F.interpolate(
                        frame,
                        size=(self.height, self.width),
                        mode="bicubic",
                        align_corners=False,
                    )
                dummyTensor.copy_(
                    frame.mul(mul).clamp(0, mul).squeeze(0).permute(1, 2, 0),
                    non_blocking=False,
                )

            if self.channels == 1:
                # Should work for both 8bit and 16bit
                frame = dummyTensor.cpu().numpy()

            elif self.channels == 3:
                # for 8 bit, gotta convert the rgb24 -> yuv420p to save time on the subprocess write call.
                frame = (
                    cv2.cvtColor(dummyTensor.cpu().numpy(), cv2.COLOR_RGB2YUV_I420)
                    if self.bitDepth == "8bit"
                    else dummyTensor.cpu().numpy()
                )
            elif self.channels == 4:
                if self.bitDepth == "8bit":
                    frame = dummyTensor.cpu().numpy()
                else:
                    raise ValueError("RGBA 10bit encoding is not supported.")
            ffmpegSubprocess.stdin.write(np.ascontiguousarray(frame))
            # ffmpegSubprocess.stdin.flush()
            writtenFrames += 1

        self.frameQueue.put(None)
        logging.info(f"Encoded {writtenFrames} frames")

        if ffmpegSubprocess and ffmpegSubprocess.stdin:
            ffmpegSubprocess.stdin.close()
        if ffmpegSubprocess:
            ffmpegSubprocess.wait()
        if self.realtime and mpvSubprocess:
            mpvSubprocess.wait()

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
