import logging
import subprocess
import os
import torch
import numpy as np
import cv2
import threading
import json

from queue import Queue
from celux import VideoReader, Scale
from torch.nn import functional as F
from src.utils.logAndPrint import logAndPrint
from src.utils.encodingSettings import matchEncoder, getPixFMT

from .isCudaInit import CudaChecker

checker = CudaChecker()


def writeToSTDIN(
    command: list,
    frameQueue: Queue,
    mainPath: str,
    realtime: bool,
    input: str = "",
    mpvPath: str = "",
):
    """
    Pipes FFmpeg output to FFplay for real-time preview if realtime is True,
    otherwise just processes frames through FFmpeg

    command: list - The command to use for encoding the video.
    frameQueue: Queue - The queue to store frames.
    realtime: bool - Whether to preview the video in real-time using FFPLAY.
    input: str - The path to the input video file.
    mpvPath: str - The path to the FFplay executable.
    """
    try:
        ffmpegSubprocess = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            cwd=mainPath,
            shell=False,
        )

        if realtime:
            mpvSubprocess = subprocess.Popen(
                [
                    mpvPath,
                    "-",
                    "--no-terminal",
                    "--force-window=yes",
                    "--keep-open=no",
                    "--title=" + input,
                    "--no-config",
                    "--cache=yes",
                    "--demuxer-max-bytes=5M",
                ],
                stdin=ffmpegSubprocess.stdout,
                shell=False,
            )
            ffmpegSubprocess.stdout.close()

        while True:
            frame = frameQueue.get()
            if frame is None:
                break
            ffmpegSubprocess.stdin.write(np.ascontiguousarray(frame))
            ffmpegSubprocess.stdin.flush()

    except Exception as e:
        logAndPrint(f"Error during encoding / playback: {e}", "red")
        return
    finally:
        if ffmpegSubprocess and ffmpegSubprocess.stdin:
            ffmpegSubprocess.stdin.close()
        if ffmpegSubprocess:
            ffmpegSubprocess.wait()
        if realtime and mpvSubprocess:
            mpvSubprocess.wait()


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
        mainPath: str = "",
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
        self.decodeBuffer = Queue(maxsize=50)
        self.useOpenCV = False

        inputFramePoint = round(inpoint * fps)
        outputFramePoint = round(outpoint * fps) if outpoint != 0.0 else totalFrames
        if resize:
            filters = [Scale(width=str(width), height=str(height), flags=resizeMethod)]
        else:
            filters = []

        logging.info(f"Decoding frames from {inputFramePoint} to {outputFramePoint}")
        jsonMetadata = json.load(open(os.path.join(mainPath, "metadata.json"), "r"))

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
            while self.reader.isOpened():
                ret, frame = self.reader.read()
                if not ret or decodedFrames >= self.outputFramePoint:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
        if self.useOpenCV:
            frame = torch.from_numpy(frame)

        if frame.dtype == torch.uint8:
            mul = 1 / 255
        elif frame.dtype == torch.uint16:
            mul = 1 / 65535

        if checker.cudaAvailable:
            with torch.cuda.stream(normStream):
                if self.half:
                    frame = (
                        frame.to(device="cuda", non_blocking=True, dtype=torch.float16)
                        .mul(mul)
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                        .clamp(0, 1)
                    )
                else:
                    frame = (
                        frame.to(device="cuda", non_blocking=True, dtype=torch.float32)
                        .mul(mul)
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                        .clamp(0, 1)
                    )
            normStream.synchronize()
            return frame
        else:
            return frame.mul(mul).permute(2, 0, 1).unsqueeze(0).clamp(0, 1)

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
        mainPath: str = "",
        input: str = "",
        output: str = "",
        ffmpegPath: str = "",
        encode_method: str = "x264",
        custom_encoder: str = "",
        width: int = 1920,
        height: int = 1080,
        fps: float = 60.0,
        queueSize: int = 50,
        sharpen: bool = False,
        sharpen_sens: float = 0.0,
        grayscale: bool = False,
        transparent: bool = False,
        audio: bool = True,
        benchmark: bool = False,
        bitDepth: str = "8bit",
        inpoint: float = 0.0,
        outpoint: float = 0.0,
        realtime: bool = False,
    ):
        """
        A class meant to Pipe the input to FFMPEG from a queue.

        output: str - The path to the output video file.
        ffmpegPath: str - The path to the FFmpeg executable.
        encode_method: str - The method to use for encoding the video. Options include "x264", "x264_animation", "nvenc_h264", etc.
        custom_encoder: str - A custom encoder string to use for encoding the video.
        grayscale: bool - Whether to encode the video in grayscale.
        width: int - The width of the output video in pixels.
        height: int - The height of the output video in pixels.
        fps: float - The frames per second of the output video.
        sharpen: bool - Whether to apply a sharpening filter to the video.
        sharpen_sens: float - The sensitivity of the sharpening filter.
        queueSize: int - The size of the queue.
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
        self.ffmpegPath = os.path.normpath(ffmpegPath)
        self.encode_method = encode_method
        self.custom_encoder = custom_encoder
        self.grayscale = grayscale
        self.width = width
        self.height = height
        self.fps = fps
        self.sharpen = sharpen
        self.sharpen_sens = sharpen_sens
        self.queueSize = queueSize
        self.transparent = transparent
        self.audio = audio
        self.benchmark = benchmark
        self.bitDepth = bitDepth
        self.inpoint = inpoint
        self.outpoint = outpoint
        self.mainPath = mainPath
        self.realtime = realtime
        # ffmpeg path "C:\Users\User\AppData\Roaming\TheAnimeScripter\ffmpeg\ffmpeg.exe"
        self.mpvPath = os.path.join(os.path.dirname(self.ffmpegPath), "mpv.exe")

        self.writtenFrames = 0
        self.writeBuffer = Queue(maxsize=self.queueSize)

    def encodeSettings(self) -> list:
        """
        Simplified structure for setting input/output pix formats
        and building FFMPEG command.
        """

        os.environ["FFREPORT"] = "file=FFmpeg-Log.log:level=32"

        if "av1" in [self.encode_method, self.custom_encoder]:
            os.environ["SVT_LOG"] = "0"

        inputPixFormat, outputPixFormat, self.encode_method = getPixFMT(
            self.encode_method, self.bitDepth, self.grayscale, self.transparent
        )

        if self.benchmark:
            # if benchmarking is enabled, we skip all the shenaningans and just measure the fps.
            command = [
                self.ffmpegPath,
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

        else:
            command = [
                self.ffmpegPath,
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
                "-video_size",
                f"{self.width}x{self.height}",
                "-r",
                str(self.fps),
            ]
            if self.outpoint != 0:
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

            if self.audio:
                command.extend(["-i", self.input])
            command.extend(["-map", "0:v"])

            if not self.realtime:
                if not self.custom_encoder:
                    command.extend(matchEncoder(self.encode_method))
                    filters = []
                    if self.sharpen:
                        filters.append(f"cas={self.sharpen_sens}")
                    if self.grayscale:
                        filters.append("format=gray")
                    if self.transparent:
                        filters.append("format=yuva420p")
                    if filters:
                        command.extend(["-vf", ",".join(filters)])
                    command.extend(["-pix_fmt", outputPixFormat])
                else:
                    customEncoderList = self.custom_encoder.split()
                    if "-vf" in customEncoderList:
                        vfIndex = customEncoderList.index("-vf")
                        if self.sharpen:
                            customEncoderList[vfIndex + 1] += (
                                f",cas={self.sharpen_sens}"
                            )
                        if self.grayscale:
                            customEncoderList[vfIndex + 1] += (
                                ",format=gray"
                                if self.bitDepth == "8bit"
                                else ",format=gray16be"
                            )
                        if self.transparent:
                            customEncoderList[vfIndex + 1] += ",format=yuva420p"
                    else:
                        filters = []
                        if self.sharpen:
                            filters.append(f"cas={self.sharpen_sens}")
                        if self.grayscale:
                            filters.append(
                                "format=gray"
                                if self.bitDepth == "8bit"
                                else "format=gray16be"
                            )
                        if self.transparent:
                            filters.append("format=yuva420p")
                        if filters:
                            customEncoderList.extend(["-vf", ",".join(filters)])
                    if "-pix_fmt" not in customEncoderList:
                        logging.info(
                            f"-pix_fmt was not found, adding {outputPixFormat}."
                        )
                        customEncoderList.extend(["-pix_fmt", outputPixFormat])
                    command.extend(customEncoderList)

            if self.audio:
                command.extend(["-map", "1:a"])
                audioCodec = "copy"
                subCodec = "copy"
                if self.output.endswith(".webm"):
                    audioCodec = "libopus"
                    subCodec = "webvtt"
                command.extend(["-c:a", audioCodec, "-map", "1:s?", "-c:s", subCodec])

                if self.outpoint != 0:
                    command.extend(
                        ["-ss", str(self.inpoint), "-to", str(self.outpoint)]
                    )

            if self.realtime:
                command.extend(["-f", "matroska", "-"])
            else:
                command.append(self.output)

        return command

    def __call__(self):
        self.frameQueue = Queue(maxsize=self.queueSize)
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
            device="cpu",
        )

        waiterThread = threading.Thread(
            target=writeToSTDIN,
            args=(
                command,
                self.frameQueue,
                self.mainPath,
                self.realtime,
                self.input,
                self.mpvPath,
            ),
        )
        waiterThread.start()

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

        while True:
            frame = self.writeBuffer.get()
            if frame is None:
                break
            frame = frame.cpu()

            if NEEDSRESIZE:
                frame = F.interpolate(
                    frame,
                    size=(self.height, self.width),
                    mode="bicubic",
                    align_corners=False,
                )
            frame = frame.squeeze(0).permute(1, 2, 0).mul(mul).clamp(0, mul)
            dummyTensor.copy_(frame, non_blocking=False)

            if self.channels == 1:
                # Should work for both 8bit and 16bit
                frame = dummyTensor.numpy()

            elif self.channels == 3:
                # for 8 bit, gotta convert the rgb24 -> yuv420p to save time on the subprocess write call.
                frame = (
                    cv2.cvtColor(dummyTensor.numpy(), cv2.COLOR_RGB2YUV_I420)
                    if self.bitDepth == "8bit"
                    else dummyTensor.numpy()
                )
            elif self.channels == 4:
                if self.bitDepth == "8bit":
                    frame = dummyTensor.numpy()
                else:
                    raise ValueError("RGBA 10bit encoding is not supported.")
            self.frameQueue.put(frame)
            writtenFrames += 1

        self.frameQueue.put(None)
        logging.info(f"Encoded {writtenFrames} frames")

    def write(self, frame: torch.Tensor):
        """
        Add a frame to the queue. Must be in RGB format.
        """
        self.writeBuffer.put(frame)

    def close(self):
        """
        Close the queue.
        """
        self.writeBuffer.put(None)
