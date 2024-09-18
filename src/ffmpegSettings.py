import logging
import subprocess
import os
import torch
import sys
import numpy as np
import cv2
import platform
import threading
import re

from src.coloredPrints import green
from torch.multiprocessing import Process
from multiprocessing import Queue as MPQueue
from queue import Full, Queue

workingFrames = 100

if platform.system() == "Windows":
    appdata = os.getenv("APPDATA")
    mainPath = os.path.join(appdata, "TheAnimeScripter")

    if not os.path.exists(mainPath):
        os.makedirs(mainPath)
else:
    xdg_config_home = os.getenv("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
    mainPath = os.path.join(xdg_config_home, "TheAnimeScripter")

    if not os.path.exists(mainPath):
        os.makedirs(mainPath)

ffmpegLogPath = os.path.join(mainPath, "ffmpegLog.txt")

if getattr(sys, "frozen", False):
    outputPath = os.path.dirname(sys.executable)
else:
    outputPath = os.path.dirname(os.path.abspath(__file__))


def checkForCudaWorkflow(verbose: bool = True) -> bool:
    try:
        isCudaAvailable = torch.cuda.is_available()
    except Exception:
        if verbose:
            logging.info("Couldn't check for CUDA availability, defaulting to CPU")
        isCudaAvailable = False

    if isCudaAvailable:
        if verbose:
            logging.info("CUDA is available, defaulting to full CUDA workflow")
    else:
        if verbose:
            logging.info("CUDA is not available, defaulting to CPU workflow")

    return isCudaAvailable


def matchEncoder(encode_method: str):
    """
    encode_method: str - The method to use for encoding the video. Options include "x264", "x264_animation", "nvenc_h264", etc.
    """
    command = []
    match encode_method:
        case "x264":
            command.extend(["-c:v", "libx264", "-preset", "ultrafast", "-crf", "50"])
        case "x264_10bit":
            command.extend(
                [
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "15",
                    "-profile:v",
                    "high10",
                ]
            )
        case "x264_animation":
            command.extend(
                [
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-tune",
                    "animation",
                    "-crf",
                    "15",
                ]
            )
        case "x264_animation_10bit":
            command.extend(
                [
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-tune",
                    "animation",
                    "-crf",
                    "15",
                    "-profile:v",
                    "high10",
                ]
            )
        case "x265":
            command.extend(["-c:v", "libx265", "-preset", "veryfast", "-crf", "15"])
        case "x265_10bit":
            command.extend(
                [
                    "-c:v",
                    "libx265",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "15",
                    "-profile:v",
                    "main10",
                ]
            )
        case "nvenc_h264":
            command.extend(["-c:v", "h264_nvenc", "-preset", "p1", "-cq", "15"])
        case "nvenc_h265":
            command.extend(["-c:v", "hevc_nvenc", "-preset", "p1", "-cq", "15"])
        case "nvenc_h265_10bit":
            command.extend(
                [
                    "-c:v",
                    "hevc_nvenc",
                    "-preset",
                    "p1",
                    "-cq",
                    "15",
                    "-profile:v",
                    "main10",
                ]
            )
        case "qsv_h264":
            command.extend(
                ["-c:v", "h264_qsv", "-preset", "veryfast", "-global_quality", "15"]
            )
        case "qsv_h265":
            command.extend(
                ["-c:v", "hevc_qsv", "-preset", "veryfast", "-global_quality", "15"]
            )
        case "qsv_h265_10bit":
            command.extend(
                [
                    "-c:v",
                    "hevc_qsv",
                    "-preset",
                    "veryfast",
                    "-global_quality",
                    "15",
                    "-profile:v",
                    "main10",
                ]
            )
        case "nvenc_av1":
            command.extend(["-c:v", "av1_nvenc", "-preset", "p1", "-cq", "15"])
        case "av1":
            command.extend(["-c:v", "libsvtav1", "-preset", "8", "-crf", "15"])
        case "h264_amf":
            command.extend(
                ["-c:v", "h264_amf", "-quality", "speed", "-rc", "cqp", "-qp", "15"]
            )
        case "hevc_amf":
            command.extend(
                ["-c:v", "hevc_amf", "-quality", "speed", "-rc", "cqp", "-qp", "15"]
            )
        case "hevc_amf_10bit":
            command.extend(
                [
                    "-c:v",
                    "hevc_amf",
                    "-quality",
                    "speed",
                    "-rc",
                    "cqp",
                    "-qp",
                    "15",
                    "-profile:v",
                    "main10",
                ]
            )
        case "prores" | "prores_segment":
            command.extend(["-c:v", "prores_ks", "-profile:v", "4", "-qscale:v", "15"])
        case "gif":
            command.extend(["-c:v", "gif", "-qscale:v", "1", "-loop", "0"])
        case "image":
            command.extend(["-c:v", "png", "-q:v", "1"])
        case "vp9":
            command.extend(["-c:v", "libvpx-vp9", "-crf", "15", "-preset", "veryfast"])
        case "qsv_vp9":
            command.extend(["-c:v", "vp9_qsv", "-preset", "veryfast"])

    return command


class BuildBuffer:
    def __init__(
        self,
        input: str = "",
        ffmpegPath: str = "",
        inpoint: float = 0.0,
        outpoint: float = 0.0,
        dedup: bool = False,
        dedupSens: float = 0.0,
        dedupMethod: str = "ssim",
        width: int = 1920,
        height: int = 1080,
        resize: bool = False,
        resizeMethod: str = "bilinear",
        buffSize: int = 10**8,
        queueSize: int = 50,
        totalFrames: int = 0,
    ):
        """
        A class meant to Pipe the Output of FFMPEG into a Queue for further processing.

        input: str - The path to the input video file.
        ffmpegPath: str - The path to the FFmpeg executable.
        inpoint: float - The start time of the segment to decode, in seconds.
        outpoint: float - The end time of the segment to decode, in seconds.
        dedup: bool - Whether to apply a deduplication filter to the video.
        dedupSens: float - The sensitivity of the deduplication filter.
        width: int - The width of the output video in pixels.
        height: int - The height of the output video in pixels.
        resize: bool - Whether to resize the video.
        resizeMethod: str - The method to use for resizing the video. Options include: "fast_bilinear", "bilinear", "bicubic", "experimental", "neighbor", "area", "bicublin", "gauss", "sinc", "lanczos",
        "spline",
        buffSize: int - The size of the subprocess buffer in bytes, don't touch unless you are working with some ginormous 8K content.
        queueSize: int - The size of the queue.
        totalFrames: int - The total amount of frames to decode.
        """
        self.input = os.path.normpath(input)
        self.ffmpegPath = os.path.normpath(ffmpegPath)
        self.inpoint = inpoint
        self.outpoint = outpoint
        self.dedup = dedup
        self.dedupSens = dedupSens
        self.dedupeMethod = dedupMethod
        self.resize = resize
        self.width = width
        self.height = height
        self.resizeMethod = resizeMethod
        self.buffSize = buffSize
        self.queueSize = queueSize
        self.totalFrames = totalFrames

    def decodeSettings(self) -> list:
        """
        This returns a command for FFMPEG to work with, it will be used inside of the scope of the class.

        input: str - The path to the input video file.
        inpoint: float - The start time of the segment to decode, in seconds.
        outpoint: float - The end time of the segment to decode, in seconds.
        dedup: bool - Whether to apply a deduplication filter to the video.
        dedup_strenght: float - The strength of the deduplication filter.
        ffmpegPath: str - The path to the FFmpeg executable.
        resize: bool - Whether to resize the video.
        resizeMethod: str - The method to use for resizing the video. Options include: "fast_bilinear", "bilinear", "bicubic", "experimental", "neighbor", "area", "bicublin", "gauss", "sinc", "lanczos",
        "spline",
        """
        command = [
            self.ffmpegPath,
            "-vsync",
            "0",
        ]

        if self.outpoint != 0:
            command.extend(["-ss", str(self.inpoint), "-to", str(self.outpoint)])

        command.extend(
            [
                "-i",
                self.input,
            ]
        )

        filters = []
        if self.resize:
            if self.resizeMethod in ["spline16", "spline36", "point"]:
                filters.append(
                    f"zscale={self.width}:{self.height}:filter={self.resizeMethod}"
                )
            else:
                filters.append(
                    f"scale={self.width}:{self.height}:flags={self.resizeMethod}"
                )

        if filters:
            command.extend(["-vf", ",".join(filters)])

        command.extend(["-f", "rawvideo", "-pix_fmt", "yuv420p", "-"])

        return command

    def start(self, queue: Queue = None):
        """
        The actual underlying logic for decoding, it starts a queue and gets the necessary FFMPEG command from decodeSettings.
        This is meant to be used in a separate thread for faster processing.

        queue : queue.Queue, optional - The queue to put the frames into. If None, a new queue will be created.
        verbose : bool - Whether to log the progress of the decoding.
        """
        self.readBuffer = queue if queue is not None else Queue(maxsize=self.queueSize)
        verbose = True
        command = self.decodeSettings()

        if verbose:
            logging.info(f"Decoding options: {' '.join(map(str, command))}")

        self.isCudaAvailable = checkForCudaWorkflow(verbose=True)

        try:
            yPlane = self.width * self.height
            uPlane = (self.width // 2) * (self.height // 2)
            vPlane = (self.width // 2) * (self.height // 2)
            reshape = (self.height * 3 // 2, self.width)
            chunk = yPlane + uPlane + vPlane
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            self.decodedFrames = 0
            self.readingDone = False

            self.chunkQueue = Queue(maxsize=self.queueSize)

            chunkExecutor = threading.Thread(target=self.convertFrames, args=(reshape,))
            readExecutor = threading.Thread(target=self.readSTDOUT, args=(chunk,))

            chunkExecutor.start()
            readExecutor.start()

            chunkExecutor.join()
            readExecutor.join()

        except Exception as e:
            logging.error(f"An error occurred: {str(e)}")
        finally:
            if verbose:
                logging.info(f"Built buffer with {self.decodedFrames} frames")
            self.readingDone = True
            self.readBuffer.put(None)
            self.process.stdout.close()

    def readSTDOUT(self, chunk):
        for _ in range(self.totalFrames):
            rawFrame = self.process.stdout.read(chunk)
            try:
                self.chunkQueue.put_nowait(rawFrame)
            except Full:
                self.chunkQueue.put(rawFrame)

    def convertFrames(self, reshape):
        for _ in range(self.totalFrames):
            rawFrame = self.chunkQueue.get()
            frame = torch.from_numpy(
                cv2.cvtColor(
                    np.frombuffer(rawFrame, dtype=np.uint8).reshape(reshape),
                    cv2.COLOR_YUV2RGB_I420,
                )
            )
            if self.isCudaAvailable:
                frame = frame.to(device="cuda", non_blocking=True)
            try:
                self.readBuffer.put_nowait(frame)
            except Full:
                self.readBuffer.put(frame)
            self.decodedFrames += 1

    def read(self):
        """
        Returns a torch array in RGB format.
        """
        return self.readBuffer.get()

    def isReadingDone(self):
        """
        Check if the reading is done, safelock for the queue environment.
        """
        return self.readingDone

    def getDecodedFrames(self):
        """
        Get the amount of processed frames.
        """
        return self.decodedFrames

    def getSizeOfQueue(self):
        """
        Get the size of the queue.
        """
        return self.readBuffer.qsize()

    def getTotalFrames(self):
        """
        Get the total amount of frames.
        """
        return self.totalFrames


class WriteBuffer:
    def __init__(
        self,
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
        preview: bool = False,
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
        preview: bool - Whether to preview the video.
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
        self.preview = preview

        self.writtenFrames = 0
        self.processQueue = MPQueue(maxsize=(workingFrames - 2))

        self.command = self.encodeSettings()

        self.latestFrame = None

        if self.grayscale:
            self.channels = 1
        elif self.transparent:
            self.channels = 4
        else:
            self.channels = 3

        logging.info(f"Encoding options: {' '.join(map(str, self.command))}")

        dimensions = torch.tensor(
            [workingFrames, self.height, self.width, self.channels]
        )
        self.torchArray = torch.zeros(
            *dimensions.tolist(), dtype=torch.uint8
        ).share_memory_()

        try:
            self.torchArray = self.torchArray.cuda()
        except Exception:
            pass

        self.process = Process(
            target=self.childProcessEncode,
            args=(
                self.torchArray,
                self.processQueue,
                dimensions,
                self.command,
                self.bitDepth,
                self.channels,
            ),
        )

    def encodeSettings(self, verbose: bool = False) -> list:
        """
        This will return the command for FFMPEG to work with, it will be used inside of the scope of the class.

        verbose : bool - Whether to log the progress of the encoding.
        """
        if self.bitDepth == "8bit":
            inputPixFormat = "yuv420p"
            outputPixFormat = "yuv420p"
        else:
            inputPixFormat = "rgb48le"
            outputPixFormat = "yuv444p10le"

        if self.transparent:
            if self.encode_method not in ["prores_segment"]:
                if verbose:
                    logging.info(
                        "Switching internally to prores for transparency support"
                    )
                self.encode_method = "prores_segment"

                inputPixFormat = "yuva420p"
                outputPixFormat = "yuva444p10le"

        elif self.grayscale:
            if self.bitDepth == "8bit":
                inputPixFormat = "gray"
                outputPixFormat = "yuv420p"
            else:
                inputPixFormat = "gray16le"
                outputPixFormat = "yuv444p10le"

        elif self.encode_method in ["x264_10bit", "x265_10bit", "x264_animation_10bit"]:
            if self.bitDepth == "8bit":
                inputPixFormat = "yuv420p"
                outputPixFormat = "yuv420p10le"
            else:
                inputPixFormat = "rgb48le"
                outputPixFormat = "yuv420p10le"

        elif self.encode_method in [
            "nvenc_h265_10bit",
            "hevc_amf_10bit",
            "qsv_h265_10bit",
        ]:
            if self.bitDepth == "8bit":
                inputPixFormat = "yuv420p"
                outputPixFormat = "p010le"
            else:
                inputPixFormat = "rgb48le"
                outputPixFormat = "p010le"

        elif self.encode_method in ["prores"]:
            if self.bitDepth == "8bit":
                inputPixFormat = "yuv420p"
                outputPixFormat = "yuv444p10le"
            else:
                inputPixFormat = "rgb48le"
                outputPixFormat = "yuv444p10le"

        if not self.benchmark:
            command = [
                self.ffmpegPath,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-stats",
                "-f",
                "rawvideo",
                "-pixel_format",
                f"{inputPixFormat}",
                "-video_size",
                f"{self.width}x{self.height}",
                "-r",
                str(self.fps),
                "-i",
                "pipe:0",
            ]
            if self.audio:
                command.extend(
                    [
                        "-i",
                        self.input,
                    ]
                )

            command.extend(
                [
                    "-map",
                    "0:v",
                ]
            )

            if not self.custom_encoder:
                command.extend(matchEncoder(self.encode_method))

                filters = []
                if self.sharpen:
                    filters.append("cas={}".format(self.sharpen_sens))
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
                        customEncoderList[vfIndex + 1] += ",cas={}".format(
                            self.sharpen_sens
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
                        filters.append("cas={}".format(self.sharpen_sens))
                    if self.grayscale:
                        customEncoderList[vfIndex + 1] += (
                            ",format=gray"
                            if self.bitDepth == "8bit"
                            else ",format=gray16be"
                        )
                    if self.transparent:
                        filters.append("format=yuva420p")
                    if filters:
                        customEncoderList.extend(["-vf", ",".join(filters)])

                if "-pix_fmt" not in customEncoderList:
                    logging.info(
                        f"-pix_fmt was not found in the custom encoder list, adding {outputPixFormat}, for future reference, it is recommended to add it."
                    )
                    customEncoderList.extend(["-pix_fmt", outputPixFormat])

                command.extend(customEncoderList)

            if self.audio:
                command.extend(
                    [
                        "-map",
                        "1:a",
                        "-c:a",
                        "copy",
                        "-map",
                        "1:s?",
                        "-c:s",
                        "copy",
                        "-shortest",
                    ]
                )
            command.extend([self.output])

        else:
            command = [
                self.ffmpegPath,
                "-y",
                "-hide_banner",
                "-v",
                "warning",
                "-stats",
                "-f",
                "rawvideo",
                "-video_size",
                f"{self.width}x{self.height}",
                "-pix_fmt",
                f"{inputPixFormat}",
                "-r",
                str(self.fps),
                "-i",
                "-",
                "-benchmark",
                "-f",
                "null",
                "-",
            ]

        return command

    def start(self, queue: Queue = None):
        self.process.start()

    @staticmethod
    def childProcessEncode(
        sharedTensor,
        processQueue: MPQueue,
        torchShape,
        command,
        bitDepth: str = "8bit",
        channels: int = 3,
    ):
        with open(ffmpegLogPath, "w") as logPath:
            with subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stderr=logPath,
                stdout=logPath,
            ) as process:
                while True:
                    dataID = processQueue.get()
                    if dataID is None:
                        break

                    frame = sharedTensor[dataID].cpu().numpy()

                    if channels == 3:
                        if bitDepth == "8bit":
                            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV_I420)
                    elif channels == 4:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2YUVA_I420)

                    if bitDepth == "8bit":
                        frame = frame.tobytes()
                    else:
                        frame = frame.astype(np.float32) * 257
                        frame = frame.astype(np.uint16).tobytes()

                    process.stdin.write(frame)

        with open(ffmpegLogPath, "r") as logPath:
            lines = logPath.readlines()
            pattern = re.compile(r"fps=\s*(\d+)")

            for line in reversed(lines):
                match = pattern.search(line)
                if match:
                    fps = float(match.group(1))
                    fpsStr = f"{fps:5.2f} FPS"
                    message = f"FFMPEG Reported FPS: {fpsStr}"
                    totalLength = len(message)
                    boxWidth = 42
                    paddingLeft = (boxWidth - totalLength) // 2
                    paddingRight = boxWidth - totalLength - paddingLeft

                    print(
                        green(
                            f"\n┌{'─'*boxWidth}┐\n"
                            f"│{' '*boxWidth}│\n"
                            f"│{' '*paddingLeft}{message}{' '*paddingRight}│\n"
                            f"│{' '*boxWidth}│\n"
                            f"└{'─'*boxWidth}┘\n"
                        )
                    )
                    break

    def write(self, frame: torch.Tensor):
        """
        Add a frame to the queue. Must be in RGB format.
        """
        try:
            dataID = self.writtenFrames % workingFrames
            self.torchArray[dataID].copy_(frame, non_blocking=True)
            self.processQueue.put(dataID)
            self.writtenFrames += 1
        except Exception as e:
            logging.error(f"An error occurred: {str(e)}")

    def close(self):
        """
        Close the queue.
        """
        self.processQueue.put(None)
        self.isWritingDone = True
        self.process.join()
