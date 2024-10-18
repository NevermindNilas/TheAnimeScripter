import logging
import subprocess
import os
import torch
import sys
import numpy as np
import cv2
import threading

from torch.multiprocessing import Process, Queue as MPQueue
from queue import Queue

workingFrames = 25


if getattr(sys, "frozen", False):
    outputPath = os.path.dirname(sys.executable)
else:
    outputPath = os.path.dirname(os.path.abspath(__file__))


def checkForCudaWorkflow() -> bool:
    try:
        isCudaAvailable = torch.cuda.is_available()
        torchVersion = torch.__version__
        cudaVersion = torch.version.cuda
    except Exception as e:
        logging.info(
            f"Couldn't check for CUDA availability, defaulting to CPU. Error: {e}"
        )
        isCudaAvailable = False
        torchVersion = "Unknown"
        cudaVersion = "Unknown"

    if isCudaAvailable:
        logging.info(
            f"CUDA is available, defaulting to full CUDA workflow. PyTorch version: {torchVersion}, CUDA version: {cudaVersion}"
        )
    else:
        logging.info(
            f"CUDA is not available, defaulting to CPU workflow. PyTorch version: {torchVersion}"
        )

    return isCudaAvailable


def matchEncoder(encode_method: str):
    """
    encode_method: str - The method to use for encoding the video. Options include "x264", "x264_animation", "nvenc_h264", etc.
    """
    command = []
    match encode_method:
        case "x264":
            command.extend(["-c:v", "libx264", "-preset", "veryfast", "-crf", "15"])
        case "slow_x264":
            command.extend(
                [
                    "-c:v",
                    "libx264",
                    "-preset",
                    "slow",
                    "-crf",
                    "18",
                    "-profile:v",
                    "high",
                    "-level",
                    "4.1",
                    "-tune",
                    "animation",
                    "-x264-params",
                    "ref=4:bframes=8:b-adapt=2:direct=auto:me=umh:subme=10:merange=24:trellis=2:deblock=-1,-1:psy-rd=1.00,0.15:aq-strength=1.0:rc-lookahead=60",
                    "-bf",
                    "3",
                    "-g",
                    "250",
                    "-keyint_min",
                    "25",
                    "-sc_threshold",
                    "40",
                    "-qcomp",
                    "0.6",
                    "-qmin",
                    "10",
                    "-qmax",
                    "51",
                    "-maxrate",
                    "5000k",
                    "-bufsize",
                    "10000k",
                    "-movflags",
                    "+faststart",
                ]
            )

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

        case "slow_x265":
            command.extend(
                [
                    "-c:v",
                    "libx265",
                    "-preset",
                    "slow",
                    "-crf",
                    "18",
                    "-profile:v",
                    "main",
                    "-level",
                    "5.1",
                    "-tune",
                    "ssim",
                    "-x265-params",
                    "ref=6:bframes=8:b-adapt=2:direct=auto:me=umh:subme=7:merange=57:rd=6:psy-rd=2.0:aq-mode=3:aq-strength=0.8:rc-lookahead=60",
                    "-bf",
                    "4",
                    "-g",
                    "250",
                    "-keyint_min",
                    "25",
                    "-sc_threshold",
                    "40",
                    "-qcomp",
                    "0.7",
                    "-qmin",
                    "10",
                    "-qmax",
                    "51",
                    "-maxrate",
                    "5000k",
                    "-bufsize",
                    "10000k",
                    "-movflags",
                    "+faststart",
                ]
            )
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

        case "slow_av1":
            command.extend(
                [
                    "-c:v",
                    "libsvtav1",
                    "-preset",
                    "4",
                    "-crf",
                    "30",
                    "-pix_fmt",
                    "yuv420p",
                    "-g",
                    "240",
                    "-keyint_min",
                    "23",
                    "-sc_threshold",
                    "40",
                    "-rc",
                    "vbr",
                    "-b:v",
                    "0",
                    "-maxrate",
                    "5000k",
                    "-bufsize",
                    "10000k",
                    "-tile-columns",
                    "2",
                    "-tile-rows",
                    "2",
                    "-row-mt",
                    "1",
                    "-movflags",
                    "+faststart",
                ]
            )
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
        case "h266":
            # Placeholder QP until I figure what the actual fuck is going on
            command.extend(["-c:v", "libvvenc", "-qp", "24", "-preset", "0"])

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

    def start(self):
        """
        The actual underlying logic for decoding, it starts a queue and gets the necessary FFMPEG command from decodeSettings.
        This is meant to be used in a separate thread for faster processing.

        queue : queue.Queue, optional - The queue to put the frames into. If None, a new queue will be created.
        """
        self.readBuffer = Queue(maxsize=self.queueSize)
        command = self.decodeSettings()

        logging.info(f"Decoding options: {' '.join(map(str, command))}")

        self.isCudaAvailable = checkForCudaWorkflow()

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
        chunkExecutor = threading.Thread(
            target=self.convertFrames, args=(reshape, self.isCudaAvailable)
        )
        readExecutor = threading.Thread(target=self.readSTDOUT, args=(chunk,))
        chunkExecutor.start()
        readExecutor.start()

        chunkExecutor.join()
        readExecutor.join()

        logging.info(f"Built buffer with {self.decodedFrames} frames")
        self.readingDone = True
        self.readBuffer.put(None)
        self.process.stdout.close()

    def readSTDOUT(self, chunk):
        for _ in range(self.totalFrames):
            rawFrame = self.process.stdout.read(chunk)
            self.chunkQueue.put(rawFrame)

    @torch.inference_mode()
    def convertFrames(self, reshape, isCudaAvailable):
        dummyTensor = torch.zeros(
            (self.height, self.width, 3), dtype=torch.uint8, device="cpu"
        )

        if isCudaAvailable:
            self.normStream = torch.cuda.Stream()
            dummyTensor = dummyTensor.pin_memory()

        for _ in range(self.totalFrames):
            dummyTensor.copy_(
                torch.from_numpy(
                    cv2.cvtColor(
                        np.frombuffer(self.chunkQueue.get(), dtype=np.uint8).reshape(
                            reshape
                        ),
                        cv2.COLOR_YUV2RGB_I420,
                    )
                )
            )
            if self.isCudaAvailable:
                with torch.cuda.stream(self.normStream):
                    self.readBuffer.put(
                        dummyTensor.to(device="cuda", non_blocking=True)
                    )
                    self.normStream.synchronize()

            else:
                self.readBuffer.put(dummyTensor)

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
        ffmpegLogPath = os.path.join(mainPath, "ffmpeg.log")

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
        dimsList = [self.height, self.width, self.channels]

        self.torchArray = torch.zeros(
            *dimensions.tolist(), dtype=torch.uint8
        ).share_memory_()

        try:
            self.torchArray = self.torchArray.cuda()
            self.isCudaAvailable = True
            self.normStream = torch.cuda.Stream()
        except Exception:
            self.isCudaAvailable = False
            pass

        self.process = Process(
            target=self.childProcessEncode,
            args=(
                self.torchArray,
                self.processQueue,
                dimsList,
                self.command,
                self.bitDepth,
                self.channels,
                self.isCudaAvailable,
                ffmpegLogPath,
            ),
        )

    def encodeSettings(self) -> list:
        """
        This will return the command for FFMPEG to work with, it will be used inside of the scope of the class.

        """
        if self.bitDepth == "8bit":
            inputPixFormat = "yuv420p"
            outputPixFormat = "yuv420p"
        else:
            inputPixFormat = "rgb48le"
            outputPixFormat = "yuv444p10le"

        if self.transparent:
            if self.encode_method not in ["prores_segment"]:
                logging.info("Switching internally to prores for transparency support")
                self.encode_method = "prores_segment"

                inputPixFormat = "rgba"
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
                "verbose",
                "-stats",
                "-f",
                "rawvideo",
                "-pixel_format",
                f"{inputPixFormat}",
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
                command.extend(
                    [
                        "-i",
                        "pipe:0",
                    ]
                )

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
                    ]
                )

                if self.output.endswith(".mp4"):
                    command.extend(
                        [
                            "-map",
                            "1:s?",
                            "-c:s",
                            "srt",
                        ]
                    )
                elif self.output.endswith(".webm"):
                    command.extend(
                        [
                            "-map",
                            "1:s?",
                            "-c:s",
                            "webvtt",
                        ]
                    )
                else:
                    command.extend(
                        [
                            "-map",
                            "1:s?",
                            "-c:s",
                            "srt",
                        ]
                    )

                command.append("-shortest")

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

    def start(self):
        self.process.start()

    @staticmethod
    def childProcessEncode(
        sharedTensor,
        processQueue: MPQueue,
        dimsList,
        command,
        bitDepth: str = "8bit",
        channels: int = 3,
        isCudaAvailable: bool = False,
        ffmpegLogPath: str = "",
    ):
        dummyTensor = torch.zeros(dimsList, dtype=torch.uint8, device="cpu")

        if isCudaAvailable:
            dummyTensor = dummyTensor.pin_memory()

        try:
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

                        dummyTensor.copy_(sharedTensor[dataID], non_blocking=False)

                        if channels == 1:
                            frame = dummyTensor.numpy()

                        if channels == 3:
                            if bitDepth == "8bit":
                                frame = cv2.cvtColor(
                                    dummyTensor.numpy(), cv2.COLOR_RGB2YUV_I420
                                )
                            else:
                                frame = dummyTensor.numpy()

                        elif channels == 4:
                            frame = dummyTensor.numpy()

                        if bitDepth == "8bit":
                            frame = frame.tobytes()
                        else:
                            frame = np.ascontiguousarray(
                                (
                                    (frame.astype(np.float32) * 257)
                                    .astype(np.uint16)
                                    .tobytes()
                                )
                            )

                        process.stdin.write(frame)
        except Exception as e:
            logging.exception(f"Error encoding frame: {e}")

    @torch.inference_mode()
    def write(self, frame: torch.Tensor):
        """
        Add a frame to the queue. Must be in RGB format.
        """
        if self.isCudaAvailable:
            dataID = self.writtenFrames % workingFrames
            with torch.cuda.stream(self.normStream):
                frame = frame.mul(255)
                self.torchArray[dataID].copy_(frame, non_blocking=False)
            self.normStream.synchronize()
            self.processQueue.put(dataID)
            self.writtenFrames += 1
        else:
            dataID = self.writtenFrames % workingFrames
            self.torchArray[dataID].copy_(frame.mul(255), non_blocking=False)
            self.processQueue.put(dataID)
            self.writtenFrames += 1

    def close(self):
        """
        Close the queue.
        """
        self.processQueue.put(None)
        self.isWritingDone = True
        self.process.join()
        logging.info(f"Encoded {self.writtenFrames} frames")
