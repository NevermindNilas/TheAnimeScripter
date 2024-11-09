import logging
import subprocess
import os
import torch
import numpy as np
import cv2
from celux import VideoReader, Scale
import threading

from queue import Queue

ISCUDA = torch.cuda.is_available()


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
        case "slow_nvenc_h264":
            command.extend(
                [
                    "-c:v",
                    "h264_nvenc",
                    "-preset",
                    "p7",
                    "-cq",
                    "15",
                    "-rc",
                    "vbr_hq",
                    "-b:v",
                    "0",
                    "-maxrate",
                    "5000k",
                    "-bufsize",
                    "10000k",
                    "-g",
                    "240",
                    "-keyint_min",
                    "23",
                    "-sc_threshold",
                    "40",
                    "-spatial_aq",
                    "1",
                    "-temporal_aq",
                    "1",
                    "-aq-strength",
                    "15",
                    "-rc-lookahead",
                    "60",
                    "-surfaces",
                    "64",
                    "-gpu",
                    "all",
                    "-movflags",
                    "+faststart",
                ]
            )
        case "nvenc_h265":
            command.extend(["-c:v", "hevc_nvenc", "-preset", "p1", "-cq", "15"])

        case "slow_nvenc_h265":
            command.extend(
                [
                    "-c:v",
                    "hevc_nvenc",
                    "-preset",
                    "p7",
                    "-cq",
                    "15",
                    "-rc",
                    "vbr_hq",
                    "-b:v",
                    "0",
                    "-maxrate",
                    "5000k",
                    "-bufsize",
                    "10000k",
                    "-g",
                    "240",
                    "-keyint_min",
                    "23",
                    "-sc_threshold",
                    "40",
                    "-spatial_aq",
                    "1",
                    "-temporal_aq",
                    "1",
                    "-aq-strength",
                    "15",
                    "-rc-lookahead",
                    "60",
                    "-surfaces",
                    "64",
                    "-gpu",
                    "all",
                    "-movflags",
                    "+faststart",
                    "-bf",
                    "2",
                ]
            )
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
            reader (celux.VideoReader): Video reader object for decoding frames.
        """
        self.half = half
        self.decodeBuffer = Queue(maxsize=50)

        inputFramePoint = round(inpoint * fps)
        outputFramePoint = round(outpoint * fps) if outpoint != 0.0 else totalFrames
        if resize:
            filters = [Scale(width=str(width), height=str(height), flags=resizeMethod)]
        else:
            filters = []

        logging.info(f"Decoding frames from {inputFramePoint} to {outputFramePoint}")
        self.reader = VideoReader(
            videoInput, device="cpu", num_threads=decodeThreads, filters=filters
        )([inputFramePoint, outputFramePoint])

    def __call__(self):
        """
        Decodes frames from the video and stores them in the decodeBuffer.
        """
        decodedFrames = 0
        self.isFinished = False

        try:
            if ISCUDA:
                normStream = torch.cuda.Stream()

            for frame in self.reader:
                frame = self.processFrame(frame, normStream if ISCUDA else None)
                self.decodeBuffer.put(frame)
                decodedFrames += 1

            self.isFinished = True
            logging.info(f"Decoded {decodedFrames} frames")
        except Exception as e:
            logging.error(f"Error during frame decoding: {e}")
            self.isFinished = True

    def processFrame(self, frame, normStream=None):
        """
        Processes a single frame.

        Args:
            frame: The frame to process.
            normStream: The CUDA stream for normalization (if applicable).

            Returns:
                The processed frame.
        """
        mul = 1 / 255 if frame.dtype == torch.uint8 else 1 / 65535

        if ISCUDA:
            with torch.cuda.stream(normStream):
                if self.half:
                    frame = (
                        frame.to(device="cuda", non_blocking=True, dtype=torch.float16)
                        .mul(mul)
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                    )
                else:
                    frame = (
                        frame.to(device="cuda", non_blocking=True, dtype=torch.float32)
                        .mul(mul)
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                    )
            normStream.synchronize()
            return frame
        else:
            return frame.mul(mul).permute(2, 0, 1).unsqueeze(0)

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
        self.mainPath = mainPath

        self.writtenFrames = 0
        self.writeBuffer = Queue(maxsize=self.queueSize)

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
                command.extend(["-map", "1:a"])

                audioCodec = "copy"
                subCodec = "srt"

                if self.output.endswith(".webm"):
                    audioCodec = "libopus"
                    subCodec = "webvtt"

                command.extend(
                    [
                        "-c:a",
                        audioCodec,
                        "-map",
                        "1:s?",
                        "-c:s",
                        subCodec,
                        "-shortest",
                    ]
                )

            command.append(self.output)

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
        command = self.encodeSettings()
        logging.info(f"Encoding options: {' '.join(map(str, command))}")
        ffmpegLogPath = os.path.join(self.mainPath, "ffmpeg.log")

        if self.grayscale:
            self.channels = 1
        elif self.transparent:
            self.channels = 4
        else:
            self.channels = 3

        self.frameQueue = Queue(maxsize=self.queueSize)

        def writeToStdin():
            with open(ffmpegLogPath, "w") as logFile:
                with subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=logFile,
                    stderr=subprocess.STDOUT,
                ) as process:
                    while True:
                        frame = self.frameQueue.get()
                        if frame is None:
                            break
                        process.stdin.write(np.ascontiguousarray(frame))
                        process.stdin.flush()

            process.stdin.close()
            process.wait()

        writtenFrames = 0
        dtype = torch.uint8 if self.bitDepth == "8bit" else torch.uint16
        mul = 255 if self.bitDepth == "8bit" else 65535

        dummyTensor = torch.zeros(
            (self.height, self.width, self.channels),
            dtype=dtype,
            device="cuda" if ISCUDA else "cpu",
        )

        if ISCUDA:
            normStream = torch.cuda.Stream()
        else:
            try:
                dummyTensor = dummyTensor.pin_memory()
            except Exception:
                pass

        waiterThread = threading.Thread(target=writeToStdin)
        waiterThread.start()

        while True:
            frame = self.writeBuffer.get()
            if frame is None:
                break
            if ISCUDA:
                with torch.cuda.stream(normStream):
                    frame = frame.mul(mul).clamp(0, mul).squeeze(0).permute(1, 2, 0)
                    dummyTensor.copy_(frame, non_blocking=True)
                normStream.synchronize()
            else:
                frame = frame.mul(mul).clamp(0, mul).squeeze(0).permute(1, 2, 0)
                dummyTensor.copy_(frame, non_blocking=False)

            if self.channels == 1:
                frame = (
                    dummyTensor.cpu().numpy().tobytes()
                    if self.bitDepth == "8bit"
                    else (dummyTensor.cpu().numpy().tobytes())
                )
            elif self.channels == 3:
                frame = (
                    cv2.cvtColor(dummyTensor.cpu().numpy(), cv2.COLOR_RGB2YUV_I420)
                    if self.bitDepth == "8bit"
                    else dummyTensor.cpu().numpy().tobytes()
                )
            elif self.channels == 4:
                if self.bitDepth == "8bit":
                    frame = dummyTensor.cpu().numpy().tobytes()
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
