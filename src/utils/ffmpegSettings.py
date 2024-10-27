import logging
import subprocess
import os
import torch
import sys
import numpy as np
import cv2
import celux
import threading

from queue import Queue

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


def processChunk(pixFmt, chunkQueue, reshape):
    if pixFmt in ["unknown", "yuv420p8le", "yuv422p8le"]:
        return cv2.cvtColor(
            np.frombuffer(chunkQueue.get(), dtype=np.uint8).reshape(reshape),
            cv2.COLOR_YUV2RGB_I420,
        )
    elif pixFmt == "yuv420p10le":
        return cv2.cvtColor(
            ((np.frombuffer(chunkQueue.get(), dtype=np.uint16) + 2) >> 2)
            .astype(np.uint8)
            .reshape(reshape),
            cv2.COLOR_YUV2RGB_I420,
        )


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
        videoInput: str = "",
        inpoint: float = 0.0,
        outpoint: float = 0.0,
        totalFrames: int = 0,
        fps: float = 0,
    ):
        self.isCudaAvailable = checkForCudaWorkflow()
        self.decodeBuffer = Queue(maxsize=50)

        inputFramePoint = round(inpoint * fps)
        outputFramePoint = round(outpoint * fps) if outpoint != 0.0 else totalFrames

        logging.info(f"Decoding frames from {inputFramePoint} to {outputFramePoint}")
        self.reader = celux.VideoReader(videoInput, device="cpu")(
            [inputFramePoint, outputFramePoint]
        )

    def __call__(self):
        decodedFrames = 0
        for frame in self.reader:
            frame = (
                frame.cuda().mul(1 / 255)
                if self.isCudaAvailable
                else frame.mul(1 / 255)
            )
            self.decodeBuffer.put(frame)
            decodedFrames += 1

        logging.info(f"Decoded {decodedFrames} frames")

    def read(self):
        return self.decodeBuffer.get()


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
        isCudaAvailable = checkForCudaWorkflow()

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

        try:
            writtenFrames = 0
            if self.bitDepth == "8bit":
                dummyTensor = torch.zeros(
                    (self.height, self.width, self.channels),
                    dtype=torch.uint8,
                    device="cpu",
                )
            else:
                dummyTensor = torch.zeros(
                    (self.height, self.width, self.channels),
                    dtype=torch.uint16,
                    device="cpu",
                )
            try:
                dummyTensor = dummyTensor.pin_memory()
                if isCudaAvailable:
                    dummyTensor = dummyTensor.cuda()
                    normStream = torch.cuda.Stream()
            except Exception as e:
                logging.info(f"Couldn't pin memory, defaulting to CPU. Error: {e}")

            waiterTread = threading.Thread(target=writeToStdin)
            waiterTread.start()

            while True:
                if isCudaAvailable:
                    with torch.cuda.stream(normStream):
                        dummyTensor.copy_(
                            self.writeBuffer.get().mul(255.0).clamp(0, 255),
                            non_blocking=False,
                        )
                    torch.cuda.synchronize()
                else:
                    dummyTensor.copy_(
                        self.writeBuffer.get().mul(255.0).clamp(0, 255),
                        non_blocking=False,
                    )
                if self.channels == 1:
                    if self.bitDepth == "8bit":
                        frame = dummyTensor.to(torch.uint8).cpu().numpy().tobytes()
                    else:
                        frame = (
                            dummyTensor.to(torch.float32)
                            .mul(257)
                            .to(torch.uint16)
                            .cpu()
                            .numpy()
                            .tobytes()
                        )
                elif self.channels == 3:
                    if self.bitDepth == "8bit":
                        frame = cv2.cvtColor(
                            dummyTensor.to(torch.uint8).cpu().numpy(),
                            cv2.COLOR_RGB2YUV_I420,
                        )
                    else:
                        frame = (
                            dummyTensor.to(torch.float32)
                            .mul(257)
                            .to(torch.uint16)
                            .cpu()
                            .numpy()
                            .tobytes()
                        )
                elif self.channels == 4:
                    if self.bitDepth == "8bit":
                        frame = dummyTensor.to(torch.uint8).cpu().numpy().tobytes()
                    else:
                        raise ValueError("RGBA 10bit encoding is not supported.")
                self.frameQueue.put(frame)
                writtenFrames += 1

        except Exception as e:
            logging.info(f"Error during Encoding: {e}")

        except BrokenPipeError:
            logging.info("Broken pipe, exiting encoding process.")

        finally:
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
