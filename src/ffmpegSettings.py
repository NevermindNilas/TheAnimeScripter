import logging
import subprocess
import numpy as np
import os
import sys
import shutil

from queue import Queue


def getDedupStrenght(
    dedupSens: float = 0.0,
    hi_min: float = 64 * 2,
    hi_max: float = 64 * 150,
    lo_min: float = 64 * 2,
    lo_max: float = 64 * 30,
    frac_min: float = 0.1,
    frac_max: float = 0.3,
) -> str:
    """
    Get FFMPEG dedup Params based on the dedupSens attribute.
    The min maxes are based on preset values that work well for most content.
    returns: str - hi={hi}:lo={lo}:frac={frac},setpts=N/FRAME_RATE/TB
    """

    def interpolate(x, x1, x2, y1, y2):
        return y1 + (x - x1) * (y2 - y1) / (x2 - x1)

    hi = interpolate(dedupSens, 0, 100, hi_min, hi_max)
    lo = interpolate(dedupSens, 0, 100, lo_min, lo_max)
    frac = interpolate(dedupSens, 0, 100, frac_min, frac_max)
    return f"hi={hi}:lo={lo}:frac={frac},setpts=N/FRAME_RATE/TB"


def encodeYTDLP(input, output, ffmpegPath, encode_method, custom_encoder):
    command = [ffmpegPath, "-i", input]

    if custom_encoder == "":
        command.extend(matchEncoder(encode_method))
    else:
        command.extend(custom_encoder.split())

    command.append(output)

    logging.info(f"Encoding options: {' '.join(command)}")

    return command


def matchEncoder(encode_method: str):
    """
    encode_method: str - The method to use for encoding the video. Options include "x264", "x264_animation", "nvenc_h264", etc.
    """
    command = []
    # Settings inspiration from: https://www.xaymar.com/guides/obs/high-quality-recording/
    match encode_method:
        case "x264":
            command.extend(["-c:v", "libx264", "-preset", "fast", "-crf", "15"])
        case "x264_animation":
            command.extend(
                [
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-tune",
                    "animation",
                    "-crf",
                    "15",
                ]
            )
        case "x265":
            command.extend(["-c:v", "libx265", "-preset", "fast", "-crf", "15"])

        case "nvenc_h264":
            command.extend(["-c:v", "h264_nvenc", "-preset", "p1", "-cq", "15"])
        case "nvenc_h265":
            command.extend(["-c:v", "hevc_nvenc", "-preset", "p1", "-cq", "15"])
        case "qsv_h264":
            command.extend(
                ["-c:v", "h264_qsv", "-preset", "veryfast", "-global_quality", "15"]
            )
        case "qsv_h265":
            command.extend(
                ["-c:v", "hevc_qsv", "-preset", "veryfast", "-global_quality", "15"]
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
        case "vp9":
            command.extend(["-c:v", "libvpx-vp9", "-crf", "15"])
        case "qsv_vp9":
            command.extend(["-c:v", "vp9_qsv", "-preset", "veryfast"])
        # Needs further testing, -qscale:v 15 seems to be extremely lossy
        case "prores":
            command.extend(["-c:v", "prores_ks", "-profile:v", "4", "-qscale:v", "15"])

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
        dedupMethod: str = "ffmpeg",
        width: int = 1920,
        height: int = 1080,
        resize: bool = False,
        resizeMethod: str = "bilinear",
        buffSize: int = 10**8,
        queueSize: int = 50,
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

    def parameters(self):
        """
        Returns the parameters of the class.
        """
        return {
            "input": self.input,
            "ffmpegPath": self.ffmpegPath,
            "inpoint": self.inpoint,
            "outpoint": self.outpoint,
            "dedup": self.dedup,
            "dedupSens": self.dedupSens,
            "resize": self.resize,
            "width": self.width,
            "height": self.height,
            "resizeMethod": self.resizeMethod,
            "buffSize": self.buffSize,
            "queueSize": self.queueSize,
        }

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
        ]

        if self.outpoint != 0:
            command.extend(["-ss", str(self.inpoint), "-to", str(self.outpoint)])

        """
        command.extend(["-c:v", "h264_qsv"])
        """

        command.extend(
            [
                "-i",
                self.input,
            ]
        )

        filters = []
        if self.dedup:
            if self.dedupeMethod == "ffmpeg":
                filters.append(f"mpdecimate={getDedupStrenght(self.dedupSens)}")

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

        command.extend(
            [
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
            ]
        )

        return command

    def start(self, queue: Queue = None, verbose: bool = False):
        """
        The actual underlying logic for decoding, it starts a queue and gets the necessary FFMPEG command from decodeSettings.
        This is meant to be used in a separate thread for faster processing.

        queue : queue.Queue, optional - The queue to put the frames into. If None, a new queue will be created.
        verbose : bool - Whether to log the progress of the decoding.
        """
        self.readBuffer = queue if queue is not None else Queue(maxsize=self.queueSize)
        command = self.decodeSettings()

        if verbose:
            logging.info(f"Decoding options: {' '.join(map(str, command))}")

        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=1,
            )

            self.readingDone = False
            frame_size = self.width * self.height * 3
            self.decodedFrames = 0

            while True:
                chunk = process.stdout.read(frame_size)
                if len(chunk) < frame_size:
                    if verbose:
                        logging.info(f"Built buffer with {self.decodedFrames} frames")

                    process.stdout.close()
                    process.terminate()
                    self.readingDone = True
                    self.readBuffer.put(None)

                    break

                frame = np.frombuffer(chunk, dtype=np.uint8).reshape(
                    (self.height, self.width, 3)
                )

                self.readBuffer.put(frame)
                self.decodedFrames += 1

        except Exception as e:
            if verbose:
                logging.error(f"An error occurred: {str(e)}")
            self.readingDone = True
            self.readBuffer.put(None)

    def read(self):
        """
        Returns a numpy array in RGB format.
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

    def encodeSettings(self, verbose: bool = False) -> list:
        """
        This will return the command for FFMPEG to work with, it will be used inside of the scope of the class.

        verbose : bool - Whether to log the progress of the encoding.
        """

        if self.transparent:
            if self.encode_method not in ["prores", "prores_ks"]:
                if verbose:
                    logging.info(
                        "Switching internally to prores for transparency support"
                    )
                self.encode_method = "prores"

            pix_fmt = "rgba"
            output_pix_fmt = "yuva444p10le"

        elif self.grayscale:
            pix_fmt = "gray"
            output_pix_fmt = "yuv420p10le"

        else:
            pix_fmt = "rgb24"
            output_pix_fmt = "yuv420p"

        command = [
            self.ffmpegPath,
            "-y",
            "-v",
            "warning",
            "-stats",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{self.width}x{self.height}",
            "-pix_fmt",
            f"{pix_fmt}",
            "-r",
            str(self.fps),
            "-thread_queue_size",
            "100",
            "-i",
            "-",
            "-an",
            "-fps_mode",
            "vfr",
        ]

        if self.custom_encoder == "":
            command.extend(matchEncoder(self.encode_method))

            filters = []
            if self.sharpen:
                filters.append("cas={}".format(self.sharpen_sens))
            if self.grayscale:
                filters.append("format=gray")
            if self.transparent:
                filters.append("format=rgba")
            if filters:
                command.extend(["-vf", ",".join(filters)])

        else:
            custom_encoder_list = self.custom_encoder.split()

            if "-vf" in custom_encoder_list:
                vf_index = custom_encoder_list.index("-vf")

                if self.sharpen:
                    custom_encoder_list[vf_index + 1] += ",cas={}".format(
                        self.sharpen_sens
                    )

                if self.grayscale:
                    custom_encoder_list[vf_index + 1] += ",format=gray"

                if self.transparent:
                    custom_encoder_list[vf_index + 1] += ",format=rgba"
            else:
                filters = []
                if self.sharpen:
                    filters.append("cas={}".format(self.sharpen_sens))
                if self.grayscale:
                    filters.append("format=gray")
                if self.transparent:
                    filters.append("format=rgba")
                if filters:
                    custom_encoder_list.extend(["-vf", ",".join(filters)])

            command.extend(custom_encoder_list)

        command.extend(["-pix_fmt", output_pix_fmt, self.output])
        return command

    def start(self, verbose: bool = False, queue: Queue = None):
        """
        The actual underlying logic for encoding, it starts a queue and gets the necessary FFMPEG command from encodeSettings.
        This is meant to be used in a separate thread for faster processing.

        verbose : bool - Whether to log the progress of the encoding.
        queue : queue.Queue, optional - The queue to get the frames
        """

        command = self.encodeSettings(verbose=verbose)

        self.writeBuffer = queue if queue is not None else Queue(maxsize=self.queueSize)

        if verbose:
            logging.info(f"Encoding options: {' '.join(map(str, command))}")

        try:
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=sys.stdout,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
            )

            writtenFrames = 0
            self.isWritingDone = False
            while True:
                frame = self.writeBuffer.get()
                if frame is None:
                    if verbose:
                        logging.info(f"Encoded {writtenFrames} frames")

                    self.process.stdin.close()
                    self.process.wait()
                    self.isWritingDone = True
                    break

                frame = np.ascontiguousarray(frame)
                self.process.stdin.buffer.write(frame.tobytes())
                writtenFrames += 1

        except Exception as e:
            if verbose:
                logging.error(f"An error occurred: {str(e)}")

        finally:
            if self.audio:
                self.mergeAudio()

    def write(self, frame: np.ndarray):
        """
        Add a frame to the queue. Must be in RGB format.
        """
        self.writeBuffer.put(frame)

    def close(self):
        """
        Close the queue.
        """
        self.writeBuffer.put(None)

    def isWritingDone(self):
        """
        Check if the writing is done, safelock for the queue environment.
        """
        return self.isWritingDone

    def mergeAudio(self):
        try:
            # Checking first if the clip has audio to begin with, if not we skip the audio merge
            ffmpegCommand = [
                self.ffmpegPath,
                "-i",
                self.input,
            ]
            result = subprocess.run(ffmpegCommand, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if "Stream #0:1" not in result.stderr.decode():
                logging.info("No audio stream found, skipping audio merge")
                return

            audioFile = os.path.splitext(self.output)[0] + "_audio.aac"
            extractCommand = [
                self.ffmpegPath,
                "-v",
                "error",
                "-stats",
                "-i",
                self.input,
                "-vn",
                "-acodec",
                "copy",
                audioFile,
            ]
            subprocess.run(extractCommand)

            mergedFile = os.path.splitext(self.output)[0] + "_merged.mp4"

            ffmpegCommand = [
                self.ffmpegPath,
                "-v",
                "error",
                "-stats",
                "-i",
                audioFile,
                "-i",
                self.output,
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-map",
                "1:v:0",
                "-map",
                "0:a:0",
                "-shortest",
                "-y",
                mergedFile,
            ]

            logging.info(
                f"Merging audio with: {' '.join(ffmpegCommand)}"
            )

            subprocess.run(ffmpegCommand)
            shutil.move(mergedFile, self.output)
            os.remove(audioFile)

        except Exception as e:
            logging.error(f"An error occurred: {str(e)}")