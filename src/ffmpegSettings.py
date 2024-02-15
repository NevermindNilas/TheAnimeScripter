import logging
import subprocess
import numpy as np
import os

from queue import Queue


def encodeSettings(
    encode_method: str,
    new_width: int,
    new_height: int,
    fps: float,
    output: str,
    ffmpegPath: str,
    sharpen: bool,
    sharpen_sens: float,
    custom_encoder,
    grayscale: bool = False,
):
    """
    encode_method: str - The method to use for encoding the video. Options include "x264", "x264_animation", "nvenc_h264", etc.
    new_width: int - The width of the output video in pixels.
    new_height: int - The height of the output video in pixels.
    fps: float - The frames per second of the output video.
    output: str - The path to the output file.
    ffmpegPath: str - The path to the FFmpeg executable.
    sharpen: bool - Whether to apply a sharpening filter to the video.
    sharpen_sens: float - The sensitivity of the sharpening filter.
    grayscale: bool - Whether to encode the video in grayscale.
    """
    if grayscale:
        pix_fmt = "gray"
        output_pix_fmt = "yuv420p16le"
        if encode_method not in ["x264", "x264_animation", "x265", "av1"]:
            logging.info(
                "The selected encoder does not support yuv420p16le, switching to yuv420p10le."
            )
            output_pix_fmt = "yuv420p10le"

    else:
        pix_fmt = "rgb24"
        output_pix_fmt = "yuv420p"

    command = [
        ffmpegPath,
        "-hide_banner",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{new_width}x{new_height}",
        "-pix_fmt",
        f"{pix_fmt}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-fps_mode",
        "vfr",
    ]

    if custom_encoder == "":
        command.extend(match_encoder(encode_method))

        filters = []
        if sharpen:
            filters.append("cas={}".format(sharpen_sens))
        if grayscale:
            filters.append("format=gray")
        if filters:
            command.extend(["-vf", ",".join(filters)])

    else:
        custom_encoder_list = custom_encoder.split()

        if "-vf" in custom_encoder_list:
            vf_index = custom_encoder_list.index("-vf")

            if sharpen:
                custom_encoder_list[vf_index + 1] += ",cas={}".format(sharpen_sens)

            if grayscale:
                custom_encoder_list[vf_index + 1] += ",format=gray"
        else:
            filters = []
            if sharpen:
                filters.append("cas={}".format(sharpen_sens))
            if grayscale:
                filters.append("format=gray")

            if filters:
                custom_encoder_list.extend(["-vf", ",".join(filters)])

        command.extend(custom_encoder_list)

    command.extend(["-pix_fmt", output_pix_fmt, output])

    logging.info(f"Encoding options: {' '.join(map(str, command))}")
    return command


def getDedupStrenght(dedupSens):
    hi = interpolate(dedupSens, 0, 100, 64 * 2, 64 * 150)
    lo = interpolate(dedupSens, 0, 100, 64 * 2, 64 * 30)
    frac = interpolate(dedupSens, 0, 100, 0.1, 0.3)
    return f"hi={hi}:lo={lo}:frac={frac},setpts=N/FRAME_RATE/TB"


def interpolate(x, x1, x2, y1, y2):
    return y1 + (x - x1) * (y2 - y1) / (x2 - x1)


def encodeYTDLP(input, output, ffmpegPath, encode_method, custom_encoder):
    # This is for non rawvideo bytestreams, it's simpler to keep track this way
    # And have everything FFMPEG related organized in one file

    command = [ffmpegPath, "-i", input]

    if custom_encoder == "":
        command.extend(match_encoder(encode_method))
    else:
        command.extend(custom_encoder.split())

    command.append(output)

    logging.info(f"Encoding options: {' '.join(command)}")

    return command


def match_encoder(encode_method: str):
    """
    encode_method: str - The method to use for encoding the video. Options include "x264", "x264_animation", "nvenc_h264", etc.
    """
    command = []
    # Settings inspiration from: https://www.xaymar.com/guides/obs/high-quality-recording/
    match encode_method:
        case "x264":
            command.extend(["-c:v", "libx264", "-preset", "veryfast", "-crf", "14"])
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
                    "14",
                ]
            )
        case "x265":
            command.extend(["-c:v", "libx265", "-preset", "veryfast", "-crf", "14"])

        # Experimental, not tested
        # case "x265_animation":
        #    command.extend(['-c:v', 'libx265', '-preset', 'veryfast', '-crf', '14', '-psy-rd', '1.0', '-psy-rdoq', '10.0'])

        case "nvenc_h264":
            command.extend(["-c:v", "h264_nvenc", "-preset", "p1", "-cq", "14"])
        case "nvenc_h265":
            command.extend(["-c:v", "hevc_nvenc", "-preset", "p1", "-cq", "14"])
        case "qsv_h264":
            command.extend(
                ["-c:v", "h264_qsv", "-preset", "veryfast", "-global_quality", "14"]
            )
        case "qsv_h265":
            command.extend(
                ["-c:v", "hevc_qsv", "-preset", "veryfast", "-global_quality", "14"]
            )
        case "nvenc_av1":
            command.extend(["-c:v", "av1_nvenc", "-preset", "p1", "-cq", "14"])
        case "av1":
            command.extend(["-c:v", "libsvtav1", "-preset", "8", "-crf", "14"])
        case "h264_amf":
            command.extend(
                ["-c:v", "h264_amf", "-quality", "speed", "-rc", "cqp", "-qp", "14"]
            )
        case "hevc_amf":
            command.extend(
                ["-c:v", "hevc_amf", "-quality", "speed", "-rc", "cqp", "-qp", "14"]
            )

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
        width: int = 1920,
        height: int = 1080,
        resize: bool = False,
        resizeMethod: str = "bilinear",
        buffSize: int = 10**8,
        queueSize: int = 100,
    ):
        """
        A class meant to Pipe the Output of FFMPEG into a Queue for further processing.
        Still has slight OOM issues, but it's a good start.

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
        self.resize = resize
        self.width = width
        self.height = height
        self.resizeMethod = resizeMethod
        self.buffSize = buffSize
        self.queueSize = queueSize

    def getDedupStrenght(
        self,
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
        
        hi = interpolate(self.dedupSens, 0, 100, hi_min, hi_max)
        lo = interpolate(self.dedupSens, 0, 100, lo_min, lo_max)
        frac = interpolate(self.dedupSens, 0, 100, frac_min, frac_max)

        return f"hi={hi}:lo={lo}:frac={frac},setpts=N/FRAME_RATE/TB"

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

        command.extend(
            [
                "-i",
                self.input,
            ]
        )

        filters = []
        if self.dedup:
            filters.append(f"mpdecimate={self.getDedupStrenght()}")

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
                bufsize=self.buffSize,
            )

            self.readingDone = False
            frame_size = self.width * self.height * 3
            self.decodedFrames = 0

            while True:
                chunk = process.stdout.read(frame_size)
                if len(chunk) < frame_size:
                    if verbose:
                        logging.info(
                            f"Read {len(chunk)} bytes but expected {frame_size}"
                        )

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
