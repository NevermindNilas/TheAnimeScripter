import os
import argparse
import _thread
import logging
import subprocess
import numpy as np
import warnings
import sys

from queue import SimpleQueue, Queue
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Some default values
if getattr(sys, 'frozen', False):
    main_path = os.path.dirname(sys.executable)
else:
    main_path = os.path.dirname(os.path.abspath(__file__))

scriptVersion = "0.1.7"
"""
TO:DO
    - Add bounding box support for Segmentation
    - Look into Rife NCNN / Wrapper
    - Status bar isn't updating properly, needs fixing
    - Make the jsx file default to the last selected settings, even after reboot
    - Get system info and display it in the log file for easier debugging
"""
warnings.filterwarnings("ignore")


class videoProcessor:
    def __init__(self, args):
        self.input = args.input
        self.output = args.output
        self.interpolate = args.interpolate
        self.interpolate_factor = args.interpolate_factor
        self.interpolate_method = args.interpolate_method
        self.upscale = args.upscale
        self.upscale_factor = args.upscale_factor
        self.upscale_method = args.upscale_method
        self.cugan_kind = args.cugan_kind
        self.dedup = args.dedup
        self.dedup_method = args.dedup_method
        self.dedup_sens = args.dedup_sens
        self.nt = args.nt
        self.half = args.half
        self.inpoint = args.inpoint
        self.outpoint = args.outpoint
        self.sharpen = args.sharpen
        self.sharpen_sens = args.sharpen_sens
        self.segment = args.segment
        self.scenechange = args.scenechange
        self.scenechange_sens = args.scenechange_sens
        self.depth = args.depth
        self.encode_method = args.encode_method
        self.motion_blur = args.motion_blur

        # This is necessary on the top since the script heavily relies on FFMPEG
        self.check_ffmpeg()
        self.get_video_metadata()

        if self.scenechange:
            from src.scenechange.scene_change import Scenechange

            logging.info(
                "Detecting scene changes")

            scenechange = Scenechange(
                self.input, self.ffmpeg_path, self.scenechange_sens, main_path)

            scenechange.run()

            return

        if self.depth:
            from src.depth.depth import Depth

            logging.info(
                "Detecting depth")

            Depth(self.input, self.output, self.ffmpeg_path, self.width, self.height,
                  self.fps, self.nframes, self.half, self.inpoint, self.outpoint, self.encode_method)

            return

        if self.segment:
            from src.segment.segment import Segment

            logging.info(
                "Segmenting video")

            Segment(self.input, self.output, self.ffmpeg_path, self.width,
                    self.height, self.fps, self.nframes, self.inpoint, self.outpoint, self.encode_method)

            return

        if self.motion_blur:
            from src.motionblur.motionblur import motionBlur

            logging.info(
                "Adding motion blur")

            motionBlur(self.input, self.output, self.ffmpeg_path, self.width,
                       self.height, self.fps, self.nframes, self.inpoint, self.outpoint, self.interpolate_method, self.interpolate_factor, self.half, self.encode_method, self.dedup, self.dedup_sens)

            return

        # There's no need to start the decode encode cycle if the user only wants to dedup
        # Therefore I just hand the input to ffmpeg and call upon mpdecimate
        if self.interpolate == False and self.upscale == False and self.dedup == True:
            if self.sharpen == True:
                self.dedup_sens += f',cas={self.sharpen_sens}'

            logging.info(
                "Deduping video")

            if self.outpoint != 0:
                from src.dedup.dedup import trim_input_dedup
                trim_input_dedup(self.input, self.output, self.inpoint,
                                 self.outpoint, self.dedup_sens, self.ffmpeg_path, self.encode_method)

            else:
                from src.dedup.dedup import dedup_ffmpeg
                dedup_ffmpeg(self.input, self.output,
                             self.dedup_sens, self.ffmpeg_path, self.encode_method)

            return

        self.intitialize_models()
        self.start()

    def start(self):

        self.pbar = tqdm(total=self.nframes, desc="Processing Frames",
                         unit="frames", dynamic_ncols=True, colour="green")

        self.read_buffer = Queue(maxsize=500)
        self.processed_frames = SimpleQueue()

        _thread.start_new_thread(self.build_buffer, ())
        _thread.start_new_thread(self.clear_write_buffer, ())

        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(self.process)

    def intitialize_models(self):

        # Generating output data,
        # This is necessary for the encode_settings function to work properly
        self.new_width = self.width
        self.new_height = self.height
        self.fps = self.fps * self.interpolate_factor if self.interpolate else self.fps

        if self.upscale:
            self.new_width *= self.upscale_factor
            self.new_height *= self.upscale_factor
            logging.info(
                f"Upscaling to {self.new_width}x{self.new_height}")

            match self.upscale_method:
                case "shufflecugan" | "cugan":
                    from src.cugan.cugan import Cugan
                    self.upscale_process = Cugan(
                        self.upscale_method, int(self.upscale_factor), self.cugan_kind, self.half, self.width, self.height)

                case "cugan-amd":
                    from src.cugan.cugan import CuganAMD
                    self.upscale_process = CuganAMD(
                        self.nt, self.upscale_factor
                    )

                case "compact" | "ultracompact" | "superultracompact":
                    from src.compact.compact import Compact
                    self.upscale_process = Compact(
                        self.upscale_method, self.half, self.width, self.height)

                case "swinir":
                    from src.swinir.swinir import Swinir
                    self.upscale_process = Swinir(
                        self.upscale_factor, self.half, self.width, self.height)

                case _:
                    logging.info(
                        f"There was an error in choosing the upscale method, {self.upscale_method} is not a valid option")

        if self.interpolate:
            match self.interpolate_method:
                case "rife414" | "rife414lite" | "rife413lite" | "rife":
                    from src.rife.rife import Rife

                    UHD = True if self.new_width >= 3840 and self.new_height >= 2160 else False
                    self.interpolate_process = Rife(
                        int(self.interpolate_factor), self.half, self.new_width, self.new_height, UHD, self.interpolate_method)

                case "gmfss":
                    from src.gmfss.gmfss_fortuna_union import GMFSS

                    UHD = True if self.new_width >= 3840 and self.new_height >= 2160 else False
                    self.interpolate_process = GMFSS(
                        int(self.interpolate_factor), self.half, self.new_width, self.new_height, UHD)

                case "rife_ncnn":
                    # Need to implement Rife NCNN
                    # Current options are with frame extraction which is not ideal, I will look into rife ncnn wrapper but it only supports python 3.10
                    # And building with cmake throws a tantrum, so I will look into it later
                    logging.info(
                        f"Rife NCNN is not implemented yet, please use Rife or GMFSS for now")

                case _:
                    logging.info(
                        f"There was an error in choosing the interpolation method, {self.interpolate_method} is not a valid option")

    def build_buffer(self):

        from src.ffmpegSettings import decodeSettings

        command: list = decodeSettings(
            self.input, self.inpoint, self.outpoint, self.dedup, self.dedup_sens, self.ffmpeg_path)

        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        self.reading_done = False
        frame_size = self.width * self.height * 3
        frame_count = 0

        try:
            for chunk in iter(lambda: process.stdout.read(frame_size), b''):
                if len(chunk) != frame_size:
                    logging.error(
                        f"Read {len(chunk)} bytes but expected {frame_size}")
                    continue
                frame = np.frombuffer(chunk, dtype=np.uint8).reshape(
                    (self.height, self.width, 3))

                self.read_buffer.put(frame)
                frame_count += 1

        except Exception as e:
            logging.exception(
                f"An error occurred during reading, {e}")

        finally:
            stderr = process.stderr.read().decode()
            if stderr:
                if "bitrate=" not in stderr:
                    logging.error(f"ffmpeg error: {stderr}")

            logging.info(f"Built buffer with {frame_count} frames")

            if self.interpolate == True:
                frame_count = frame_count * self.interpolate_factor

            self.pbar.total = frame_count
            self.pbar.refresh()

            # For terminating the pipe and subprocess properly
            process.stdout.close()
            process.stderr.close()
            process.terminate()

            self.reading_done = True
            self.read_buffer.put(None)

    def process(self):
        prev_frame = None
        self.processing_done = False
        try:
            while True:
                frame = self.read_buffer.get()
                if frame is None:
                    if self.reading_done == True and self.read_buffer.empty():
                        break

                if self.upscale:
                    frame = self.upscale_process.run(frame)

                if self.interpolate:
                    if prev_frame is not None:

                        self.interpolate_process.run(prev_frame, frame)

                        for i in range(self.interpolate_factor - 1):
                            result = self.interpolate_process.make_inference(
                                (i + 1) * 1. / (self.interpolate_factor + 1))

                            self.processed_frames.put(result)

                        prev_frame = frame
                    else:
                        prev_frame = frame

                self.processed_frames.put_nowait(frame)

        except Exception as e:
            logging.exception("An error occurred during processing")

        finally:
            self.processing_done = True
            self.processed_frames.put_nowait(None)

    def clear_write_buffer(self):

        from src.ffmpegSettings import encodeSettings
        command: list = encodeSettings(self.encode_method, self.new_width, self.new_height,
                                       self.fps, self.output, self.ffmpeg_path, self.sharpen, self.sharpen_sens)

        pipe = subprocess.Popen(
            command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        try:
            while True:
                frame = self.processed_frames.get()
                if frame is None:
                    if self.processing_done == True and self.processed_frames.empty():
                        break

                frame = np.ascontiguousarray(frame)
                pipe.stdin.write(frame.tobytes())
                self.pbar.update(1)

        except Exception as e:
            logging.exception("An error occurred during writing")

        finally:
            pipe.stdin.close()
            pipe.wait()
            self.pbar.close()

    def get_video_metadata(self):
        import cv2
        cap = cv2.VideoCapture(self.input)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        self.codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

        logging.info(
            f"Video Metadata: {self.width}x{self.height} @ {self.fps}fps, {self.nframes} frames, {self.codec} codec")

        cap.release()

    def check_ffmpeg(self):
        dir_path = os.path.dirname(os.path.abspath(__file__))
        self.ffmpeg_path = os.path.join(
            dir_path, "src", "ffmpeg", "ffmpeg.exe")

        if not os.path.exists(self.ffmpeg_path):
            from src.get_ffmpeg import get_ffmpeg
            print("Couldn't find FFMPEG, downloading it now")
            print("This might add an aditional few seconds to the startup time of the process until FFMPEG is downloaded and caches are built, but it will only happen once")
            logging.info("The user doesn't have FFMPEG, downloading it now")
            get_ffmpeg(ffmpeg_path=self.ffmpeg_path)
            print("\n")


def main():
    log_file_path = os.path.join(main_path, "log.txt")
    logging.basicConfig(filename=log_file_path, filemode='w',
                        format='%(message)s', level=logging.INFO)

    argparser = argparse.ArgumentParser()
    # This is for JSX Debugging mostly
    argparser.add_argument("--version", action="store_true")
    argparser.add_argument("--input", type=str)
    argparser.add_argument("--output", type=str)
    argparser.add_argument("--interpolate", type=int, default=0)
    argparser.add_argument("--interpolate_factor", type=int, default=2)
    argparser.add_argument("--interpolate_method",
                           type=str, default="rife")
    argparser.add_argument("--upscale", type=int, default=0)
    argparser.add_argument("--upscale_factor", type=int, default=2)
    argparser.add_argument("--upscale_method",  type=str,
                           default="shufflecugan")
    argparser.add_argument("--cugan_kind", type=str, default="no-denoise")
    argparser.add_argument("--dedup", type=int, default=0)
    argparser.add_argument("--dedup_method", type=str, default="ffmpeg")
    argparser.add_argument("--dedup_sens", type=float, default=50)
    argparser.add_argument("--nt", type=int, default=1)
    argparser.add_argument("--half", type=int, default=1)
    argparser.add_argument("--inpoint", type=float, default=0)
    argparser.add_argument("--outpoint", type=float, default=0)
    argparser.add_argument("--sharpen", type=int, default=0)
    argparser.add_argument("--sharpen_sens", type=float, default=50)
    argparser.add_argument("--segment", type=int, default=0)
    argparser.add_argument("--scenechange", type=int, default=0)
    argparser.add_argument("--scenechange_sens", type=float, default=50)
    argparser.add_argument("--depth", type=int, default=0)
    argparser.add_argument("--encode_method", type=str, default="x264")
    argparser.add_argument("--motion_blur", type=int, default=0)
    args = argparser.parse_args()

    if args.version:
        print(scriptVersion)
        return
    else:
        args.version = scriptVersion

    # Whilst this is ugly, it was easier to work with the Extendscript interface this way
    args.interpolate = True if args.interpolate == 1 else False
    args.scenechange = True if args.scenechange == 1 else False
    args.sharpen = True if args.sharpen == 1 else False
    args.upscale = True if args.upscale == 1 else False
    args.segment = True if args.segment == 1 else False
    args.dedup = True if args.dedup == 1 else False
    args.depth = True if args.depth == 1 else False
    args.half = True if args.half == 1 else False

    args.interpolate_method = args.interpolate_method.lower()
    args.upscale_method = args.upscale_method.lower()
    args.encode_method = args.encode_method.lower()
    args.dedup_method = args.dedup_method.lower()
    args.cugan_kind = args.cugan_kind.lower()

    args.sharpen_sens /= 100  # CAS works from 0.0 to 1.0
    args.scenechange_sens /= 100  # same for scene change

    logging.info("============== Arguments ==============")
    logging.info("")

    args_dict = vars(args)
    for arg in args_dict:
        logging.info(f"{arg.upper()}: {args_dict[arg]}")

    logging.info("")
    logging.info("============== Processing Outputs ==============")
    logging.info("")

    if args.output and not os.path.isabs(args.output):
        dir_path = os.path.dirname(args.input)
        args.output = os.path.join(dir_path, args.output)

    if args.upscale_factor not in [2, 3, 4] or (args.upscale_method in ["shufflecugan", "compact", "ultracompact", "superultracompact", "swinir"] and args.upscale_factor != 2):
        logging.info(
            f"Invalid upscale factor for {args.upscale_method}. Setting upscale_factor to 2.")
        args.upscale_factor = 2

    if args.dedup:
        from src.ffmpegSettings import get_dedup_strength
        args.dedup_sens = get_dedup_strength(args.dedup_sens)
        logging.info(f"Setting dedup strenght to {args.dedup_sens}")

    if args.encode_method not in ["x264", "x264_animation", "nvenc_h264", "nvenc_h265", "qsv_h264", "qsv_h265", "nvenc_av1", "av1"]:
        logging.exception(
            f"There was an error in choosing the encode method, {args.encode_method} is not a valid option, setting the encoder to x264")
        args.encode_method = "x264"

    if args.interpolate_method not in ["rife", "rife414", "rife414lite", "rife413lite", "gmfss", "rife_ncnn"]:
        """
        I will keep a default rife value that will always utilize the latest available model
        Unless the user doesn't explicitly specify the interpolation method
        This is also the default argument for args.interpolate_method
        I am not planning to add one too many arches, and probably will only add the latest ones
        It will always be Ensemble False and FastMode true just because the usecase is more than likely going to be for massive interpolations
        like 8x/16x and performance is key.
        """
        try:
            # This is for JSX compatibility
            interpolate_list = {
                "rife_4.14": "rife414",
                "rife_4.14_lite": "rife414lite",
                "rife_4.13_lite": "rife413lite",
            }
            args.interpolate_method = interpolate_list[args.interpolate_method]

            logging.info(
                f"Switched interpolation method to {args.interpolate_method}")

        except Exception as e:
            logging.exception(
                f"There was an error in choosing the interpolation method, {args.interpolate_method} is not a valid option, setting the interpolation method to rife")
            args.interpolate_method = "rife"

    if args.output is None:
        logging.info(
            "No output was specified, generating output name")

        import random
        randomNumber = random.randint(0, 100000)
        args.output = os.path.splitext(
            args.input)[0] + "_TAS" + str(randomNumber) + ".mp4"

    if args.input is not None:
        videoProcessor(args)
    else:
        logging.info("No input was specified, exiting")


if __name__ == "__main__":
    main()
