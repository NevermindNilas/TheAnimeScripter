import os
import argparse
import _thread
import logging
import subprocess
import time
import numpy as np
import warnings

from queue import SimpleQueue, Queue
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Some default values
main_path = os.path.dirname(os.path.realpath(__file__))
"""
TO:DO
    - Fix Rife padding, again.
    - Add bounding box support for Segmentation
    - Multihread the writing process, probably through an indexing system, going to be a freaking nightmare
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
        self.nt = args.nt
        self.half = args.half
        self.inpoint = args.inpoint
        self.outpoint = args.outpoint
        self.sharpen = args.sharpen
        self.sharpen_sens = args.sharpen_sens
        self.segment = args.segment
        self.dedup_strenght = args.dedup_strenght
        self.scenechange = args.scenechange
        self.scenechange_sens = args.scenechange_sens
        self.depth = args.depth
        self.encode_method = args.encode_method

        # This is necessary on the top since the script heavily relies on FFMPEG
        self.check_ffmpeg()

        if self.scenechange:
            from src.scenechange.scene_change import Scenechange

            scenechange = Scenechange(
                self.input, self.ffmpeg_path, self.scenechange_sens, main_path)

            scenechange.run()

            logging.info(
                "Detecting scene changes")

            return

        if self.depth:
            from src.depth.depth import Depth

            self.get_video_metadata()
            process = Depth(
                self.input, self.output, self.ffmpeg_path, self.width, self.height, self.fps, self.nframes, self.half, self.inpoint, self.outpoint)

            process.run()

            logging.info(
                "Detecting depth")

            return

        if self.segment:
            from src.segment.segment import Segment

            self.get_video_metadata()

            process = Segment(self.input, self.output, self.ffmpeg_path, self.width,
                              self.height, self.fps, self.nframes, self.inpoint, self.outpoint)
            process.run()

            logging.info(
                "Segmenting video")

            return

        # There's no need to start the decode encode cycle if the user only wants to dedup
        # Therefore I just hand the input to ffmpeg and call upon mpdecimate
        if self.interpolate == False and self.upscale == False and self.dedup == True:
            if self.sharpen == True:
                self.dedup_strenght += f',cas={self.sharpen_sens}'

            if self.outpoint != 0:
                from src.trim_input import trim_input_dedup
                trim_input_dedup(self.input, self.output, self.inpoint,
                                 self.outpoint, self.dedup_strenght, self.ffmpeg_path).run()
            else:
                from src.dedup.dedup import DedupFFMPEG
                DedupFFMPEG(self.input, self.output,
                            self.dedup_strenght, self.ffmpeg_path).run()

            logging.info(
                "Deduping video")

            return

        self.get_video_metadata()
        self.intitialize_models()
        self.start()

    def start(self):

        self.pbar = tqdm(total=self.nframes, desc="Processing Frames",
                         unit="frames", dynamic_ncols=True, colour="green")

        self.read_buffer = Queue(maxsize=500)
        self.processed_frames = SimpleQueue()

        self.threads_done = False

        _thread.start_new_thread(self.build_buffer, ())
        _thread.start_new_thread(self.clear_write_buffer, ())

        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(self.process)

        while self.read_buffer.qsize() > 0 and self.processed_frames.qsize() > 0:
            time.sleep(0.1)

        self.threads_done = True

    def intitialize_models(self):

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
            if self.interpolate_method == "rife":
                from src.rife.rife import Rife

                UHD = True if self.new_width >= 3840 and self.new_height >= 2160 else False
                self.interpolate_process = Rife(
                    int(self.interpolate_factor), self.half, self.new_width, self.new_height, UHD)
            elif self.interpolate_method == "rife-ncnn":
                # Need to implenet Rife NCNN
                # Current options are with frame extraction which is not ideal, I will look into rife ncnn wrapper but it only supports python 3.10
                # And building with cmake throws a tantrum, so I will look into it later
                logging.info(
                    f"There was an error in choosing the interpolation method, {self.interpolate_method} is not a valid option")

    def build_buffer(self):
        
        ffmpeg_command = [
            self.ffmpeg_path,
            "-i", str(self.input),
        ]
        if self.outpoint != 0:
            ffmpeg_command.extend(
                ["-ss", str(self.inpoint), "-to", str(self.outpoint)])

        if self.dedup == True:
            ffmpeg_command.extend(
                ["-vf", self.dedup_strenght, "-an"])

        ffmpeg_command.extend([
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-v", "quiet",
            "-stats",
            "-",
        ])

        process = subprocess.Popen(
            ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        logging.info(
            f"Building the buffer with: {ffmpeg_command}")
        self.reading_done = False
        frame_size = self.width * self.height * 3
        frame_count = 0
        for chunk in iter(lambda: process.stdout.read(frame_size), b''):
            if len(chunk) != frame_size:
                logging.error(
                    f"Read {len(chunk)} bytes but expected {frame_size}")
                break
            frame = np.frombuffer(chunk, dtype=np.uint8).reshape(
                (self.height, self.width, 3))
            self.read_buffer.put(frame)  # Modified line
            frame_count += 1

        stderr = process.stderr.read().decode()
        if stderr:
            if "bitrate=" not in stderr:
                logging.error(f"ffmpeg error: {stderr}")

        if self.interpolate == True:
            frame_count = frame_count * self.interpolate_factor

        self.pbar.total = frame_count

        self.pbar.refresh()

        # For terminating the pipe and subprocess properly
        process.stdout.close()
        process.stderr.close()
        process.terminate()
        
        self.read_buffer.put(None)

        self.reading_done = True
        logging.info(f"Read {frame_count} frames")

    def process(self):
        prev_frame = None
        try:
            while True:
                frame = self.read_buffer.get()
                if frame is None:
                    if self.read_buffer.empty() and self.reading_done == True:
                        break


                if self.upscale:
                    frame = self.upscale_process.run(frame)

                if self.interpolate:
                    if prev_frame is not None:
                        results = self.interpolate_process.run(
                            prev_frame, frame)

                        for result in results:
                            self.processed_frames.put(result)

                        prev_frame = frame
                    else:
                        prev_frame = frame

                self.processed_frames.put(frame)

        except Exception as e:
            logging.exception("An error occurred during processing")

    def clear_write_buffer(self):
        try:
            from src.encode_settings import encode_settings
            command: list = encode_settings(self.encode_method, self.new_width, self.new_height,
                                            self.fps, self.output, self.ffmpeg_path, self.sharpen, self.sharpen_sens)

            logging.info(
                f"Encoding options: {command}")

        except Exception as e:
            logging.exception(
                f"There was an error in choosing the encode method, {self.encode_method} is not a valid option")

            return

        pipe = subprocess.Popen(
            command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        try:
            while True:
                if self.processed_frames.empty() and self.threads_done == True:
                    break
                
                frame = self.processed_frames.get()

                frame = np.ascontiguousarray(frame)
                pipe.stdin.write(frame.tobytes())
                self.pbar.update(1)

        except Exception as e:
            logging.exception("An error occurred during writing")

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

        logging.info(
            f"Video Metadata: {self.width}x{self.height} @ {self.fps}fps, {self.nframes} frames")

        cap.release()

    def check_ffmpeg(self):
        dir_path = os.path.dirname(os.path.abspath(__file__))
        self.ffmpeg_path = os.path.join(dir_path, "ffmpeg", "ffmpeg.exe")

        # Check if FFMPEG exists at that path
        if not os.path.exists(self.ffmpeg_path):
            print("Couldn't find FFMPEG, downloading it now")
            print("This might add an aditional few seconds to the startup time of the process until FFMPEG is downloaded and caches are built, but it will only happen once")
            logging.info("The user doesn't have FFMPEG, downloading it now")
            ffmpeg_bat_location = os.path.join(dir_path, "get_ffmpeg.bat")
            subprocess.call(ffmpeg_bat_location, shell=True)


def main():
    script_version = "0.1.6"
    log_file_path = os.path.join(main_path, "log.txt")

    logging.basicConfig(filename=log_file_path, filemode='w',
                        format='%(message)s', level=logging.INFO)

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input", type=str, required=True)
    argparser.add_argument("--output", type=str, required=True)
    argparser.add_argument("--interpolate", type=int, default=0)
    argparser.add_argument("--interpolate_factor", type=int, default=2)
    argparser.add_argument("--interpolate_method", type=str, default="rife")
    argparser.add_argument("--upscale", type=int, default=0)
    argparser.add_argument("--upscale_factor", type=int, default=2)
    argparser.add_argument("--upscale_method",  type=str,
                           default="ShuffleCugan")
    argparser.add_argument("--cugan_kind", type=str, default="no-denoise")
    argparser.add_argument("--dedup", type=int, default=0)
    argparser.add_argument("--dedup_method", type=str, default="ffmpeg")
    argparser.add_argument("--dedup_strenght", type=str, default="light")
    argparser.add_argument("--nt", type=int, default=1)
    argparser.add_argument("--half", type=int, default=1)
    argparser.add_argument("--inpoint", type=float, default=0)
    argparser.add_argument("--outpoint", type=float, default=0)
    argparser.add_argument("--sharpen", type=int, default=0)
    argparser.add_argument("--sharpen_sens", type=float, default=50)
    argparser.add_argument("--segment", type=int, default=0)
    argparser.add_argument("--scenechange", type=int, default=0)
    argparser.add_argument("--scenechange_sens", type=float, default=40)
    argparser.add_argument("--depth", type=int, default=0)
    argparser.add_argument("--encode_method", type=str, default="x264")

    args = argparser.parse_args()

    # Whilst this is ugly, it was easier to work with the Extendscript interface this way
    args.interpolate = True if args.interpolate == 1 else False
    args.scenechange = True if args.scenechange == 1 else False
    args.sharpen = True if args.sharpen == 1 else False
    args.upscale = True if args.upscale == 1 else False
    args.segment = True if args.segment == 1 else False
    args.dedup = True if args.dedup == 1 else False
    args.depth = True if args.depth == 1 else False
    args.half = True if args.half == 1 else False

    args.upscale_method = args.upscale_method.lower()
    args.dedup_strenght = args.dedup_strenght.lower()
    args.dedup_method = args.dedup_method.lower()
    args.cugan_kind = args.cugan_kind.lower()

    args.sharpen_sens /= 100  # CAS works from 0.0 to 1.0
    args.scenechange_sens /= 100  # same for scene change

    logging.info("============== Arguments ==============")
    logging.info("")

    logging.info("Script Version: " + script_version)

    logging.info("")

    args_dict = vars(args)
    for arg in args_dict:
        logging.info(f"{arg.upper()}: {args_dict[arg]}")

    logging.info("")
    logging.info("============== Processing Outputs ==============")

    if args.output and not os.path.isabs(args.output):
        dir_path = os.path.dirname(args.input)
        args.output = os.path.join(dir_path, args.output)

    if args.upscale_factor not in [2, 3, 4] or (args.upscale_method in ["shufflecugan", "compact", "ultracompact", "superultracompact", "swinir"] and args.upscale_factor != 2):
        logging.info(
            f"Invalid upscale factor for {args.upscale_method}. Setting upscale_factor to 2.")
        args.upscale_factor = 2

    if args.interpolate_factor >= 6:
        print(f"Interpolation factor was set to {args.interpolate_factor}, each image processed is kept in VRAM and uses as much as 1GB of VRAM / image if the input is 3840x2160, this might cause VRAM overflows on some gpus, if you experience some form of blocking, try lowering the interpolation factor")
        print("I will try to implement a workaround for this in the future")

    dedup_strenght_list = {
        "light": "mpdecimate=hi=64*24:lo=64*12:frac=0.1,setpts=N/FRAME_RATE/TB",
        "medium": "mpdecimate=hi=64*100:lo=64*35:frac=0.2,setpts=N/FRAME_RATE/TB",
        "high": "mpdecimate=hi=64*200:lo=64*50:frac=0.33,setpts=N/FRAME_RATE/TB"
    }
    args.dedup_strenght = dedup_strenght_list[args.dedup_strenght]

    if args.encode_method not in ["x264", "x264_animation", "x265", "x265_animation", "nvenc_h264", "nvenc_h265", "qsv_h264", "qsv_h265"]:
        try:
            # This is for JSX compatibility
            encode_list = {
                "X264": "x264",
                "X264_Animation": "x264_animation",
                "X265": "x265",
                "X265_Animation": "x265_animation",
                "NVENC_H264": "nvenc_h264",
                "NVENC_H265": "nvenc_h265",
                "QSV_H264": "qsv_h264",
                "QSV_H265": "qsv_h265"
            }
            args.encode_method = encode_list[args.encode_method]
        except Exception as e:
            logging.exception(
                f"There was an error in choosing the encode method, {args.encode_method} is not a valid option, setting the encoder to x264")
            args.encode_method = "x264"

    if args.input is not None and args.output is not None:
        videoProcessor(args)
    else:
        logging.info("No input or output was specified, exiting")


if __name__ == "__main__":
    main()
