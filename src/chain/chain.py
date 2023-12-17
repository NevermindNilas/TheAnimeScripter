import os
import argparse
import _thread
import time

from threading import Lock
from tqdm import tqdm
from moviepy.editor import VideoFileClip
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from multiprocessing import Queue
from concurrent.futures import ThreadPoolExecutor

"""
I have basically reinvented VS, I am so cool and so dead inside
"""

class main:
    def __init__(self, args):
        self.input = os.path.normpath(args.input)
        self.output = os.path.normpath(args.output)
        self.interpolate = args.interpolate
        self.interpolate_factor = args.interpolate_factor
        self.upscale = args.upscale
        self.upscale_factor = args.upscale_factor
        self.upscale_method = args.upscale_method
        self.cugan_kind = args.cugan_kind
        self.dedup = args.dedup
        self.dedup_sens = args.dedup_sens
        self.dedup_method = args.dedup_method
        self.nt = args.nt
        self.lock = Lock()
        self.half = args.half

        self.processed_frames = Queue()

        self.intitialize()

        self.threads = []
        self.threads_are_running = True
        
        with ThreadPoolExecutor(max_workers=self.nt) as executor:
            for _ in range(self.nt):
                thread = executor.submit(self.start_process)
                self.threads.append(thread)

        while not all(thread.done() for thread in self.threads):
            time.sleep(0.01)
            
        self.threads_are_running = False

        if self.dedup_method == "FFMPEG":
            if self.interpolate or self.upscale:
                os.remove(self.input)
                
    def intitialize(self):

        if self.dedup:
            if self.dedup_method == "SSIM":
                from dedup import DedupSSIM
                print("Not available yet")
                # self.dedup = DedupSSIM()

            if self.dedup_method == "MSE":
                from dedup import DedupMSE
                print("Not available yet")
                # self.dedup = DedupMSE()

            if self.dedup_method == "FFMPEG":
                from src.dedup.dedup import DedupFFMPEG
                self.dedup = DedupFFMPEG(self.input, self.output)
                self.input = self.dedup.run()

        # Metadata needs a little time to be written.
        time.sleep(1)

        self.video = VideoFileClip(self.input)
        self.frames = self.video.iter_frames()
        self.ffmpeg_params = ["-c:v", "libx264", "-preset", "fast", "-crf",
                              "15", "-tune", "animation", "-movflags", "+faststart", "-y"]
        
        self.fps = self.video.fps * self.interpolate_factor if self.interpolate else self.video.fps

        self.frame_size = (self.video.w * self.upscale_factor, self.video.h *
                           self.upscale_factor) if self.upscale else (self.video.w, self.video.h)

        self.writer = FFMPEG_VideoWriter(
            self.output, self.frame_size, self.fps, ffmpeg_params=self.ffmpeg_params)

        self.pbar = tqdm(total=self.video.reader.nframes,
                         desc="Processing", unit="frames", colour="green")

        if self.upscale:
            if self.upscale_method == "shufflecugan" or "cugan":
                from src.cugan.cugan_node import Cugan
                self.upscale = Cugan(self.upscale_method, self.upscale_factor, self.cugan_kind, self.half)
            else:
                print("not yet implemented")
                
        if self.interpolate:
            from src.rife.rife import Rife
            UHD = True if self.frame_size[0] > 3840 or self.frame_size[1] > 2160 else False

            self.interpolate = Rife(
                self.interpolate_factor, self.half, self.frame_size, UHD)


        self.read_buffer = Queue(maxsize=500)
        _thread.start_new_thread(self.build_buffer, ())
        _thread.start_new_thread(self.write_buffer, ())

    def build_buffer(self):
        for index, frame in enumerate(self.frames):
            if frame is None:
                break
                
            self.read_buffer.put((index, frame))

        for _ in range(self.nt):
            self.read_buffer.put(None)

        self.video.close()

    def start_process(self):
        prev_frame = None
        while True:
            item = self.read_buffer.get()
            if item is None:
                print("\nThread finished")
                break
            index, frame = item
            if self.upscale:
                frame = self.upscale.run(frame)
                
            if self.interpolate and prev_frame is not None:
                results = self.interpolate.run(
                    prev_frame, frame, self.interpolate_factor, self.frame_size)
                if results is not None:
                    for result in results:
                        with self.lock:
                            self.processed_frames.put((index, result))
                else:
                    print("\nFailed to interpolate frame")
                    
            self.pbar.update(1)
            with self.lock:
                self.processed_frames.put((index, frame))
            prev_frame = frame

    def write_buffer(self):
        buffer = []
        while True:
            while not self.processed_frames.empty():
                with self.lock:
                    frame = self.processed_frames.get()
                buffer.append(frame)

            buffer.sort(key=lambda x: x[0])

            for _, frame in buffer:
                self.writer.write_frame(frame)

            buffer = []

            if not self.threads_are_running and self.processed_frames.empty():
                print("\nWrite_buffer finished")
                break

            time.sleep(0.1)

        self.writer.close()
        self.pbar.close()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input", type=str, required=True)
    argparser.add_argument("--output", type=str, required=True)
    argparser.add_argument("--interpolate", type=bool, default=False)
    argparser.add_argument("--interpolate_factor", type=int, default=2)
    argparser.add_argument("--upscale", type=bool, default=False)
    argparser.add_argument("--upscale_factor", type=int, default=2)
    argparser.add_argument("--upscale_method", type=str,
                           default="shufflecugan")
    argparser.add_argument("--cugan_kind", type=str, default="no-denoise")
    argparser.add_argument("--dedup", type=bool, default=False)
    argparser.add_argument("--dedup_sens", type=int, default=5)
    argparser.add_argument("--dedup_method", type=str, default="FFMPEG")
    argparser.add_argument("--nt", type=int, default=2)
    argparser.add_argument("--half", type=bool, default=True)

    args = argparser.parse_args()

    if args.input is not None:
        main(args)
    else:
        print("Please select a video file")
    

