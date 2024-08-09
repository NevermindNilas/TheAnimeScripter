import torch
import logging
import numpy as np

from vidgear.gears.stabilizer import Stabilizer
from concurrent.futures import ThreadPoolExecutor
from src.ffmpegSettings import BuildBuffer, WriteBuffer
from alive_progress import alive_bar
from src.coloredPrints import yellow

class VideoStabilizer:
    def __init__(
        self,
        input,
        output,
        ffmpeg_path,
        width,
        height,
        fps,
        half,
        inpoint,
        outpoint,
        encode_method,
        custom_encoder,
        buffer_limit,
        benchmark,
        totalFrames,
    ):
        self.input = input
        self.output = output
        self.ffmpeg_path = ffmpeg_path
        self.width = width
        self.height = height
        self.fps = fps
        self.half = half
        self.inpoint = inpoint
        self.outpoint = outpoint
        self.encode_method = encode_method
        self.custom_encoder = custom_encoder
        self.buffer_limit = buffer_limit
        self.benchmark = benchmark
        self.totalFrames = totalFrames

        try:
            self.readBuffer = BuildBuffer(
                input=self.input,
                ffmpegPath=self.ffmpeg_path,
                inpoint=self.inpoint,
                outpoint=self.outpoint,
                dedup=False,
                dedupSens=None,
                width=self.width,
                height=self.height,
                resize=False,
                resizeMethod=None,
                queueSize=self.buffer_limit,
                totalFrames=self.totalFrames,
            )

            self.writeBuffer = WriteBuffer(
                self.input,
                self.output,
                self.ffmpeg_path,
                self.encode_method,
                self.custom_encoder,
                self.width,
                self.height,
                self.fps,
                self.buffer_limit,
                sharpen=False,
                sharpen_sens=None,
                benchmark=self.benchmark,
            )

            with ThreadPoolExecutor(max_workers=3) as executor:
                executor.submit(self.readBuffer.start)
                executor.submit(self.process)
                executor.submit(self.writeBuffer.start)

        except Exception as e:
            logging.exception(f"Something went wrong during initialization: {e}")
        

    def processFrame(self, frame):
        try:
            frame = frame.cpu().numpy()
            frame = self.stabilizer.stabilize(frame)
            self.writeBuffer.write(torch.from_numpy(frame))
        except Exception as e:
            logging.exception(f"Something went wrong during stabilization: {e}")

    def process(self):
        frameCount = 0
        self.stabilizer = Stabilizer(smoothing_radius=30)
        self.firstRun = True

        with alive_bar(
            self.totalFrames, title="Processing", bar="smooth", unit="frames"
        ) as bar:
            for _ in range(self.totalFrames):
                frame = self.readBuffer.read()
                self.processFrame(frame)
                frameCount += 1
                bar(1)

        logging.info(f"Processed {frameCount} frames")
        self.writeBuffer.close()
        self.stabilizer.clean()
        self.stabilizer.stop()
