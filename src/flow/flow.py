import numpy as np
import logging
import os
import cv2
import torch

from src.downloadModels import downloadModels, weightsDir, modelsMap
from concurrent.futures import ThreadPoolExecutor
from src.ffmpegSettings import BuildBuffer, WriteBuffer
from src.coloredPrints import yellow

class OpticalFlowPytorch:
    def __init__(
            self,
            input,
            output,
            ffmpegPath,
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
    ):
        
        self.input = input
        self.output = output
        self.ffmpegPath = ffmpegPath
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

        self.handleModel()
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
                grayscale=False,
                transparent=True,
                audio=False,
                benchmark=self.benchmark,
            )

            with ThreadPoolExecutor(max_workers=3) as executor:
                executor.submit(self.readBuffer.start)
                executor.submit(self.process)
                executor.submit(self.writeBuffer.start)
            
        except Exception as e:
            logging.error(f"Error processing the flow video: {e}")

    def handleModel(self):
        filename = modelsMap("Flow")
        if not os.path.exists(os.path.join(weightsDir, "Flow", filename)):
            modelPath = downloadModels(model="Flow")
        else:
            modelPath = os.path.join(weightsDir, "Flow", filename)

        self.isCudaAvailable = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.isCudaAvailable else "cpu")

        self.model = None

        # Do the rest