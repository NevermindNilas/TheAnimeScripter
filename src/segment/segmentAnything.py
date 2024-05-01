# https://github.com/opengeos/segment-anything <- Special thanks to the author of this repo
import numpy as np
import logging
import os
import cv2

from threading import Semaphore
from src.downloadModels import downloadModels, weightsDir, modelsMap
from concurrent.futures import ThreadPoolExecutor
from src.ffmpegSettings import BuildBuffer, WriteBuffer
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

# TO:DO
# - Eventually use this https://github.com/opengeos/segment-anything/blob/pypi/segment_anything/modeling/sam.py#L18 instead of the current implementation
# - The original code is already quite good but it abstracts many things that are not to my taste.


class SegmentAnything:
    def __init__(
        self,
        input,
        output,
        ffmpeg_path,
        width,
        height,
        fps,
        inpoint=0,
        outpoint=0,
        encode_method="x264",
        custom_encoder="",
        nt=1,
        buffer_limit=50,
        benchmark=False,
        model="vit_h",
    ) -> None:
        self.input = input
        self.output = output
        self.ffmpeg_path = ffmpeg_path
        self.width = width
        self.height = height
        self.fps = fps
        self.inpoint = inpoint
        self.outpoint = outpoint
        self.encode_method = encode_method
        self.custom_encoder = custom_encoder
        self.nt = nt
        self.buffer_limit = buffer_limit
        self.benchmark = benchmark
        self.model = model

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
            logging.error(f"An error occurred while processing the video: {e}")

    def handleModel(self):

        """Handle the model for the segmentation task."""
        filename = modelsMap(self.model)
        if not os.path.exists(os.path.join(weightsDir, self.model, filename)):
            modelPath = downloadModels(model=self.model)
        else:
            modelPath = os.path.join(weightsDir, self.model, filename)

        # sam_model_registry = {
        #     "vit_h": build_sam_vit_h,
        #     "vit_l": build_sam_vit_l,
        #     "vit_b": build_sam_vit_b,
        # }
        # sam_model_registry is a dictionary that maps the model name to the function that builds the model
        match self.model:
            case "sam-vitb":
                self.model = "vit_b"

            case "sam-vitl":
                self.model = "vit_l"

            case "sam-vith":
                self.model = "vit_h"

        # Maybe not exactly a smart thing to reassign the model variable so many times, but it's a quick fix
        self.model = sam_model_registry[self.model](modelPath)
        self.model = SamAutomaticMaskGenerator(self.model)

    def processFrame(self, frame):
        try:
            mask = self.model.generate(frame)[0]
            #cv2.imshow("mask", mask)
            #cv2.waitKey(1)
            print(mask)
            #mask = (mask * 255).astype(np.uint8)
            #mask = np.squeeze(mask, axis=2)
            #frame_with_mask = np.concatenate((frame, mask[..., np.newaxis]), axis=2)

            #self.writeBuffer.write(frame_with_mask)
        except Exception as e:
            logging.exception(f"An error occurred while processing the frame, {e}")

        finally:
            self.semaphore.release()

    def process(self):
        frameCount = 0
        self.semaphore = Semaphore(self.nt * 4)
        with ThreadPoolExecutor(max_workers=self.nt) as executor:
            while True:
                frame = self.readBuffer.read()
                if frame is None:
                    break

                self.semaphore.acquire()
                executor.submit(self.processFrame, frame)
                frameCount += 1

        logging.info(f"Processed {frameCount} frames")
        self.writeBuffer.close()
