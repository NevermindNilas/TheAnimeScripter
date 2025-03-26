import logging
import os
import torch
import cv2

from torch.nn import functional as F
from src.utils.downloadModels import downloadModels, weightsDir, modelsMap
from src.utils.ffmpegSettings import BuildBuffer, WriteBuffer
from concurrent.futures import ThreadPoolExecutor
from src.utils.progressBarLogic import ProgressBarLogic
from src.utils.isCudaInit import CudaChecker

checker = CudaChecker()


class ObjectDetection:
    def __init__(
        self,
        input: str,
        output: str,
        ffmpegPath: str,
        width: int,
        height: int,
        outputFPS: int,
        inpoint: int,
        outpoint: int,
        encodeMethod: str,
        customEncoder: str,
        benchmark: bool,
        totalFrames: int,
        half: bool,
        decodeThreads: int,
    ):
        self.input = input
        self.output = output
        self.ffmpegPath = ffmpegPath
        self.width = width
        self.height = height
        self.outputFPS = outputFPS
        self.inpoint = inpoint
        self.outpoint = outpoint
        self.encodeMethod = encodeMethod
        self.customEncoder = customEncoder
        self.benchmark = benchmark
        self.totalFrames = totalFrames
        self.half = half
        self.decodeThreads = decodeThreads

        self._handleModel()

        self._start()

    def _handleModel(self):
        filename = modelsMap("yolo11n")
        if not os.path.exists(os.path.join(weightsDir, "yolo11n", filename)):
            modelPath = downloadModels(model="yolo11n")
        else:
            modelPath = os.path.join(weightsDir, "yolo11n", filename)

        self.model = torch.load(modelPath, map_location=checker.device)["model"]

        self.model.eval()
        self.model.to(checker.device)
        self.model.half() if self.half else self.model.float()

        self.stream = torch.cuda.Stream()

    def _start(self):
        try:
            self.readBuffer = BuildBuffer(
                videoInput=self.input,
                inpoint=self.inpoint,
                outpoint=self.outpoint,
                totalFrames=self.totalFrames,
                fps=self.outputFPS,
                decodeThreads=self.decodeThreads,
            )

            self.writeBuffer = WriteBuffer(
                input=self.input,
                output=self.output,
                ffmpegPath=self.ffmpegPath,
                encode_method=self.encodeMethod,
                custom_encoder=self.customEncoder,
                grayscale=False,
                width=self.width,
                height=self.height,
                fps=self.outputFPS,
                sharpen=False,
                transparent=True,
                audio=False,
                benchmark=self.benchmark,
                inpoint=self.inpoint,
                outpoint=self.outpoint,
            )

            with ThreadPoolExecutor(max_workers=3) as executor:
                executor.submit(self.writeBuffer)
                executor.submit(self.readBuffer)
                executor.submit(self.process)

        except Exception as e:
            logging.error(f"An error occurred while processing the video: {e}")

    @torch.inference_mode()
    def resizeFrame(self, frame, height=640, width=640):
        try:
            with torch.cuda.stream(self.stream):
                frame = F.interpolate(frame, size=(height, width), mode="bilinear")
            return frame
        except Exception as e:
            logging.error(f"An error occurred while resizing the frame: {e}")

    @torch.inference_mode()
    def processFrame(self, frame):
        try:
            reference = self.resizeFrame(frame)

            with torch.cuda.stream(self.stream):
                predictions = self.model(reference)

            ogFrame = frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
            ogFrame = (ogFrame * 255).astype("uint8")

            if isinstance(predictions, tuple) and len(predictions) > 0:
                detections = predictions[0]

                for det in detections:
                    if len(det) >= 6:
                        box = det[1:5].int().cpu().numpy()
                        conf = float(det[5])
                        cls = int(det[6])

                        cv2.rectangle(
                            ogFrame,
                            (box[0], box[1]),
                            (box[2], box[3]),
                            (0, 255, 0),
                            2,
                        )

                        label = f"Class {cls}: {conf:.2f}"
                        cv2.putText(
                            ogFrame,
                            label,
                            (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2,
                        )

            outputTensor = torch.from_numpy(ogFrame).to(checker.device)
            outputTensor = outputTensor.permute(2, 0, 1).unsqueeze(0).float() / 255.0

            self.writeBuffer.write(outputTensor)

        except Exception as e:
            logging.error(f"An error occurred while processing the frame: {e}")
            self.writeBuffer.write(frame)

    def process(self):
        frameCount = 0

        with ProgressBarLogic(self.totalFrames) as bar:
            for _ in range(self.totalFrames):
                frame = self.readBuffer.read()
                self.processFrame(frame)
                frameCount += 1
                bar(1)
                if self.readBuffer.isReadFinished():
                    if self.readBuffer.isQueueEmpty():
                        break

        logging.info(f"Processed {frameCount} frames")

        self.writeBuffer.close()
