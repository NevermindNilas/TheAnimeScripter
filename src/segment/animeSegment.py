import numpy as np
import logging
import os
import cv2
import torch

import tensorrt as trt
import torch.nn.functional as F

from polygraphy.backend.trt import (
    TrtRunner,
    engine_from_network,
    network_from_onnx_path,
    CreateConfig,
    Profile,
    SaveEngine,
)


from threading import Semaphore
from src.downloadModels import downloadModels, weightsDir, modelsMap
from concurrent.futures import ThreadPoolExecutor
from src.ffmpegSettings import BuildBuffer, WriteBuffer
from .train import AnimeSegmentation
from src.coloredPrints import yellow


class AnimeSegment:  # A bit ambiguous because of .train import AnimeSegmentation but it's fine
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
    ):
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
        filename = modelsMap("segment")
        if not os.path.exists(os.path.join(weightsDir, "segment", filename)):
            modelPath = downloadModels(model="segment")
        else:
            modelPath = os.path.join(weightsDir, "segment", filename)

        if torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"

        self.model = AnimeSegmentation.try_load(
            "isnet_is", modelPath, self.device, img_size=1024
        )
        self.model.eval()
        self.model.to(self.device)

    def get_mask(self, input_img):
        s = 1024
        input_img = (input_img / 255).astype(np.float32)
        h, w = h0, w0 = input_img.shape[:-1]
        h, w = (s, int(s * w / h)) if h > w else (int(s * h / w), s)
        ph, pw = s - h, s - w
        img_input = cv2.resize(input_img, (w, h))
        img_input = np.pad(
            img_input, ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2), (0, 0))
        )
        img_input = np.transpose(img_input, (2, 0, 1))
        img_input = img_input[np.newaxis, :]
        tmpImg = torch.from_numpy(img_input).type(torch.FloatTensor).to(self.device)

        with torch.no_grad():
            pred = self.model(tmpImg)
            pred = pred[0]
            pred = pred[:, ph // 2 : ph // 2 + h, pw // 2 : pw // 2 + w]
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(0), size=(h0, w0), mode="bilinear", align_corners=False
            ).squeeze(0)
            pred = pred.permute(1, 2, 0)
            return pred.mul(255).byte()

    def processFrame(self, frame):
        try:
            mask = self.get_mask(frame.numpy())
            mask = torch.squeeze(mask, dim=2)
            frameWithmask = torch.cat(
                (frame.to(self.device), mask.unsqueeze(2)), dim=2
            )
            self.writeBuffer.write(frameWithmask)
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


class AnimeSegmentTensorRT:  # A bit ambiguous because of .train import AnimeSegmentation but it's fine
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
    ):
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
        filename = modelsMap("segment-tensorrt")
        if not os.path.exists(os.path.join(weightsDir, "segment-tensorrt", filename)):
            modelPath = downloadModels(model="segment-tensorrt")
        else:
            modelPath = os.path.join(weightsDir, "segment-tensorrt", filename)

        if torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"

        self.padHeight = ((self.height - 1) // 64 + 1) * 64 - self.height
        self.padWidth = ((self.width - 1) // 64 + 1) * 64 - self.width

        enginePrecision = "fp32"

        if not os.path.exists(modelPath.replace(".onnx", f"_{enginePrecision}.engine")):
            toPrint = f"Model engine not found, creating engine for model: {modelPath}, this may take a while..."
            print(yellow(toPrint))
            logging.info(toPrint)
            profiles = [
                Profile().add(
                    "input",
                    min=(1, 3, 64, 64),
                    opt=(
                        1,
                        3,
                        self.height + self.padHeight,
                        self.width + self.padWidth,
                    ),
                    max=(1, 3, 2160, 3840),
                ),
            ]
            self.engine = engine_from_network(
                network_from_onnx_path(modelPath),
                config=CreateConfig(profiles=profiles),
            )
            self.engine = SaveEngine(
                self.engine, modelPath.replace(".onnx", f"_{enginePrecision}.engine")
            )

            with TrtRunner(self.engine) as runner:
                self.runner = runner

        with open(
            modelPath.replace(".onnx", f"_{enginePrecision}.engine"), "rb"
        ) as f, trt.Runtime(trt.Logger(trt.Logger.INFO)) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()

        self.stream = torch.cuda.Stream()
        self.dummyInput = torch.zeros(
            (1, 3, self.height + self.padHeight, self.width + self.padWidth),
            device=self.device,
            dtype=torch.float32,
        )

        self.dummyOutput = torch.zeros(
            (1, 1, self.height + self.padHeight, self.width + self.padWidth),
            device=self.device,
            dtype=torch.float32,
        )

        self.bindings = [self.dummyInput.data_ptr(), self.dummyOutput.data_ptr()]

        for i in range(self.engine.num_io_tensors):
            self.context.set_tensor_address(
                self.engine.get_tensor_name(i), self.bindings[i]
            )
            tensor_name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(tensor_name, self.dummyInput.shape)

        with torch.cuda.stream(self.stream):
            for _ in range(5):
                self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
                self.stream.synchronize()

    def processFrame(self, frame):
        try:
            with torch.cuda.stream(self.stream):
                frame = (
                    frame.to(self.device)
                    .float()
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .mul(1 / 255)
                )
                frame = F.pad(frame, (0, 0, self.padHeight, self.padWidth))
    
                self.dummyInput.copy_(frame)
                self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
                self.stream.synchronize()
    
                frameWithmask = torch.cat((frame, self.dummyOutput), dim=1)
                frameWithmask = frameWithmask[
                    :,
                    :,
                    : frameWithmask.shape[2] - self.padHeight,
                    : frameWithmask.shape[3] - self.padWidth,
                ]
    
                self.writeBuffer.write(
                    frameWithmask.squeeze(0).permute(1, 2, 0).mul_(255).byte()
                )
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
