import numpy as np
import logging
import os
import torch
import torch.nn.functional as F

from src.utils.downloadModels import downloadModels, weightsDir, modelsMap
from src.utils.ffmpegSettings import BuildBuffer, WriteBuffer
from concurrent.futures import ThreadPoolExecutor
from src.utils.progressBarLogic import ProgressBarLogic
from src.utils.isCudaInit import CudaChecker

checker = CudaChecker()


class AnimeSegment:  # A bit ambiguous because of .train import AnimeSegmentation but it's fine
    def __init__(
        self,
        input,
        output,
        width,
        height,
        fps,
        inpoint=0,
        outpoint=0,
        encode_method="x264",
        custom_encoder="",
        benchmark=False,
        totalFrames=0,
    ):
        self.input = input
        self.output = output
        self.width = width
        self.height = height
        self.fps = fps
        self.inpoint = inpoint
        self.outpoint = outpoint
        self.encode_method = encode_method
        self.custom_encoder = custom_encoder
        self.benchmark = benchmark
        self.totalFrames = totalFrames

        self.handleModel()
        try:
            self.readBuffer = BuildBuffer(
                videoInput=self.input,
                inpoint=self.inpoint,
                outpoint=self.outpoint,
            )

            self.writeBuffer = WriteBuffer(
                input=self.input,
                output=self.output,
                encode_method=self.encode_method,
                custom_encoder=self.custom_encoder,
                grayscale=False,
                width=self.width,
                height=self.height,
                fps=self.fps,
                sharpen=False,
                transparent=True,
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

    def handleModel(self):
        filename = modelsMap("segment")
        if not os.path.exists(os.path.join(weightsDir, "segment", filename)):
            modelPath = downloadModels(model="segment")
        else:
            modelPath = os.path.join(weightsDir, "segment", filename)

        from .train import AnimeSegmentation

        self.model = AnimeSegmentation.try_load(
            "isnet_is", modelPath, checker.device, img_size=1024
        )
        self.model.eval()
        self.model.to(checker.device)
        self.stream = torch.cuda.Stream()

    @torch.inference_mode()
    def getMask(self, input_img: torch.Tensor) -> torch.Tensor:
        with torch.cuda.stream(self.stream):
            input_img = input_img.to(checker.device).float()
            s = 1024
            h, w = h0, w0 = input_img.shape[2], input_img.shape[3]
            h, w = (s, int(s * w / h)) if h > w else (int(s * h / w), s)
            ph, pw = s - h, s - w
            img_input = F.interpolate(
                input_img,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
            pred = self.model(
                F.pad(img_input, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2))
            )
            pred = pred[:, :, ph // 2 : ph // 2 + h, pw // 2 : pw // 2 + w]
            pred = F.interpolate(
                pred, size=(h0, w0), mode="bilinear", align_corners=False
            )
            pred = torch.cat((input_img, pred), dim=1)
        self.stream.synchronize()
        return pred

    @torch.inference_mode()
    def processFrame(self, frame):
        try:
            mask = self.getMask(frame)
            self.writeBuffer.write(mask)

        except Exception as e:
            logging.exception(f"An error occurred while processing the frame, {e}")

    def process(self):
        frameCount = 0

        with ProgressBarLogic(self.totalFrames) as bar:
            for _ in range(self.totalFrames):
                frame = self.readBuffer.read()
                if frame is None:
                    break
                self.processFrame(frame)
                frameCount += 1
                bar(1)
                if self.readBuffer.isReadFinished():
                    if self.readBuffer.isQueueEmpty():
                        break

        logging.info(f"Processed {frameCount} frames")

        self.writeBuffer.close()


class AnimeSegmentTensorRT:
    def __init__(
        self,
        input,
        output,
        width,
        height,
        fps,
        inpoint=0,
        outpoint=0,
        encode_method="x264",
        custom_encoder="",
        benchmark=False,
        totalFrames=0,
    ):
        self.input = input
        self.output = output
        self.width = width
        self.height = height
        self.fps = fps
        self.inpoint = inpoint
        self.outpoint = outpoint
        self.encode_method = encode_method
        self.custom_encoder = custom_encoder
        self.benchmark = benchmark
        self.totalFrames = totalFrames

        import tensorrt as trt
        from src.utils.trtHandler import (
            tensorRTEngineCreator,
            tensorRTEngineLoader,
            tensorRTEngineNameHandler,
        )

        self.trt = trt
        self.tensorRTEngineCreator = tensorRTEngineCreator
        self.tensorRTEngineLoader = tensorRTEngineLoader
        self.tensorRTEngineNameHandler = tensorRTEngineNameHandler

        self.handleModel()
        try:
            self.readBuffer = BuildBuffer(
                videoInput=self.input,
                inpoint=self.inpoint,
                outpoint=self.outpoint,
            )

            self.writeBuffer = WriteBuffer(
                input=self.input,
                output=self.output,
                encode_method=self.encode_method,
                custom_encoder=self.custom_encoder,
                grayscale=False,
                width=self.width,
                height=self.height,
                fps=self.fps,
                sharpen=False,
                transparent=True,
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

    def handleModel(self):
        filename = modelsMap("segment-tensorrt")
        folderName = "segment-onnx"
        if not os.path.exists(os.path.join(weightsDir, folderName, filename)):
            self.modelPath = downloadModels(model="segment-tensorrt")
        else:
            self.modelPath = os.path.join(weightsDir, folderName, filename)

        self.padHeight = ((self.height - 1) // 64 + 1) * 64 - self.height
        self.padWidth = ((self.width - 1) // 64 + 1) * 64 - self.width

        enginePath = self.tensorRTEngineNameHandler(
            modelPath=self.modelPath,
            fp16=False,  # Setting this to false cuz fp16 results are really bad compared to fp32
            optInputShape=[
                1,
                3,
                self.height + self.padHeight,
                self.width + self.padWidth,
            ],
        )

        self.engine, self.context = self.tensorRTEngineLoader(enginePath)
        if (
            self.engine is None
            or self.context is None
            or not os.path.exists(enginePath)
        ):
            self.engine, self.context = self.tensorRTEngineCreator(
                modelPath=self.modelPath,
                enginePath=enginePath,
                fp16=False,  # Setting this to false cuz fp16 results are really bad compared to fp32
                inputsMin=[
                    1,
                    3,
                    self.height + self.padHeight,
                    self.width + self.padWidth,
                ],
                inputsOpt=[
                    1,
                    3,
                    self.height + self.padHeight,
                    self.width + self.padWidth,
                ],
                inputsMax=[
                    1,
                    3,
                    self.height + self.padHeight,
                    self.width + self.padWidth,
                ],
                inputName=["input"],
            )

        self.stream = torch.cuda.Stream()
        self.dummyInput = torch.zeros(
            (1, 3, self.height + self.padHeight, self.width + self.padWidth),
            device=checker.device,
            dtype=torch.float32,
        )

        self.dummyOutput = torch.zeros(
            (1, 1, self.height + self.padHeight, self.width + self.padWidth),
            device=checker.device,
            dtype=torch.float32,
        )

        self.bindings = [self.dummyInput.data_ptr(), self.dummyOutput.data_ptr()]

        for i in range(self.engine.num_io_tensors):
            self.context.set_tensor_address(
                self.engine.get_tensor_name(i), self.bindings[i]
            )
            tensor_name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(tensor_name) == self.trt.TensorIOMode.INPUT:
                self.context.set_input_shape(tensor_name, self.dummyInput.shape)

        with torch.cuda.stream(self.stream):
            for _ in range(5):
                self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
                self.stream.synchronize()

        self.normStream = torch.cuda.Stream()
        self.outputStream = torch.cuda.Stream()

    @torch.inference_mode()
    def normFrame(self, frame):
        with torch.cuda.stream(self.normStream):
            frame = F.pad(
                frame.float(),
                (0, 0, self.padHeight, self.padWidth),
            )
            self.dummyInput.copy_(frame, non_blocking=True)
            self.normStream.synchronize()
            return frame

    @torch.inference_mode()
    def outputNorm(self, frame):
        with torch.cuda.stream(self.outputStream):
            frameWithMask = torch.cat((frame, self.dummyOutput), dim=1)
            frameWithMask = frameWithMask[
                :,
                :,
                : frameWithMask.shape[2] - self.padHeight,
                : frameWithMask.shape[3] - self.padWidth,
            ]
            self.outputStream.synchronize()
            return frameWithMask

    @torch.inference_mode()
    def processFrame(self, frame):
        try:
            frame = self.normFrame(frame)
            self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
            self.stream.synchronize()
            frameWithMask = self.outputNorm(frame)
            self.writeBuffer.write(frameWithMask)
        except Exception as e:
            logging.exception(f"An error occurred while processing the frame, {e}")

    def process(self):
        frameCount = 0

        with ProgressBarLogic(self.totalFrames) as bar:
            for _ in range(self.totalFrames):
                frame = self.readBuffer.read()
                if frame is None:
                    break
                self.processFrame(frame)
                frameCount += 1
                bar(1)
                if self.readBuffer.isReadFinished():
                    if self.readBuffer.isQueueEmpty():
                        break

        logging.info(f"Processed {frameCount} frames")

        self.writeBuffer.close()


class AnimeSegmentDirectML:
    def __init__(
        self,
        input,
        output,
        width,
        height,
        fps,
        inpoint=0,
        outpoint=0,
        encode_method="x264",
        custom_encoder="",
        benchmark=False,
        totalFrames=0,
    ):
        self.input = input
        self.output = output
        self.width = width
        self.height = height
        self.fps = fps
        self.inpoint = inpoint
        self.outpoint = outpoint
        self.encode_method = encode_method
        self.custom_encoder = custom_encoder
        self.benchmark = benchmark
        self.totalFrames = totalFrames

        import onnxruntime as ort

        ort.set_default_logger_severity(3)
        self.ort = ort

        self.handleModel()
        try:
            self.readBuffer = BuildBuffer(
                videoInput=self.input,
                inpoint=self.inpoint,
                outpoint=self.outpoint,
                totalFrames=self.totalFrames,
                fps=self.fps,
            )

            self.writeBuffer = WriteBuffer(
                self.input,
                self.output,
                self.encode_method,
                self.custom_encoder,
                self.width,
                self.height,
                self.fps,
                sharpen=False,
                sharpen_sens=None,
                grayscale=False,
                transparent=True,
                benchmark=self.benchmark,
            )

            with ThreadPoolExecutor(max_workers=3) as executor:
                executor.submit(self.readBuffer)
                executor.submit(self.process)
                executor.submit(self.writeBuffer)

        except Exception as e:
            logging.error(f"An error occurred while processing the video: {e}")

    def handleModel(self):
        self.filename = modelsMap("segment-directml")
        folderName = "segment-onnx"
        if not os.path.exists(os.path.join(weightsDir, folderName, self.filename)):
            modelPath = downloadModels(model="segment-directml")
        else:
            modelPath = os.path.join(weightsDir, folderName, self.filename)

        self.padHeight = ((self.height - 1) // 64 + 1) * 64 - self.height
        self.padWidth = ((self.width - 1) // 64 + 1) * 64 - self.width

        providers = self.ort.get_available_providers()
        if "DmlExecutionProvider" in providers:
            logging.info("DirectML provider available. Defaulting to DirectML")
            provider = "DmlExecutionProvider"
        else:
            logging.info(
                "DirectML provider not available, falling back to CPU, expect significantly worse performance, ensure that your drivers are up to date and your GPU supports DirectX 12"
            )
            provider = "CPUExecutionProvider"

        self.model = self.ort.InferenceSession(modelPath, providers=[provider])
        self.deviceType = "cpu"
        self.device = torch.device(self.deviceType)
        self.numpyDType = np.float32
        self.torchDType = torch.float32

        self.IoBinding = self.model.io_binding()
        self.dummyInput = torch.zeros(
            (1, 3, self.height + self.padHeight, self.width + self.padWidth),
            device=self.device,
            dtype=torch.float32,
        ).contiguous()

        self.dummyOutput = torch.zeros(
            (1, 1, self.height + self.padHeight, self.width + self.padWidth),
            device=self.device,
            dtype=torch.float32,
        ).contiguous()

        self.IoBinding.bind_output(
            name="output",
            device_type=self.deviceType,
            device_id=0,
            element_type=self.numpyDType,
            shape=self.dummyOutput.shape,
            buffer_ptr=self.dummyOutput.data_ptr(),
        )

    def processFrame(self, frame: torch.tensor) -> torch.tensor:
        try:
            frame = frame.to(self.device).float()
            frame = F.pad(frame, (0, 0, self.padHeight, self.padWidth))
            self.dummyInput.copy_(frame)

            self.IoBinding.bind_input(
                name="input",
                device_type=self.deviceType,
                device_id=0,
                element_type=self.numpyDType,
                shape=self.dummyInput.shape,
                buffer_ptr=self.dummyInput.data_ptr(),
            )

            frameWithMask = torch.cat((frame, self.dummyOutput), dim=1)
            frameWithMask = frameWithMask[
                :,
                :,
                : frameWithMask.shape[2] - self.padHeight,
                : frameWithMask.shape[3] - self.padWidth,
            ]
            self.writeBuffer.write(frameWithMask)

        except Exception as e:
            logging.exception(f"An error occurred while processing the frame, {e}")

    def process(self):
        frameCount = 0

        with ProgressBarLogic(self.totalFrames) as bar:
            for _ in range(self.totalFrames):
                frame = self.readBuffer.read()
                if frame is None:
                    break
                self.processFrame(frame)
                frameCount += 1
                bar(1)
                if self.readBuffer.isReadFinished():
                    if self.readBuffer.isQueueEmpty():
                        break

        logging.info(f"Processed {frameCount} frames")

        self.writeBuffer.close()
