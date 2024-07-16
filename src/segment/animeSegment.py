import numpy as np
import logging
import os
import torch
import torch.nn.functional as F

from src.downloadModels import downloadModels, weightsDir, modelsMap
from concurrent.futures import ThreadPoolExecutor
from src.ffmpegSettings import BuildBuffer, WriteBuffer
from src.coloredPrints import yellow
from alive_progress import alive_bar

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
        buffer_limit=50,
        benchmark=False,
        totalFrames=0,
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
        self.buffer_limit = buffer_limit
        self.benchmark = benchmark
        self.totalFrames = totalFrames

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

        from .train import AnimeSegmentation

        self.model = AnimeSegmentation.try_load(
            "isnet_is", modelPath, self.device, img_size=1024
        )
        self.model.eval()
        self.model.to(self.device)

    def get_mask(self, input_img: torch.Tensor) -> torch.Tensor:
        s = 1024
        h, w = h0, w0 = input_img.shape[:-1]
        h, w = (s, int(s * w / h)) if h > w else (int(s * h / w), s)
        ph, pw = s - h, s - w
        input_img = input_img.float().to(self.device).mul_(1 / 255).permute(2, 0, 1).unsqueeze(0)
        img_input = F.interpolate(
            input_img,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )
        img_input = F.pad(img_input, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2))
        with torch.no_grad():
            pred = self.model(img_input)
            pred = pred[:, :, ph // 2 : ph // 2 + h, pw // 2 : pw // 2 + w]
            pred = F.interpolate(
                pred, size=(h0, w0), mode="bilinear", align_corners=False
            ).squeeze_(0).permute(1, 2, 0).mul_(255).to(torch.uint8)
            return pred

    @torch.inference_mode()
    def processFrame(self, frame):
        try:
            mask = self.get_mask(frame)
            mask = torch.squeeze(mask, dim=2)
            frameWithmask = torch.cat((frame.to(self.device), mask.unsqueeze(2)), dim=2)
            self.writeBuffer.write(frameWithmask)
        except Exception as e:
            logging.exception(f"An error occurred while processing the frame, {e}")

    def process(self):
        frameCount = 0

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
        buffer_limit=50,
        benchmark=False,
        totalFrames=0,
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
        self.buffer_limit = buffer_limit
        self.benchmark = benchmark
        self.totalFrames = totalFrames

        import tensorrt as trt
        from polygraphy.backend.trt import (
            engine_from_network,
            network_from_onnx_path,
            CreateConfig,
            Profile,
            SaveEngine,
        )

        self.trt = trt
        self.engine_from_network = engine_from_network
        self.network_from_onnx_path = network_from_onnx_path
        self.CreateConfig = CreateConfig
        self.Profile = Profile
        self.SaveEngine = SaveEngine

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

        self.device = torch.device("cuda")
        self.padHeight = ((self.height - 1) // 64 + 1) * 64 - self.height
        self.padWidth = ((self.width - 1) // 64 + 1) * 64 - self.width

        enginePrecision = "fp32"

        if not os.path.exists(modelPath.replace(".onnx", f"_{enginePrecision}.engine")):
            toPrint = f"Model engine not found, creating engine for model: {modelPath}, this may take a while..."
            print(yellow(toPrint))
            logging.info(toPrint)
            profiles = [
                self.Profile().add(
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
            self.engine = self.engine_from_network(
                self.network_from_onnx_path(modelPath),
                config=self.CreateConfig(profiles=profiles),
            )
            self.engine = self.SaveEngine(
                self.engine, modelPath.replace(".onnx", f"_{enginePrecision}.engine")
            )

            self.engine.__call__()

        with open(
            modelPath.replace(".onnx", f"_{enginePrecision}.engine"), "rb"
        ) as f, self.trt.Runtime(self.trt.Logger(self.trt.Logger.INFO)) as runtime:
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
            if self.engine.get_tensor_mode(tensor_name) == self.trt.TensorIOMode.INPUT:
                self.context.set_input_shape(tensor_name, self.dummyInput.shape)

        with torch.cuda.stream(self.stream):
            for _ in range(5):
                self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
                self.stream.synchronize()

    @torch.inference_mode()
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

    def process(self):
        frameCount = 0

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

class AnimeSegmentDirectML:
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
        buffer_limit=50,
        benchmark=False,
        totalFrames=0,
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
        self.buffer_limit = buffer_limit
        self.benchmark = benchmark
        self.totalFrames = totalFrames

        import onnxruntime as ort
        ort.set_default_logger_severity(3)
        self.ort = ort

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
        self.filename = modelsMap("segment-directml")
        if not os.path.exists(os.path.join(weightsDir, "segment-directml", self.filename)):
            modelPath = downloadModels(model="segment-directml")
        else:
            modelPath = os.path.join(weightsDir, "segment-directml", self.filename)
        
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
            frame = (
                frame.to(self.device)
                .float()
                .permute(2, 0, 1)
                .unsqueeze(0)
                .mul(1 / 255)
            )
            frame = F.pad(frame, (0, 0, self.padHeight, self.padWidth))
            print(frame.shape)
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
            frameWithMask = frameWithMask.squeeze(0).permute(1, 2, 0).mul(255).byte()
            print(frameWithMask.shape)
            self.writeBuffer.write(frameWithMask)

        except Exception as e:
            logging.exception(f"An error occurred while processing the frame, {e}")

    def process(self):
        frameCount = 0

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
