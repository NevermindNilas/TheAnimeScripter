import os
import torch
import logging
import numpy as np
import torch.nn.functional as F

from concurrent.futures import ThreadPoolExecutor
from src.coloredPrints import yellow
from src.ffmpegSettings import BuildBuffer, WriteBuffer
from src.downloadModels import downloadModels, weightsDir, modelsMap
from alive_progress import alive_bar


def calculateAspectRatio(width, height):
    newWidth = ((width + 13 ) // 14) * 14
    newHeight = ((height + 13) // 14) * 14

    return newHeight, newWidth

class DepthV2:
    def __init__(
        self,
        input,
        output,
        ffmpeg_path,
        width,
        height,
        fps,
        half,
        inpoint=0,
        outpoint=0,
        encode_method="x264",
        depth_method="small",
        custom_encoder="",
        buffer_limit=50,
        benchmark=False,
        totalFrames=0,
        bitDepth: str = "16bit",
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
        self.depth_method = depth_method
        self.custom_encoder = custom_encoder
        self.buffer_limit = buffer_limit
        self.benchmark = benchmark
        self.totalFrames = totalFrames
        self.bitDepth = bitDepth

        self.handleModels()

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
                grayscale=True,
                audio=False,
                benchmark=self.benchmark,
                bitDepth=self.bitDepth,
            )

            with ThreadPoolExecutor(max_workers=3) as executor:
                executor.submit(self.readBuffer.start)
                executor.submit(self.process)
                executor.submit(self.writeBuffer.start)

        except Exception as e:
            logging.exception(f"Something went wrong, {e}")

    def handleModels(self):
        from .dpt_v2 import DepthAnythingV2

        self.filename = modelsMap(
            model=self.depth_method, modelType="pth", half=self.half
        )

        if not os.path.exists(os.path.join(weightsDir, self.filename, self.filename)):
            modelPath = downloadModels(
                model=self.depth_method,
                half=self.half,
                modelType="pth",
            )

        else:
            modelPath = os.path.join(weightsDir, self.filename, self.filename)

        match self.depth_method:
            case "small_v2":
                method = "vits"
            case "base_v2":
                method = "vitb"
            case "large_v2":
                method = "vitl"
            case "giant_v2":
                raise NotImplementedError("Giant model not available yet")
                # method = "vitg"

        self.isCudaAvailable = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.isCudaAvailable else "cpu")

        model_configs = {
            "vits": {
                "encoder": "vits",
                "features": 64,
                "out_channels": [48, 96, 192, 384],
            },
            "vitb": {
                "encoder": "vitb",
                "features": 128,
                "out_channels": [96, 192, 384, 768],
            },
            "vitl": {
                "encoder": "vitl",
                "features": 256,
                "out_channels": [256, 512, 1024, 1024],
            },
            "vitg": {
                "encoder": "vitg",
                "features": 384,
                "out_channels": [1536, 1536, 1536, 1536],
            },
        }

        self.model = DepthAnythingV2(**model_configs[method])
        self.model.load_state_dict(torch.load(modelPath, map_location="cpu"))
        self.model = self.model.to(self.device).eval()

        if self.half and self.isCudaAvailable:
            self.model = self.model.half()

        self.mean_tensor = (
            torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
        )
        self.std_tensor = (
            torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)
        )

        self.newHeight, self.newWidth = calculateAspectRatio(self.width, self.height)

    @torch.inference_mode()
    def processFrame(self, frame):
        try:
            frame = frame.to(self.device).mul(1.0 / 255.0).permute(2, 0, 1).unsqueeze(0)

            frame = F.interpolate(
                frame.float(),
                (self.newHeight, self.newWidth),
                mode="bilinear",
                align_corners=False,
            )
            frame = (frame.to(self.device) - self.mean_tensor) / self.std_tensor
            if self.half and self.isCudaAvailable:
                frame = frame.half()

            depth = self.model(frame)
            depth = F.interpolate(
                depth,
                (self.height, self.width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            self.writeBuffer.write(depth)

        except Exception as e:
            logging.exception(f"Something went wrong while processing the frame, {e}")

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


class DepthDirectMLV2:
    def __init__(
        self,
        input,
        output,
        ffmpeg_path,
        width,
        height,
        fps,
        half,
        inpoint=0,
        outpoint=0,
        encode_method="x264",
        depth_method="small",
        custom_encoder="",
        buffer_limit=50,
        benchmark=False,
        totalFrames=0,
        bitDepth: str = "16bit",
    ):
        import onnxruntime as ort

        self.ort = ort

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
        self.depth_method = depth_method
        self.custom_encoder = custom_encoder
        self.buffer_limit = buffer_limit
        self.benchmark = benchmark
        self.totalFrames = totalFrames
        self.bitDepth = bitDepth

        self.handleModels()

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
                grayscale=True,
                audio=False,
                benchmark=self.benchmark,
                bitDepth=self.bitDepth
            )

            with ThreadPoolExecutor(max_workers=3) as executor:
                executor.submit(self.readBuffer.start)
                executor.submit(self.process)
                executor.submit(self.writeBuffer.start)

        except Exception as e:
            logging.exception(f"Something went wrong, {e}")

    def handleModels(self):
        self.filename = modelsMap(
            model=self.depth_method, modelType="onnx", half=self.half
        )

        if not os.path.exists(os.path.join(weightsDir, self.filename, self.filename)):
            modelPath = downloadModels(
                model=self.depth_method,
                half=self.half,
                modelType="onnx",
            )
        else:
            modelPath = os.path.join(weightsDir, self.filename, self.filename)

        providers = self.ort.get_available_providers()

        if "DmlExecutionProvider" in providers:
            logging.info("DirectML provider available. Defaulting to DirectML")
            self.model = self.ort.InferenceSession(
                modelPath, providers=["DmlExecutionProvider"]
            )
        else:
            logging.info(
                "DirectML provider not available, falling back to CPU, expect significantly worse performance, ensure that your drivers are up to date and your GPU supports DirectX 12"
            )
            self.model = self.ort.InferenceSession(
                modelPath, providers=["CPUExecutionProvider"]
            )

        self.deviceType = "cpu"
        self.device = torch.device(self.deviceType)

        if self.half:
            self.numpyDType = np.float16
            self.torchDType = torch.float16
        else:
            self.numpyDType = np.float32
            self.torchDType = torch.float32

        self.newWidth, self.newHeight = calculateAspectRatio(self.width, self.height)

        self.IoBinding = self.model.io_binding()
        self.dummyInput = torch.zeros(
            (1, 3, self.newHeight, self.newWidth),
            device=self.deviceType,
            dtype=self.torchDType,
        ).contiguous()

        self.dummyOutput = torch.zeros(
            (1, 1, self.newHeight, self.newWidth),
            device=self.deviceType,
            dtype=self.torchDType,
        ).contiguous()

        self.IoBinding.bind_output(
            name="depth",
            device_type=self.deviceType,
            device_id=0,
            element_type=self.numpyDType,
            shape=self.dummyOutput.shape,
            buffer_ptr=self.dummyOutput.data_ptr(),
        )

    @torch.inference_mode()
    def processFrame(self, frame):
        try:
            frame = frame.to(self.device).mul(1.0 / 255.0).permute(2, 0, 1).unsqueeze(0)

            frame = F.interpolate(
                frame,
                size=(self.newHeight, self.newWidth),
                mode="bilinear",
                align_corners=False,
            )

            if self.half:
                frame = frame.half()
            else:
                frame = frame.float()

            self.dummyInput.copy_(frame)
            self.IoBinding.bind_input(
                name="image",
                device_type=self.deviceType,
                device_id=0,
                element_type=self.numpyDType,
                shape=self.dummyInput.shape,
                buffer_ptr=self.dummyInput.data_ptr(),
            )

            self.model.run_with_iobinding(self.IoBinding)

            depth = F.interpolate(
                self.dummyOutput.float(),
                size=(self.height, self.width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255
            self.writeBuffer.write(depth)

        except Exception as e:
            logging.exception(f"Something went wrong while processing the frame, {e}")

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


class DepthTensorRTV2:
    def __init__(
        self,
        input,
        output,
        ffmpeg_path,
        width,
        height,
        fps,
        half,
        inpoint=0,
        outpoint=0,
        encode_method="x264",
        depth_method="small",
        custom_encoder="",
        buffer_limit=50,
        benchmark=False,
        totalFrames=0,
        bitDepth: str = "16bit",
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
        self.depth_method = depth_method
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

        self.handleModels()

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
                grayscale=True,
                audio=False,
                benchmark=self.benchmark,
                bitDepth=bitDepth
            )

            with ThreadPoolExecutor(max_workers=3) as executor:
                executor.submit(self.readBuffer.start)
                executor.submit(self.process)
                executor.submit(self.writeBuffer.start)

        except Exception as e:
            logging.exception(f"Something went wrong, {e}")

    def handleModels(self):
        self.isCudaAvailable = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.isCudaAvailable else "cpu")
        self.filename = modelsMap(
            model=self.depth_method, modelType="onnx", half=self.half
        )

        if not os.path.exists(os.path.join(weightsDir, self.filename, self.filename)):
            modelPath = downloadModels(
                model=self.depth_method,
                half=self.half,
                modelType="onnx",
            )
        else:
            modelPath = os.path.join(weightsDir, self.filename, self.filename)

        self.newHeight, self.newWidth = calculateAspectRatio(self.width, self.height)

        engineName = f"{self.depth_method}.engine"

        enginePath = modelPath.replace(".onnx", engineName)

        if not os.path.exists(enginePath):
            toPrint = f"Model engine not found, creating engine for model: {modelPath}, this may take a while..."
            print(yellow(toPrint))
            logging.info(toPrint)
            profiles = [
                self.Profile().add(
                    "image",
                    min=(1, 3, self.newHeight, self.newWidth),
                    opt=(1, 3, self.newHeight, self.newWidth),
                    max=(1, 3, self.newHeight, self.newWidth),
                ),
            ]
            self.engine = self.engine_from_network(
                self.network_from_onnx_path(modelPath),
                config=self.CreateConfig(fp16=self.half, profiles=profiles),
            )
            self.engine = self.SaveEngine(self.engine, enginePath)

            self.engine.__call__()

        with open(enginePath, "rb") as f, self.trt.Runtime(
            self.trt.Logger(self.trt.Logger.INFO)
        ) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()

        self.stream = torch.cuda.Stream()
        self.dummyInput = torch.zeros(
            (1, 3, self.newHeight, self.newWidth),
            device=self.device,
            dtype=torch.float16 if self.half else torch.float32,
        )

        self.dummyOutput = torch.zeros(
            (1, 1, self.newHeight, self.newWidth),
            device=self.device,
            dtype=torch.float16 if self.half else torch.float32,
        )

        self.mean_tensor = (
            torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
        )
        self.std_tensor = (
            torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)
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
            for _ in range(10):
                self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
                self.stream.synchronize()

    @torch.inference_mode()
    def processFrame(self, frame):
        with torch.cuda.stream(self.stream):
            try:
                frame = (
                    frame.to(self.device).mul(1.0 / 255.0).permute(2, 0, 1).unsqueeze(0)
                )

                frame = F.interpolate(
                    frame.float(),
                    (self.newHeight, self.newWidth),
                    mode="bilinear",
                    align_corners=False,
                )
                frame = (frame.to(self.device) - self.mean_tensor) / self.std_tensor
                if self.half and self.isCudaAvailable:
                    frame = frame.half()

                self.dummyInput.copy_(frame, non_blocking=True)
                self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
                self.stream.synchronize()

                depth = F.interpolate(
                    self.dummyOutput,
                    size=[self.height, self.width],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

                depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255
                self.writeBuffer.write(depth)

            except Exception as e:
                logging.exception(
                    f"Something went wrong while processing the frame, {e}"
                )

    def process(self):
        frameCount = 0

        with alive_bar(
            self.totalFrames, title="Processing", bar="smooth", unit="frames"
        ) as bar:
            for _ in range(self.totalFrames):
                self.processFrame(self.readBuffer.read())
                frameCount += 1
                bar(1)

        logging.info(f"Processed {frameCount} frames")

        self.writeBuffer.close()

