import os
import torch
import logging
import numpy as np
import torch.nn.functional as F

from concurrent.futures import ThreadPoolExecutor
from src.utils.ffmpegSettings import BuildBuffer, WriteBuffer
from src.utils.downloadModels import downloadModels, weightsDir, modelsMap
from src.utils.progressBarLogic import ProgressBarLogic
from src.utils.isCudaInit import CudaChecker

checker = CudaChecker()


def calculateAspectRatio(width, height, depthQuality="high"):
    if depthQuality == "high":
        # Whilst this doesn't necessarily allign with the model, it produces
        # better results than the model's native resolution
        newWidth = ((width + 13) // 14) * 14
        newHeight = ((height + 13) // 14) * 14
    else:
        newHeight = 518
        newWidth = 518

    logging.info(f"Depth Padding: {newWidth}x{newHeight}")
    return newHeight, newWidth


class DepthCuda:
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
        depthQuality: str = "high",
        mainPath: str = "",
        decodeThreads: int = 0,
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
        self.depthQuality = depthQuality
        self.mainPath = mainPath

        self.handleModels()

        try:
            self.readBuffer = BuildBuffer(
                videoInput=self.input,
                inpoint=self.inpoint,
                outpoint=self.outpoint,
                totalFrames=self.totalFrames,
                fps=self.fps,
                decodeThreads=decodeThreads,
                width=self.width,
                height=self.height,
                resize=False,
                mainPath=self.mainPath,
            )

            self.writeBuffer = WriteBuffer(
                self.mainPath,
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
                executor.submit(self.writeBuffer)
                executor.submit(self.readBuffer)
                executor.submit(self.process)

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
        self.model = self.model.to(checker.device).eval()

        if self.half and checker.cudaAvailable:
            self.model = self.model.half()

        self.mean_tensor = (
            torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(checker.device)
        )
        self.std_tensor = (
            torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(checker.device)
        )

        self.newHeight, self.newWidth = calculateAspectRatio(
            self.width, self.height, self.depthQuality
        )

        self.normStream = torch.cuda.Stream()
        self.outputNormStream = torch.cuda.Stream()
        self.stream = torch.cuda.Stream()

    @torch.inference_mode()
    def normFrame(self, frame):
        frame = F.interpolate(
            frame.float(),
            (self.newHeight, self.newWidth),
            mode="bicubic",
            align_corners=False,
        )

        frame = (frame - self.mean_tensor) / self.std_tensor

        if self.half and checker.cudaAvailable:
            frame = frame.half()

        return frame

    @torch.inference_mode()
    def outputFrameNorm(self, depth):
        depth = F.interpolate(
            depth,
            (self.height, self.width),
            mode="bicubic",
            align_corners=False,
        )
        return (depth - depth.min()) / (depth.max() - depth.min())

    @torch.inference_mode()
    def processFrame(self, frame):
        try:
            with torch.cuda.stream(self.stream):
                frame = self.normFrame(frame)
                depth = self.model(frame)
                depth = self.outputFrameNorm(depth)
            self.stream.synchronize()
            self.writeBuffer.write(depth)

        except Exception as e:
            logging.exception(f"Something went wrong while processing the frame, {e}")

    def process(self):
        frameCount = 0

        with ProgressBarLogic(self.totalFrames) as bar:
            for _ in range(self.totalFrames):
                self.processFrame(self.readBuffer.read())
                frameCount += 1
                bar(1)
                if self.readBuffer.isReadFinished():
                    if self.readBuffer.isQueueEmpty():
                        break

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
        depthQuality: str = "high",
        mainPath: str = "",
        decodeThreads: int = 0,
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
        self.depthQuality = depthQuality
        self.mainPath = mainPath

        self.handleModels()

        try:
            self.readBuffer = BuildBuffer(
                videoInput=self.input,
                inpoint=self.inpoint,
                outpoint=self.outpoint,
                totalFrames=self.totalFrames,
                fps=self.fps,
                decodeThreads=decodeThreads,
                resize=False,
                width=self.width,
                height=self.height,
                mainPath=self.mainPath,
            )

            self.writeBuffer = WriteBuffer(
                self.mainPath,
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
                executor.submit(self.writeBuffer)
                executor.submit(self.readBuffer)
                executor.submit(self.process)

        except Exception as e:
            logging.exception(f"Something went wrong, {e}")

    def handleModels(self):
        self.filename = modelsMap(
            model=self.depth_method, modelType="onnx", half=self.half
        )

        folderName = self.depth_method.replace("-directml", "-onnx")
        if not os.path.exists(os.path.join(weightsDir, folderName, self.filename)):
            modelPath = downloadModels(
                model=self.depth_method,
                half=self.half,
                modelType="onnx",
            )
        else:
            modelPath = os.path.join(weightsDir, folderName, self.filename)

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

        self.newWidth, self.newHeight = calculateAspectRatio(
            self.width, self.height, self.depthQuality
        )

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
            frame = frame.to(self.device)

            frame = F.interpolate(
                frame,
                size=(self.newHeight, self.newWidth),
                mode="bicubic",
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
                mode="bicubic",
                align_corners=False,
            )

            depth = (depth - depth.min()) / (depth.max() - depth.min())
            self.writeBuffer.write(depth)

        except Exception as e:
            logging.exception(f"Something went wrong while processing the frame, {e}")

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
        depthQuality: str = "high",
        mainPath: str = "",
        decodeThreads: int = 0,
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
        self.depthQuality = depthQuality
        self.mainPath = mainPath

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

        self.handleModels()

        try:
            self.readBuffer = BuildBuffer(
                videoInput=self.input,
                inpoint=self.inpoint,
                outpoint=self.outpoint,
                totalFrames=self.totalFrames,
                fps=self.fps,
                decodeThreads=decodeThreads,
                resize=False,
                width=self.width,
                height=self.height,
                mainPath=self.mainPath,
            )

            self.writeBuffer = WriteBuffer(
                self.mainPath,
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
                executor.submit(self.writeBuffer)
                executor.submit(self.readBuffer)
                executor.submit(self.process)

        except Exception as e:
            logging.exception(f"Something went wrong, {e}")

    def handleModels(self):
        self.filename = modelsMap(
            model=self.depth_method, modelType="onnx", half=self.half
        )

        folderName = self.depth_method.replace("-tensorrt", "-onnx")
        if not os.path.exists(os.path.join(weightsDir, folderName, self.filename)):
            self.modelPath = downloadModels(
                model=self.depth_method,
                half=self.half,
                modelType="onnx",
            )
        else:
            self.modelPath = os.path.join(weightsDir, folderName, self.filename)

        self.newHeight, self.newWidth = calculateAspectRatio(
            self.width, self.height, self.depthQuality
        )

        enginePath = self.tensorRTEngineNameHandler(
            modelPath=self.modelPath,
            fp16=self.half,
            optInputShape=[1, 3, self.newHeight, self.newWidth],
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
                fp16=self.half,
                inputsMin=[1, 3, self.newHeight, self.newWidth],
                inputsOpt=[1, 3, self.newHeight, self.newWidth],
                inputsMax=[1, 3, self.newHeight, self.newWidth],
                inputName=["image"],
            )

        self.stream = torch.cuda.Stream()
        self.dummyInput = torch.zeros(
            (1, 3, self.newHeight, self.newWidth),
            device=checker.device,
            dtype=torch.float16 if self.half else torch.float32,
        )

        self.dummyOutput = torch.zeros(
            (1, 1, self.newHeight, self.newWidth),
            device=checker.device,
            dtype=torch.float16 if self.half else torch.float32,
        )

        self.mean_tensor = (
            torch.tensor([0.485, 0.456, 0.406])
            .view(3, 1, 1)
            .to(
                checker.device,
            )
        )
        self.std_tensor = (
            torch.tensor([0.229, 0.224, 0.225])
            .view(3, 1, 1)
            .to(
                checker.device,
            )
        )

        self.bindings = [self.dummyInput.data_ptr(), self.dummyOutput.data_ptr()]

        for i in range(self.engine.num_io_tensors):
            self.context.set_tensor_address(
                self.engine.get_tensor_name(i), self.bindings[i]
            )
            tensor_name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(tensor_name) == self.trt.TensorIOMode.INPUT:
                self.context.set_input_shape(tensor_name, self.dummyInput.shape)

        self.normStream = torch.cuda.Stream()
        self.outputNormStream = torch.cuda.Stream()
        self.cudaGraph = torch.cuda.CUDAGraph()
        self.initTorchCudaGraph()

    @torch.inference_mode()
    def initTorchCudaGraph(self):
        with torch.cuda.graph(self.cudaGraph, stream=self.stream):
            self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        self.stream.synchronize()

    @torch.inference_mode()
    def normFrame(self, frame):
        with torch.cuda.stream(self.normStream):
            frame = F.interpolate(
                frame.float(),
                (self.newHeight, self.newWidth),
                mode="bicubic",
                align_corners=False,
            )
            frame = (frame - self.mean_tensor) / self.std_tensor
            if self.half:
                frame = frame.half()
            self.dummyInput.copy_(frame, non_blocking=True)
        self.normStream.synchronize()

    @torch.inference_mode()
    def normOutputFrame(self):
        with torch.cuda.stream(self.outputNormStream):
            depth = F.interpolate(
                self.dummyOutput,
                size=[self.height, self.width],
                mode="bicubic",
                align_corners=False,
            )
            depth = (depth - depth.min()) / (depth.max() - depth.min())
        self.outputNormStream.synchronize()
        return depth

    @torch.inference_mode()
    def processFrame(self, frame):
        try:
            self.normFrame(frame)
            with torch.cuda.stream(self.stream):
                self.cudaGraph.replay()
            self.stream.synchronize()
            depth = self.normOutputFrame()

            self.writeBuffer.write(depth)
        except Exception as e:
            logging.exception(f"Something went wrong while processing the frame, {e}")

    def process(self):
        frameCount = 0

        with ProgressBarLogic(self.totalFrames) as bar:
            for _ in range(self.totalFrames):
                self.processFrame(self.readBuffer.read())
                frameCount += 1
                bar(1)
                if self.readBuffer.isReadFinished():
                    if self.readBuffer.isQueueEmpty():
                        break

        logging.info(f"Processed {frameCount} frames")
        self.writeBuffer.close()
