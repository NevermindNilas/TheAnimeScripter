import os
import torch
import logging
import numpy as np
import torch.nn.functional as F
import cv2

from concurrent.futures import ThreadPoolExecutor
from src.utils.ffmpegSettings import BuildBuffer, WriteBuffer
from src.utils.downloadModels import downloadModels, weightsDir, modelsMap
from src.utils.progressBarLogic import ProgressBarLogic
from src.utils.isCudaInit import CudaChecker

checker = CudaChecker()


MEANTENSOR = (
    torch.tensor([0.485, 0.456, 0.406]).contiguous().view(3, 1, 1).to(checker.device)
)
STDTENSOR = (
    torch.tensor([0.229, 0.224, 0.225]).contiguous().view(3, 1, 1).to(checker.device)
)


def calculateAspectRatio(width, height, depthQuality="high"):
    if depthQuality == "high":
        # Whilst this doesn't necessarily allign with the model, it produces
        # better results than the model's native resolution
        newWidth = ((width + 13) // 14) * 14
        newHeight = ((height + 13) // 14) * 14
    elif depthQuality == "medium":
        newHeight = 700
        newWidth = 700
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
        width,
        height,
        fps,
        half,
        inpoint=0,
        outpoint=0,
        encode_method="x264",
        depth_method="small",
        custom_encoder="",
        benchmark=False,
        totalFrames=0,
        bitDepth: str = "16bit",
        depthQuality: str = "high",
    ):
        self.input = input
        self.output = output
        self.width = width
        self.height = height
        self.fps = fps
        self.half = half
        self.inpoint = inpoint
        self.outpoint = outpoint
        self.encode_method = encode_method
        self.depth_method = depth_method
        self.custom_encoder = custom_encoder
        self.benchmark = benchmark
        self.totalFrames = totalFrames
        self.bitDepth = bitDepth
        self.depthQuality = depthQuality

        self.handleModels()

        try:
            self.readBuffer = BuildBuffer(
                videoInput=self.input,
                inpoint=self.inpoint,
                outpoint=self.outpoint,
                totalFrames=self.totalFrames,
                fps=self.fps,
                width=self.width,
                height=self.height,
                resize=False,
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
                grayscale=True,
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

        match self.depth_method:
            case "small_v2" | "distill_small_v2":
                method = "vits"
            case "base_v2" | "distill_base_v2":
                method = "vitb"
            case "large_v2":
                method = "vitl"
            case "giant_v2":
                raise NotImplementedError("Giant model not available yet")
                # method = "vitg"

        if "distill" in self.depth_method:
            modelType = "safetensors"
        else:
            modelType = "pth"

        self.filename = modelsMap(
            model=self.depth_method, modelType=modelType, half=self.half
        )

        if not os.path.exists(os.path.join(weightsDir, self.filename, self.filename)):
            modelPath = downloadModels(
                model=self.depth_method,
                half=self.half,
                modelType=modelType,
            )

        else:
            modelPath = os.path.join(weightsDir, self.filename, self.filename)

        modelConfigs = {
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

        self.model = DepthAnythingV2(**modelConfigs[method])

        if "distill" in self.depth_method:
            from safetensors.torch import load_file

            modelWeights = load_file(modelPath)
            self.model.load_state_dict(modelWeights)
            del modelWeights
            torch.cuda.empty_cache()
        else:
            self.model.load_state_dict(torch.load(modelPath, map_location="cpu"))
        self.model = self.model.to(checker.device).eval()

        if self.half and checker.cudaAvailable:
            self.model = self.model.half()

        self.newHeight, self.newWidth = calculateAspectRatio(
            self.width, self.height, self.depthQuality
        )

        self.normStream = torch.cuda.Stream()
        self.outputNormStream = torch.cuda.Stream()
        self.stream = torch.cuda.Stream()

    @torch.inference_mode()
    def normFrame(self, frame):
        frame = F.interpolate(
            frame,
            (self.newHeight, self.newWidth),
            mode="bicubic",
            align_corners=True,
        )

        frame = (frame - MEANTENSOR) / STDTENSOR

        if self.half and checker.cudaAvailable:
            frame = frame.half()

        return frame

    @torch.inference_mode()
    def outputFrameNorm(self, depth):
        depth = F.interpolate(
            depth,
            (self.height, self.width),
            mode="bicubic",
            align_corners=True,
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
        width,
        height,
        fps,
        half,
        inpoint=0,
        outpoint=0,
        encode_method="x264",
        depth_method="small",
        custom_encoder="",
        benchmark=False,
        totalFrames=0,
        bitDepth: str = "16bit",
        depthQuality: str = "high",
    ):
        import onnxruntime as ort

        self.ort = ort

        self.input = input
        self.output = output
        self.width = width
        self.height = height
        self.fps = fps
        self.half = half
        self.inpoint = inpoint
        self.outpoint = outpoint
        self.encode_method = encode_method
        self.depth_method = depth_method
        self.custom_encoder = custom_encoder
        self.benchmark = benchmark
        self.totalFrames = totalFrames
        self.bitDepth = bitDepth
        self.depthQuality = depthQuality

        self.handleModels()

        try:
            self.readBuffer = BuildBuffer(
                videoInput=self.input,
                inpoint=self.inpoint,
                outpoint=self.outpoint,
                totalFrames=self.totalFrames,
                fps=self.fps,
                resize=False,
                width=self.width,
                height=self.height,
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
                grayscale=True,
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
            (1, self.newHeight, self.newWidth),
            device=self.deviceType,
            dtype=self.torchDType,
        ).contiguous()

        self.IoBinding.bind_output(
            name="output",
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
                align_corners=True,
            )

            if self.half:
                frame = frame.half()
            else:
                frame = frame.float()

            self.dummyInput.copy_(frame)
            self.IoBinding.bind_input(
                name="input",
                device_type=self.deviceType,
                device_id=0,
                element_type=self.numpyDType,
                shape=self.dummyInput.shape,
                buffer_ptr=self.dummyInput.data_ptr(),
            )

            self.model.run_with_iobinding(self.IoBinding)

            depth = F.interpolate(
                self.dummyOutput.float().unsqueeze(0),
                size=(self.height, self.width),
                mode="bicubic",
                align_corners=True,
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
        width,
        height,
        fps,
        half,
        inpoint=0,
        outpoint=0,
        encode_method="x264",
        depth_method="small",
        custom_encoder="",
        benchmark=False,
        totalFrames=0,
        bitDepth: str = "16bit",
        depthQuality: str = "high",
    ):
        self.input = input
        self.output = output
        self.width = width
        self.height = height
        self.fps = fps
        self.half = half
        self.inpoint = inpoint
        self.outpoint = outpoint
        self.encode_method = encode_method
        self.depth_method = depth_method
        self.custom_encoder = custom_encoder
        self.benchmark = benchmark
        self.totalFrames = totalFrames
        self.bitDepth = bitDepth
        self.depthQuality = depthQuality

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
                resize=False,
                width=self.width,
                height=self.height,
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
                grayscale=True,
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
                align_corners=True,
            )
            frame = (frame - MEANTENSOR) / STDTENSOR
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
                align_corners=True,
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


class OGDepthV2CUDA:
    def __init__(
        self,
        input,
        output,
        width,
        height,
        fps,
        half,
        inpoint=0,
        outpoint=0,
        encode_method="x264",
        depth_method="small",
        custom_encoder="",
        benchmark=False,
        totalFrames=0,
        bitDepth: str = "16bit",
        depthQuality: str = "high",
    ):
        self.input = input
        self.output = output
        self.width = width
        self.height = height
        self.fps = fps
        self.half = half
        self.inpoint = inpoint
        self.outpoint = outpoint
        self.encode_method = encode_method
        self.depth_method = depth_method
        self.custom_encoder = custom_encoder
        self.benchmark = benchmark
        self.totalFrames = totalFrames
        self.bitDepth = bitDepth
        self.depthQuality = depthQuality

        self.handleModels()

        self.newHeight, self.newWidth = calculateAspectRatio(
            self.width, self.height, self.depthQuality
        )
        try:
            self.video = cv2.VideoCapture(self.input)
            self.output = cv2.VideoWriter(
                self.output,
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps,
                (self.width, self.height),
            )

            self.process()
        except Exception as e:
            logging.exception(f"Something went wrong, {e}")

    def handleModels(self):
        from .og_dpt_v2 import DepthAnythingV2

        match self.depth_method:
            case "og_small_v2" | "og_distill_small_v2":
                method = "vits"
                toDownload = "small_v2"
            case "og_base_v2" | "og_distill_base_v2":
                method = "vitb"
            case "og_large_v2":
                method = "vitl"
                toDownload = "large_v2"
            case "giant_v2":
                raise NotImplementedError("Giant model not available yet")
                # method = "vitg"

        if "distill" in self.depth_method:
            modelType = "safetensors"
            toDownload = "distill_" + toDownload
        else:
            modelType = "pth"

        self.filename = modelsMap(model=toDownload, modelType=modelType, half=self.half)

        if not os.path.exists(os.path.join(weightsDir, self.filename, self.filename)):
            modelPath = downloadModels(
                model=toDownload,
                half=self.half,
                modelType=modelType,
            )

        else:
            modelPath = os.path.join(weightsDir, self.filename, self.filename)

        modelConfigs = {
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

        self.model = DepthAnythingV2(**modelConfigs[method])

        if "distill" in self.depth_method:
            from safetensors.torch import load_file

            modelWeights = load_file(modelPath)
            self.model.load_state_dict(modelWeights)
            del modelWeights
            torch.cuda.empty_cache()
        else:
            self.model.load_state_dict(torch.load(modelPath, map_location="cpu"))
        self.model = self.model.to(checker.device).eval()

        self.newHeight, self.newWidth = calculateAspectRatio(
            self.width, self.height, self.depthQuality
        )

        if self.half and checker.cudaAvailable:
            self.model = self.model.half()
        else:
            self.model = self.model.float()

        self.normStream = torch.cuda.Stream()
        self.stream = torch.cuda.Stream()

    @torch.inference_mode()
    def processFrame(self, frame):
        try:
            depth = self.model.infer_image(frame, self.newHeight, self.half)
            self.output.write(depth)

        except Exception as e:
            logging.exception(f"Something went wrong while processing the frame, {e}")

    def process(self):
        frameCount = 0

        with ProgressBarLogic(self.totalFrames) as bar:
            for _ in range(self.totalFrames):
                ret, frame = self.video.read()
                if not ret:
                    break
                self.processFrame(frame)
                frameCount += 1
                bar(1)

        logging.info(f"Processed {frameCount} frames")

        self.video.release()
        self.output.release()
