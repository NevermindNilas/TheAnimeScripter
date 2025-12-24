import os
import torch
import logging
import numpy as np
import torch.nn.functional as F
import cv2

from src.utils.logAndPrint import logAndPrint
from concurrent.futures import ThreadPoolExecutor
from src.utils.ffmpegSettings import BuildBuffer, WriteBuffer
from src.utils.downloadModels import downloadModels, weightsDir, modelsMap
from src.utils.progressBarLogic import ProgressBarLogic
from src.utils.isCudaInit import CudaChecker
from queue import Queue

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
        # sharper results at the cost of performance and some accuracy loss.
        newWidth = ((width + 13) // 14) * 14
        newHeight = ((height + 13) // 14) * 14
    elif depthQuality == "medium":
        # Should be preferred through and through
        newHeight = 700
        newWidth = 700
    else:
        # Depth quality low
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
        compileMode: str = "default",
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
        self.compileMode = compileMode

        self.handleModels()

        try:
            self.readBuffer = BuildBuffer(
                videoInput=self.input,
                inpoint=self.inpoint,
                outpoint=self.outpoint,
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
        from .dpt_v2 import DepthAnythingV2

        match self.depth_method:
            case "small_v2" | "distill_small_v2":
                method = "vits"
            case "base_v2" | "distill_base_v2":
                method = "vitb"
            case "large_v2" | "distill_large_v2":
                method = "vitl"
            case "giant_v2":
                method = "vitg"

        self.filename = modelsMap(
            model=self.depth_method, modelType="pth", half=self.half
        )
        if not os.path.exists(
            os.path.join(weightsDir, self.depth_method, self.filename)
        ):
            modelPath = downloadModels(
                model=self.depth_method,
                half=self.half,
                modelType="pth",
            )

        else:
            modelPath = os.path.join(weightsDir, self.depth_method, self.filename)

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
            }
            if "distill" not in self.depth_method
            else {
                "encoder": "vitl",
                "features": 256,
                "out_channels": [256, 512, 1024, 1024],
                "use_bn": False,
                "use_clstoken": False,
                "max_depth": 150.0,
                "mode": "disparity",
                "pretrain_type": "dinov2",
                "del_mask_token": False,
            },
        }

        if "distill" in self.depth_method and "large" in self.depth_method:
            from src.depth.distillanydepth.modeling.archs.dam.dam import DepthAnything

            self.model = DepthAnything(**modelConfigs[method])
        else:
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

        if self.compileMode != "default":
            try:
                if self.compileMode == "max":
                    self.model.compile(mode="max-autotune-no-cudagraphs")
                elif self.compileMode == "max-graphs":
                    self.model.compile(
                        mode="max-autotune-no-cudagraphs", fullgraph=True
                    )
            except Exception as e:
                logging.error(
                    f"Error compiling model {self.interpolateMethod} with mode {self.compileMode}: {e}"
                )
                logAndPrint(
                    f"Error compiling model {self.interpolateMethod} with mode {self.compileMode}: {e}",
                    "red",
                )

            self.compileMode = "default"

        self.normStream = torch.cuda.Stream()
        self.outputNormStream = torch.cuda.Stream()
        self.stream = torch.cuda.Stream()

    @torch.inference_mode()
    def normFrame(self, frame):
        frame = F.interpolate(
            frame,
            (self.newHeight, self.newWidth),
            mode="bilinear",
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
            mode="bilinear",
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
        # Bind on CPU memory; ORT will handle copies for DML provider
        self.deviceType = "cpu"
        self.device = torch.device(self.deviceType)

        # Calculate padded model resolution (height, width)
        self.newHeight, self.newWidth = calculateAspectRatio(
            self.width, self.height, self.depthQuality
        )

        # Discover actual I/O names and dtypes from the ONNX model
        onnxInputs = self.model.get_inputs()
        onnxOutputs = self.model.get_outputs()
        self.inputName = onnxInputs[0].name
        self.outputName = onnxOutputs[0].name

        def onnxTypeToNumpy(typ: str):
            return np.float16 if "float16" in typ else np.float32

        def onnxTypeToTorch(typ: str):
            return torch.float16 if "float16" in typ else torch.float32

        self.numpyInDType = onnxTypeToNumpy(onnxInputs[0].type)
        self.torchInDType = onnxTypeToTorch(onnxInputs[0].type)
        self.numpyOutDType = onnxTypeToNumpy(onnxOutputs[0].type)
        self.torchOutDType = onnxTypeToTorch(onnxOutputs[0].type)

        # Allocate input/output buffers with correct shapes and dtypes
        self.IoBinding = self.model.io_binding()
        self.dummyInput = torch.zeros(
            (1, 3, self.newHeight, self.newWidth),
            device=self.deviceType,
            dtype=self.torchInDType,
        ).contiguous()

        out_rank = len(onnxOutputs[0].shape) if hasattr(onnxOutputs[0], "shape") else 4
        if out_rank == 3:
            out_shape = (1, self.newHeight, self.newWidth)
        else:
            # Default to NCHW with C=1 when unknown or 4D
            out_shape = (1, 1, self.newHeight, self.newWidth)

        self.dummyOutput = torch.zeros(
            out_shape,
            device=self.deviceType,
            dtype=self.torchOutDType,
        ).contiguous()

        self.IoBinding.bind_output(
            name=self.outputName,
            device_type=self.deviceType,
            device_id=0,
            element_type=self.numpyOutDType,
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
                mode="bilinear",
                align_corners=True,
            )

            # Cast to model's expected input dtype
            frame = frame.to(dtype=self.torchInDType)

            self.dummyInput.copy_(frame)
            self.IoBinding.bind_input(
                name=self.inputName,
                device_type=self.deviceType,
                device_id=0,
                element_type=self.numpyInDType,
                shape=self.dummyInput.shape,
                buffer_ptr=self.dummyInput.data_ptr(),
            )

            self.model.run_with_iobinding(self.IoBinding)

            # Bring output to float for normalization and resize to target resolution
            out_tensor = self.dummyOutput.float()
            if out_tensor.ndim == 3:
                out_tensor = out_tensor.unsqueeze(0)
            depth = F.interpolate(
                out_tensor,
                size=(self.height, self.width),
                mode="bilinear",
                align_corners=True,
            )

            # Normalize to 0-1 range
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
            self.writeBuffer.write(depth)

        except Exception as e:
            logging.exception(f"Something went wrong while processing the frame, {e}")

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
            inputName = "image" if "distill" not in self.depth_method else "input"
            self.engine, self.context = self.tensorRTEngineCreator(
                modelPath=self.modelPath,
                enginePath=enginePath,
                fp16=self.half,
                inputsMin=[1, 3, self.newHeight, self.newWidth],
                inputsOpt=[1, 3, self.newHeight, self.newWidth],
                inputsMax=[1, 3, self.newHeight, self.newWidth],
                inputName=[inputName],
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

        if "distill" in self.depth_method:
            self.dummyConst = torch.zeros(
                (1, 1, self.newHeight, self.newWidth),
                device=checker.device,
                dtype=torch.float16 if self.half else torch.float32,
            )

        self.bindings = [self.dummyInput.data_ptr(), self.dummyOutput.data_ptr()]

        if "distill" in self.depth_method:
            self.bindings.append(self.dummyConst.data_ptr())

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
                mode="bilinear",
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
                mode="bilinear",
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
        compileMode: str = "default",
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
        self.compileMode = compileMode
        self.encodeBuffer = Queue(maxsize=10)

        self.handleModels()

        self.newHeight, self.newWidth = calculateAspectRatio(
            self.width, self.height, self.depthQuality
        )
        try:
            self.readBuffer = BuildBuffer(
                videoInput=self.input,
                inpoint=self.inpoint,
                outpoint=self.outpoint,
                width=self.width,
                height=self.height,
                half=self.half,
                resize=False,
                toTorch=False,
            )
            

            self.output = cv2.VideoWriter(
                self.output,
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps,
                (self.width, self.height),
            )
            with ThreadPoolExecutor(max_workers=3) as executor:
                executor.submit(self.readBuffer)
                executor.submit(self.encodeThread)
                executor.submit(self.process)

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
                toDownload = "base_v2"
            case "og_large_v2" | "og_distill_large_v2":
                method = "vitl"
                toDownload = "large_v2"
            case "og_giant_v2":
                method = "vitg"
                toDownload = "giant_v2"

        modelType = "pth"
        self.filename = modelsMap(model=toDownload, modelType=modelType, half=self.half)

        if not os.path.exists(os.path.join(weightsDir, toDownload, self.filename)):
            modelPath = downloadModels(
                model=toDownload,
                half=self.half,
                modelType=modelType,
            )

        else:
            modelPath = os.path.join(weightsDir, toDownload, self.filename)

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
            }
            if "distill" not in self.depth_method
            else {
                "encoder": "vitl",
                "features": 256,
                "out_channels": [256, 512, 1024, 1024],
                "use_bn": False,
                "use_clstoken": False,
                "max_depth": 150.0,
                "mode": "disparity",
                "pretrain_type": "dinov2",
                "del_mask_token": False,
            },
        }

        if "distill" in self.depth_method and "large" in self.depth_method:
            from src.depth.distillanydepth.modeling.archs.dam.dam import DepthAnything

            self.model = DepthAnything(**modelConfigs[method])
        else:
            self.model = DepthAnythingV2(**modelConfigs[method])

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

        if self.compileMode != "default":
            try:
                if self.compileMode == "max":
                    self.model.compile(mode="max-autotune-no-cudagraphs")
                elif self.compileMode == "max-graphs":
                    self.model.compile(
                        mode="max-autotune-no-cudagraphs", fullgraph=True
                    )
            except Exception as e:
                logging.error(
                    f"Error compiling model {self.interpolateMethod} with mode {self.compileMode}: {e}"
                )
                logAndPrint(
                    f"Error compiling model {self.interpolateMethod} with mode {self.compileMode}: {e}",
                    "red",
                )

            self.compileMode = "default"

    @torch.inference_mode()
    def processFrame(self, frame):
        try:
            depth = self.model.infer_image(frame, self.newHeight, self.half)
            self.encodeBuffer.put(depth)
        except Exception as e:
            logging.exception(f"Something went wrong while processing the frame, {e}")

    def process(self):
        frameCount = 0

        with ProgressBarLogic(self.totalFrames) as bar:
            print('hi')

            for _ in range(self.totalFrames):
                frame = self.readBuffer.read()
                if frame is None:
                    break
                self.processFrame(frame)
                frameCount += 1
                bar(1)

        logging.info(f"Processed {frameCount} frames")
        self.encodeBuffer.put(None)

    def encodeThread(self):
        while True:
            frame = self.encodeBuffer.get()
            if frame is None:
                break
            self.output.write(frame)

        self.output.release()


class OGDepthV2TensorRT:
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
        depth_method="og_small_v2",
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
        self.encodeBuffer = Queue(maxsize=10)

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

        self.newHeight, self.newWidth = calculateAspectRatio(
            self.width, self.height, self.depthQuality
        )
        try:
            self.readBuffer = BuildBuffer(
                videoInput=self.input,
                inpoint=self.inpoint,
                outpoint=self.outpoint,
                width=self.width,
                height=self.height,
                half=self.half,
                resize=False,
                toTorch=False,
            )

            self.outputWriter = cv2.VideoWriter(
                self.output,
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps,
                (self.width, self.height),
            )
            with ThreadPoolExecutor(max_workers=3) as executor:
                executor.submit(self.readBuffer)
                executor.submit(self.encodeThread)
                executor.submit(self.process)
        except Exception as e:
            logging.exception(f"Something went wrong, {e}")

    def handleModels(self):
        if "og_" in self.depth_method:
            self.depth_method = self.depth_method.replace("og_", "")

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
            inputName = "image" if "distill" not in self.depth_method else "input"
            self.engine, self.context = self.tensorRTEngineCreator(
                modelPath=self.modelPath,
                enginePath=enginePath,
                fp16=self.half,
                inputsMin=[1, 3, self.newHeight, self.newWidth],
                inputsOpt=[1, 3, self.newHeight, self.newWidth],
                inputsMax=[1, 3, self.newHeight, self.newWidth],
                inputName=[inputName],
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

        if "distill" in self.depth_method:
            self.dummyConst = torch.zeros(
                (1, 1, self.newHeight, self.newWidth),
                device=checker.device,
                dtype=torch.float16 if self.half else torch.float32,
            )

        self.bindings = [self.dummyInput.data_ptr(), self.dummyOutput.data_ptr()]

        if "distill" in self.depth_method:
            self.bindings.append(self.dummyConst.data_ptr())

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
            frame = torch.from_numpy(frame).to(checker.device)
            frame = frame.permute(2, 0, 1).unsqueeze(0)
            if self.half:
                frame = frame.half()
            else:
                frame = frame.float()

            frame = frame.mul(1 / 255)
            frame = F.interpolate(
                frame.float(),
                (self.newHeight, self.newWidth),
                mode="bilinear",
                align_corners=True,
            )
            frame = (frame - MEANTENSOR) / STDTENSOR
            if self.half:
                frame = frame.half()
            self.dummyInput.copy_(frame, non_blocking=True)
        self.normStream.synchronize()

    @torch.inference_mode()
    def normOutputFrame(self):
        depth = self.dummyOutput.cpu().numpy()
        depth = np.reshape(depth, (self.newHeight, self.newWidth))
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)

        return depth

    @torch.inference_mode()
    def processFrame(self, frame):
        try:
            self.normFrame(frame)
            with torch.cuda.stream(self.stream):
                self.cudaGraph.replay()
            self.stream.synchronize()
            depth = self.normOutputFrame()
            self.encodeBuffer.put(depth)
        except Exception as e:
            logging.exception(f"Something went wrong while processing the frame, {e}")

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
        self.encodeBuffer.put(None)


    def encodeThread(self):
        while True:
            frame = self.encodeBuffer.get()
            if frame is None:
                break
            if frame.ndim == 2:
                frame = np.stack([frame] * 3, axis=-1)
            frame = cv2.resize(
                frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR
            )
            self.outputWriter.write(frame)
        self.outputWriter.release()


class OGDepthV2DirectML:
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
        depth_method="og_small_v2-directml",
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
        self.encodeBuffer = Queue(maxsize=10)

        self.handleModels()

        try:
            self.readBuffer = BuildBuffer(
                videoInput=self.input,
                inpoint=self.inpoint,
                outpoint=self.outpoint,
                width=self.width,
                height=self.height,
                half=self.half,
                resize=False,
                toTorch=False,
            )

            self.outputWriter = cv2.VideoWriter(
                self.output,
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps,
                (self.width, self.height),
            )

            with ThreadPoolExecutor(max_workers=3) as executor:
                executor.submit(self.readBuffer)
                executor.submit(self.encodeThread)
                executor.submit(self.process)

        except Exception as e:
            logging.exception(f"Something went wrong, {e}")

    def handleModels(self):
        depth_method = self.depth_method
        if "og_" in depth_method:
            depth_method = depth_method.replace("og_", "")

        self.filename = modelsMap(
            model=depth_method, modelType="onnx", half=self.half
        )

        folderName = depth_method.replace("-directml", "-onnx")
        if not os.path.exists(os.path.join(weightsDir, folderName, self.filename)):
            modelPath = downloadModels(
                model=depth_method,
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
                "DirectML provider not available, falling back to CPU"
            )
            self.model = self.ort.InferenceSession(
                modelPath, providers=["CPUExecutionProvider"]
            )
        
        self.deviceType = "cpu"
        self.device = torch.device(self.deviceType)

        self.newHeight, self.newWidth = calculateAspectRatio(
            self.width, self.height, self.depthQuality
        )

        onnxInputs = self.model.get_inputs()
        onnxOutputs = self.model.get_outputs()
        self.inputName = onnxInputs[0].name
        self.outputName = onnxOutputs[0].name

        def onnxTypeToNumpy(typ: str):
            return np.float16 if "float16" in typ else np.float32

        def onnxTypeToTorch(typ: str):
            return torch.float16 if "float16" in typ else torch.float32

        self.numpyInDType = onnxTypeToNumpy(onnxInputs[0].type)
        self.torchInDType = onnxTypeToTorch(onnxInputs[0].type)
        self.numpyOutDType = onnxTypeToNumpy(onnxOutputs[0].type)
        self.torchOutDType = onnxTypeToTorch(onnxOutputs[0].type)

        self.IoBinding = self.model.io_binding()
        self.dummyInput = torch.zeros(
            (1, 3, self.newHeight, self.newWidth),
            device=self.deviceType,
            dtype=self.torchInDType,
        ).contiguous()

        out_rank = len(onnxOutputs[0].shape) if hasattr(onnxOutputs[0], "shape") else 4
        if out_rank == 3:
            out_shape = (1, self.newHeight, self.newWidth)
        else:
            out_shape = (1, 1, self.newHeight, self.newWidth)

        self.dummyOutput = torch.zeros(
            out_shape,
            device=self.deviceType,
            dtype=self.torchOutDType,
        ).contiguous()

        self.IoBinding.bind_output(
            name=self.outputName,
            device_type=self.deviceType,
            device_id=0,
            element_type=self.numpyOutDType,
            shape=self.dummyOutput.shape,
            buffer_ptr=self.dummyOutput.data_ptr(),
        )

    @torch.inference_mode()
    def processFrame(self, frame):
        try:
            frame = torch.from_numpy(frame).to(self.device)
            frame = frame.permute(2, 0, 1).unsqueeze(0)
            
            frame = F.interpolate(
                frame.float() / 255.0,
                size=(self.newHeight, self.newWidth),
                mode="bilinear",
                align_corners=True,
            )

            frame = (frame - MEANTENSOR.cpu()) / STDTENSOR.cpu()
            frame = frame.to(dtype=self.torchInDType)

            self.dummyInput.copy_(frame)
            self.IoBinding.bind_input(
                name=self.inputName,
                device_type=self.deviceType,
                device_id=0,
                element_type=self.numpyInDType,
                shape=self.dummyInput.shape,
                buffer_ptr=self.dummyInput.data_ptr(),
            )

            self.model.run_with_iobinding(self.IoBinding)

            out_tensor = self.dummyOutput.float()
            if out_tensor.ndim == 4:
                out_tensor = out_tensor.squeeze(1)
            
            depth = out_tensor[0].cpu().numpy()
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6) * 255.0
            depth = depth.astype(np.uint8)
            
            self.encodeBuffer.put(depth)

        except Exception as e:
            logging.exception(f"Something went wrong while processing the frame, {e}")

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
        self.encodeBuffer.put(None)

    def encodeThread(self):
        while True:
            frame = self.encodeBuffer.get()
            if frame is None:
                break
            if frame.ndim == 2:
                frame = np.stack([frame] * 3, axis=-1)
            frame = cv2.resize(
                frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR
            )
            self.outputWriter.write(frame)
        self.outputWriter.release()


class VideoDepthAnythingCUDA:
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
        depth_method="og_video_small_v2",
        custom_encoder="",
        benchmark=False,
        totalFrames=0,
        bitDepth: str = "16bit",
        depthQuality: str = "high",
        compileMode: str = "default",
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
        self.compileMode = compileMode

        try:
            self.output = cv2.VideoWriter(
                self.output,
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps,
                (self.width, self.height),
            )

            self.handleModels()

            with ThreadPoolExecutor(max_workers=2) as executor:
                executor.submit(self.process)

        except Exception as e:
            logging.exception(f"Something went wrong, {e}")

    def handleModels(self):
        from .video_depth_anything.video_depth import VideoDepthAnything

        self.filename = modelsMap(
            model=self.depth_method, modelType="pth", half=self.half
        )

        if not os.path.exists(
            os.path.join(weightsDir, self.depth_method, self.filename)
        ):
            modelPath = downloadModels(
                model=self.depth_method,
                half=self.half,
                modelType="pth",
            )
        else:
            modelPath = os.path.join(weightsDir, self.depth_method, self.filename)

        model_configs = {
            "vits": {
                "encoder": "vits",
                "features": 64,
                "out_channels": [48, 96, 192, 384],
            },
            "vitl": {
                "encoder": "vitl",
                "features": 256,
                "out_channels": [256, 512, 1024, 1024],
            },
        }

        if self.depth_method == "og_video_small_v2":
            encoder = "vits"
        elif self.depth_method == "og_video_large_v2":
            encoder = "vitl"
        self.model = VideoDepthAnything(**model_configs[encoder])

        self.model.load_state_dict(
            torch.load(modelPath, map_location="cpu"), strict=True
        )

        self.model = self.model.to(checker.device).eval()

        if self.half:
            self.model = self.model.half()

    def process(self):
        try:
            from .video_depth_anything.utils.dc_utils import read_video_frames

            frames, target_fps = read_video_frames(self.input, -1, self.fps, 1280)

            depths, fps = self.model.infer_video_depth(
                frames,
                target_fps,
                input_size=518,
                device="cuda",
                fp32=(not self.half),
            )

            print(depths.shape)

            d_min, d_max = depths.min(), depths.max()

            for i, depth in enumerate(depths):
                depth_norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
                depth_resized = cv2.resize(depth_norm * 255, (self.width, self.height))

                print(depth_resized.shape)
                depth_tensor = (
                    torch.from_numpy(depth_resized)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .float()
                    / 255.0
                )

                print(depth_tensor.shape)

            logging.info(f"Processed {len(depths)} frames with Video-Depth-Anything")
        except Exception as e:
            logging.exception(f"Something went wrong while processing the video, {e}")
            raise e


class OGDepthV3CUDA:
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
        depth_method="small_v3",
        custom_encoder="",
        benchmark=False,
        totalFrames=0,
        bitDepth: str = "16bit",
        depthQuality: str = "high",
        compileMode: str = "default",
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
        self.compileMode = compileMode
        self.encodeBuffer = Queue(maxsize=10)

        self.handleModels()

        self.newHeight, self.newWidth = calculateAspectRatio(
            self.width, self.height, self.depthQuality
        )
        try:
            self.readBuffer = BuildBuffer(
                videoInput=self.input,
                inpoint=self.inpoint,
                outpoint=self.outpoint,
                width=self.width,
                height=self.height,
                half=self.half,
                resize=False,
                toTorch=False,
            )

            self.outputWriter = cv2.VideoWriter(
                self.output,
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps,
                (self.width, self.height),
            )
            with ThreadPoolExecutor(max_workers=3) as executor:
                executor.submit(self.readBuffer)
                executor.submit(self.encodeThread)
                executor.submit(self.process)

        except Exception as e:
            logging.exception(f"Something went wrong, {e}")

    def handleModels(self):
        from .og_dpt_v3 import DepthAnythingV3

        match self.depth_method:
            case "small_v3":
                method = "vits"
                toDownload = "small_v3"
            case "base_v3":
                method = "vitb"
                toDownload = "base_v3"
            case "large_v3":
                method = "vitl"
                toDownload = "large_v3"
            case "giant_v3":
                method = "vitg"
                toDownload = "giant_v3"

        modelType = "pth"
        self.filename = modelsMap(model=toDownload, modelType=modelType, half=self.half)

        if not os.path.exists(os.path.join(weightsDir, toDownload, self.filename)):
            modelPath = downloadModels(
                model=toDownload,
                half=self.half,
                modelType=modelType,
            )
        else:
            modelPath = os.path.join(weightsDir, toDownload, self.filename)

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

        self.model = DepthAnythingV3(**modelConfigs[method])
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

        if self.compileMode != "default":
            try:
                if self.compileMode == "max":
                    self.model.compile(mode="max-autotune-no-cudagraphs")
                elif self.compileMode == "max-graphs":
                    self.model.compile(
                        mode="max-autotune-no-cudagraphs", fullgraph=True
                    )
            except Exception as e:
                logging.error(
                    f"Error compiling model {self.depth_method} with mode {self.compileMode}: {e}"
                )
                logAndPrint(
                    f"Error compiling model {self.depth_method} with mode {self.compileMode}: {e}",
                    "red",
                )

            self.compileMode = "default"

    @torch.inference_mode()
    def processFrame(self, frame):
        try:
            depth = self.model.infer_image(frame, self.newHeight, self.half)
            self.encodeBuffer.put(depth)
        except Exception as e:
            logging.exception(f"Something went wrong while processing the frame, {e}")

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
        self.encodeBuffer.put(None)


    def encodeThread(self):
        while True:
            frame = self.encodeBuffer.get()
            if frame is None:
                break
            self.outputWriter.write(frame)

        self.outputWriter.release()


class DepthDirectMLV3:
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
        depth_method="small_v3-directml",
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
        # Bind on CPU memory; ORT will handle copies for DML provider
        self.deviceType = "cpu"
        self.device = torch.device(self.deviceType)

        # Calculate padded model resolution (height, width)
        self.newHeight, self.newWidth = calculateAspectRatio(
            self.width, self.height, self.depthQuality
        )

        # Discover actual I/O names and dtypes from the ONNX model
        onnxInputs = self.model.get_inputs()
        onnxOutputs = self.model.get_outputs()
        self.inputName = onnxInputs[0].name
        self.outputName = onnxOutputs[0].name

        def onnxTypeToNumpy(typ: str):
            return np.float16 if "float16" in typ else np.float32

        def onnxTypeToTorch(typ: str):
            return torch.float16 if "float16" in typ else torch.float32

        self.numpyInDType = onnxTypeToNumpy(onnxInputs[0].type)
        self.torchInDType = onnxTypeToTorch(onnxInputs[0].type)
        self.numpyOutDType = onnxTypeToNumpy(onnxOutputs[0].type)
        self.torchOutDType = onnxTypeToTorch(onnxOutputs[0].type)

        # Allocate input/output buffers with correct shapes and dtypes
        self.IoBinding = self.model.io_binding()
        self.dummyInput = torch.zeros(
            (1, 3, self.newHeight, self.newWidth),
            device=self.deviceType,
            dtype=self.torchInDType,
        ).contiguous()

        out_rank = len(onnxOutputs[0].shape) if hasattr(onnxOutputs[0], "shape") else 4
        if out_rank == 3:
            out_shape = (1, self.newHeight, self.newWidth)
        else:
            # Default to NCHW with C=1 when unknown or 4D
            out_shape = (1, 1, self.newHeight, self.newWidth)

        self.dummyOutput = torch.zeros(
            out_shape,
            device=self.deviceType,
            dtype=self.torchOutDType,
        ).contiguous()

        self.IoBinding.bind_output(
            name=self.outputName,
            device_type=self.deviceType,
            device_id=0,
            element_type=self.numpyOutDType,
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
                mode="bilinear",
                align_corners=True,
            )

            # Cast to model's expected input dtype
            frame = frame.to(dtype=self.torchInDType)

            self.dummyInput.copy_(frame)
            self.IoBinding.bind_input(
                name=self.inputName,
                device_type=self.deviceType,
                device_id=0,
                element_type=self.numpyInDType,
                shape=self.dummyInput.shape,
                buffer_ptr=self.dummyInput.data_ptr(),
            )

            self.model.run_with_iobinding(self.IoBinding)

            # Bring output to float for normalization and resize to target resolution
            out_tensor = self.dummyOutput.float()
            if out_tensor.ndim == 3:
                out_tensor = out_tensor.unsqueeze(0)
            depth = F.interpolate(
                out_tensor,
                size=(self.height, self.width),
                mode="bilinear",
                align_corners=True,
            )

            # Normalize to 0-1 range
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
            self.writeBuffer.write(depth)

        except Exception as e:
            logging.exception(f"Something went wrong while processing the frame, {e}")

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


class DepthTensorRTV3:
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
        depth_method="small_v3-tensorrt",
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
                mode="bilinear",
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
                mode="bilinear",
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
