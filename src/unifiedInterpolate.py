import os
import torch
import logging
import tensorrt as trt

from torch.nn import functional as F
from .downloadModels import downloadModels, weightsDir, modelsMap
from .coloredPrints import yellow

torch.set_float32_matmul_precision("medium")


class RifeCuda:
    def __init__(
        self,
        half,
        width,
        height,
        interpolateMethod,
        ensemble=False,
        nt=1,
        interpolateFactor=2,
        sceneChange=False,
        sceneChangeThreshold=0.85,
    ):
        """
        Initialize the RIFE model

        Args:
            half (bool): Whether to use half precision.
            width (int): The width of the input frame.
            height (int): The height of the input frame.
            UHD (bool): Whether to use UHD mode.
            interpolateMethod (str): The method to use for interpolation.
            ensemble (bool): Whether to use ensemble mode.
            nt (int): The number of streams to use, not available for now.
            interpolateFactor (int): The interpolation factor.
            scenechange (bool): Whether to use scene change detection.
        """
        self.half = half
        self.scale = 1.0
        self.width = width
        self.height = height
        self.interpolateMethod = interpolateMethod
        self.ensemble = ensemble
        self.nt = nt
        self.interpolateFactor = interpolateFactor
        self.sceneChange = sceneChange
        self.sceneChangeThreshold = sceneChangeThreshold

        if self.width > 1920 or self.height > 1080:
            self.scale = 0.5
            if self.half:
                print(
                    yellow(
                        "UHD and fp16 are not compatible with RIFE, defaulting to fp32"
                    )
                )
                logging.info(
                    "UHD and fp16 for rife are not compatible due to flickering issues, defaulting to fp32"
                )
                self.half = False

        self.handle_model()

    def handle_model(self):
        """
        Load the desired model
        """

        self.filename = modelsMap(self.interpolateMethod)
        if not os.path.exists(os.path.join(weightsDir, "rife", self.filename)):
            modelPath = downloadModels(model=self.interpolateMethod)
        else:
            modelPath = os.path.join(weightsDir, "rife", self.filename)

        match self.interpolateMethod:
            case "rife" | "rife4.20":
                from .rifearches.IFNet_rife420 import IFNet
            case "rife4.18":
                from .rifearches.IFNet_rife418 import IFNet
            case "rife4.17":
                from .rifearches.IFNet_rife417 import IFNet
            case "rife4.15":
                from .rifearches.IFNet_rife415 import IFNet
            case "rife4.15-lite":
                from .rifearches.IFNet_rife415lite import IFNet
            case "rife4.16-lite":
                from .rifearches.IFNet_rife416lite import IFNet
            case "rife4.6":
                from .rifearches.IFNet_rife46 import IFNet

        self.model = IFNet(self.ensemble, self.scale, self.interpolateFactor)
        self.isCudaAvailable = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.isCudaAvailable else "cpu")

        torch.set_grad_enabled(False)
        if self.isCudaAvailable:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            if self.half:
                torch.set_default_dtype(torch.float16)

        if self.isCudaAvailable and self.half:
            self.model.half()
        else:
            self.half = False
            self.model.float()

        self.model.load_state_dict(torch.load(modelPath, map_location=self.device))
        self.model.eval().cuda() if self.isCudaAvailable else self.model.eval()
        self.model.to(self.device)

        ph = ((self.height - 1) // 64 + 1) * 64
        pw = ((self.width - 1) // 64 + 1) * 64
        self.padding = (0, pw - self.width, 0, ph - self.height)

        self.I0 = torch.zeros(
            1,
            3,
            self.height + self.padding[3],
            self.width + self.padding[1],
            dtype=torch.float16 if self.half else torch.float32,
            device=self.device,
        )

        self.I1 = torch.zeros(
            1,
            3,
            self.height + self.padding[3],
            self.width + self.padding[1],
            dtype=torch.float16 if self.half else torch.float32,
            device=self.device,
        )

        self.firstRun = True
        self.stream = torch.cuda.Stream() if self.isCudaAvailable else None

        if self.sceneChange:
            self.sceneChangeProcess = SceneChange(self.half, self.sceneChangeThreshold)
            self.sceneChangeCounter = 0

    @torch.inference_mode()
    def cacheFrame(self):
        self.I0.copy_(self.I1, non_blocking=True)
        self.model.cache()
        #if self.sceneChange:
        #    self.sceneChangeProcess.cacheFrame()

    @torch.inference_mode()
    def cacheFrameReset(self):
        self.I0.copy_(self.I1, non_blocking=True)
        self.model.cacheReset(self.I0)

    @torch.inference_mode()
    def processFrame(self, frame):
        return (
            frame.to(
                self.device,
                non_blocking=True,
                dtype=torch.float32 if not self.half else torch.float16,
            )
            .permute(2, 0, 1)
            .unsqueeze_(0)
            .mul_(1 / 255)
            .contiguous()
        )

    @torch.inference_mode()
    def padFrame(self, frame):
        return (
            F.pad(frame, [0, self.padding[1], 0, self.padding[3]])
            if self.padding != (0, 0, 0, 0)
            else frame
        )

    @torch.inference_mode()
    def run(self, frame, interpolateFactor, writeBuffer):
        with torch.cuda.stream(
            self.stream
        ) if self.isCudaAvailable else torch.no_grad():
            if self.firstRun:
                self.I0 = self.padFrame(self.processFrame(frame))
                self.firstRun = False
                return

            self.I1 = self.padFrame(self.processFrame(frame))

            if self.sceneChange:
                if self.sceneChangeProcess.run(self.I0, self.I1):
                    for _ in range(interpolateFactor - 1):
                        writeBuffer.write(frame)
                    self.stream.synchronize()
                    self.cacheFrameReset()
                    self.sceneChangeCounter += 1
                    return

            for i in range(interpolateFactor - 1):
                timestep = torch.full(
                    (1, 1, self.height + self.padding[3], self.width + self.padding[1]),
                    (i + 1) * 1 / interpolateFactor,
                    dtype=torch.float16 if self.half else torch.float32,
                    device=self.device,
                )
                output = self.model(self.I0, self.I1, timestep)
                output = output[:, :, : self.height, : self.width]
                output = output.mul(255.0).squeeze(0).permute(1, 2, 0)
                self.stream.synchronize()
                writeBuffer.write(output)

            self.cacheFrame()


    def getSceneChangeCounter(self):
        return self.sceneChangeCounter


class RifeTensorRT:
    def __init__(
        self,
        interpolateMethod: str = "rife4.20-tensorrt",
        interpolateFactor: int = 2,
        width: int = 0,
        height: int = 0,
        half: bool = True,
        ensemble: bool = False,
        nt: int = 1,
        sceneChange: bool = False,
        scneneChangeThreshold: float = 0.85,
    ):
        """
        Interpolates frames using TensorRT

        Args:
            interpolateMethod (str, optional): Interpolation method. Defaults to "rife415".
            interpolateFactor (int, optional): Interpolation factor. Defaults to 2.
            width (int, optional): Width of the frame. Defaults to 0.
            height (int, optional): Height of the frame. Defaults to 0.
            half (bool, optional): Half resolution. Defaults to True.
            ensemble (bool, optional): Ensemble. Defaults to False.
            nt (int, optional): Number of threads. Defaults to 1.
        """
        from polygraphy.backend.trt import (
            TrtRunner,
            engine_from_network,
            network_from_onnx_path,
            CreateConfig,
            Profile,
            EngineFromBytes,
            SaveEngine,
        )
        from polygraphy.backend.common import BytesFromPath

        self.TrtRunner = TrtRunner
        self.engine_from_network = engine_from_network
        self.network_from_onnx_path = network_from_onnx_path
        self.CreateConfig = CreateConfig
        self.Profile = Profile
        self.EngineFromBytes = EngineFromBytes
        self.SaveEngine = SaveEngine
        self.BytesFromPath = BytesFromPath

        self.interpolateMethod = interpolateMethod
        self.interpolateFactor = interpolateFactor
        self.width = width
        self.height = height
        self.half = half
        self.ensemble = ensemble
        self.nt = nt
        self.model = None
        self.sceneChange = sceneChange
        self.sceneChangeThreshold = scneneChangeThreshold

        if self.width > 1920 or self.height > 1080:
            if self.half:
                print(
                    yellow(
                        "UHD and fp16 are not compatible with RIFE, defaulting to fp32"
                    )
                )
                logging.info(
                    "UHD and fp16 for rife are not compatible due to flickering issues, defaulting to fp32"
                )
                self.half = False

        self.handleModel()

    def handleModel(self):
        self.filename = modelsMap(
            self.interpolateMethod,
            modelType="onnx",
            half=self.half,
            ensemble=self.ensemble,
        )

        if not os.path.exists(
            os.path.join(weightsDir, self.interpolateMethod, self.filename)
        ):
            modelPath = downloadModels(
                model=self.interpolateMethod,
                modelType="onnx",
                half=self.half,
                ensemble=self.ensemble,
            )
        else:
            modelPath = os.path.join(weightsDir, self.interpolateMethod, self.filename)

        self.isCudaAvailable = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.isCudaAvailable else "cpu")
        if self.isCudaAvailable:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            if self.half:
                torch.set_default_dtype(torch.float16)

        if "fp16" in modelPath:
            trtEngineModelPath = modelPath.replace(".onnx", "_fp16.engine")
        elif "fp32" in modelPath:
            trtEngineModelPath = modelPath.replace(".onnx", "_fp32.engine")
        else:
            trtEngineModelPath = modelPath.replace(".onnx", ".engine")

        if not os.path.exists(trtEngineModelPath):
            toPrint = f"Engine not found, creating dynamic engine for model: {modelPath}, this may take a while, but it is worth the wait..."
            print(yellow(toPrint))
            logging.info(toPrint)

            profile = [
                self.Profile().add(
                    "input",
                    min=(1, 7, 32, 32),
                    opt=(1, 7, self.height, self.width),
                    max=(1, 7, 2160, 3840),
                )
            ]

            self.config = self.CreateConfig(
                fp16=self.half,
                profiles=profile,
            )

            self.engine = self.engine_from_network(
                self.network_from_onnx_path(modelPath),
                config=self.config,
            )
            self.engine = self.SaveEngine(self.engine, trtEngineModelPath)
            self.engine.__call__()

        with open(trtEngineModelPath, "rb") as f, trt.Runtime(
            trt.Logger(trt.Logger.INFO)
        ) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()

        self.dType = torch.float16 if self.half else torch.float32
        self.stream = torch.cuda.Stream()
        self.I0 = torch.zeros(
            1,
            3,
            self.height,
            self.width,
            dtype=self.dType,
            device=self.device,
        )

        self.I1 = torch.zeros(
            1,
            3,
            self.height,
            self.width,
            dtype=self.dType,
            device=self.device,
        )

        self.dummyInput = torch.empty(
            (1, 7, self.height, self.width),
            device=self.device,
            dtype=torch.float16 if self.half else torch.float32,
        )

        self.dummyOutput = torch.zeros(
            (1, 3, self.height, self.width),
            device=self.device,
            dtype=torch.float16 if self.half else torch.float32,
        )

        self.bindings = [self.dummyInput.data_ptr(), self.dummyOutput.data_ptr()]

        for i in range(self.engine.num_io_tensors):
            self.context.set_tensor_address(
                self.engine.get_tensor_name(i), self.bindings[i]
            )
            tensor_name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(tensor_name, self.dummyInput.shape)

        self.firstRun = True
        if self.sceneChange:
            self.sceneChangeProcess = SceneChangeTensorRT(
                self.half, self.sceneChangeThreshold
            )

        self.dataTransferStream = torch.cuda.Stream()
        self.inferStream = torch.cuda.Stream()

    @torch.inference_mode()
    def processFrame(self, frame):
        return (
            frame
            .permute(2, 0, 1)
            .unsqueeze_(0)
            .mul_(1 / 255)
            .contiguous()
        )

    @torch.inference_mode()
    def cacheFrame(self):
        self.I0.copy_(self.I1, non_blocking=True)
        #if self.sceneChange:
        #    self.sceneChangeProcess.cacheFrame()

    @torch.inference_mode()
    def run(self, frame, interpolateFactor, writeBuffer):
        with torch.cuda.stream(self.dataTransferStream):
            frame = frame.to(self.device, non_blocking=True, dtype=torch.float32 if not self.half else torch.float16)

        with torch.cuda.stream(self.stream):
            if self.firstRun:
                self.I0.copy_(self.processFrame(frame), non_blocking=True)
                self.firstRun = False
                return
            
            self.I1.copy_(self.processFrame(frame), non_blocking=True)
            if self.sceneChange:
                if self.sceneChangeProcess.run(self.I0, self.I1):
                    for _ in range(interpolateFactor - 1):
                        writeBuffer.write(frame)
                    self.stream.synchronize()
                    self.cacheFrame()
                    return

            for i in range(interpolateFactor - 1):
                timestep = torch.full(
                    (1, 1, self.height, self.width),
                    (i + 1) * 1 / interpolateFactor,
                    dtype=self.dType,
                    device=self.device,
                ).contiguous()

                self.dummyInput.copy_(
                    torch.cat([self.I0, self.I1, timestep], dim=1), non_blocking=True
                )
                self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
                output = self.dummyOutput.squeeze_(0).permute(1, 2, 0).mul_(255)
                self.stream.synchronize()
                writeBuffer.write(output)

            self.cacheFrame()

class RifeNCNN:
    def __init__(
        self,
        interpolateMethod,
        ensemble=False,
        nt=1,
        width=1920,
        height=1080,
        half=True,
        sceneChange=False,
        sceneChangeThreshold=0.85,
    ):
        self.interpolateMethod = interpolateMethod
        self.nt = nt
        self.height = height
        self.width = width
        self.ensemble = ensemble
        self.half = half
        self.sceneChange = sceneChange
        self.sceneChangeThreshold = sceneChangeThreshold

        UHD = True if width >= 3840 or height >= 2160 else False
        scale = 2 if UHD else 1

        from rife_ncnn_vulkan_python import Rife

        match interpolateMethod:
            case "rife4.15-ncnn" | "rife-ncnn":
                self.interpolateMethod = "rife-v4.15-ncnn"
            case "rife4.6-ncnn":
                self.interpolateMethod = "rife-v4.6-ncnn"
            case "rife4.15-lite-ncnn":
                self.interpolateMethod = "rife-v4.15-lite-ncnn"
            case "rife4.16-lite-ncnn":
                self.interpolateMethod = "rife-v4.16-lite-ncnn"
            case "rife4.17-ncnn":
                self.interpolateMethod = "rife-v4.17-ncnn"
            case "rife4.18-ncnn":
                self.interpolateMethod = "rife-v4.18-ncnn"

        self.filename = modelsMap(
            self.interpolateMethod,
            ensemble=self.ensemble,
        )

        if self.filename.endswith("-ncnn.zip"):
            self.filename = self.filename[:-9]
        elif self.filename.endswith("-ncnn"):
            self.filename = self.filename[:-5]

        if not os.path.exists(
            os.path.join(weightsDir, self.interpolateMethod, self.filename)
        ):
            modelPath = downloadModels(
                model=self.interpolateMethod,
                ensemble=self.ensemble,
            )
        else:
            modelPath = os.path.join(weightsDir, self.interpolateMethod, self.filename)

        if modelPath.endswith("-ncnn.zip"):
            modelPath = modelPath[:-9]
        elif modelPath.endswith("-ncnn"):
            modelPath = modelPath[:-5]

        self.rife = Rife(
            gpuid=0,
            model=modelPath,
            scale=scale,
            tta_mode=False,
            tta_temporal_mode=False,
            uhd_mode=UHD,
            num_threads=self.nt,
        )

        self.frame1 = None
        self.shape = (self.height, self.width)

        if self.sceneChange:
            self.sceneChangeProcess = SceneChange(self.half, self.sceneChangeThreshold)

    def cacheFrame(self):
        self.frame1 = self.frame2
        #if self.sceneChange:
        #    self.sceneChangeProcess.cacheFrame()

    def run(self, frame, interpolateFactor, writeBuffer):
        if self.frame1 is None:
            self.frame1 = frame.cpu().numpy().astype("uint8")
            return False

        self.frame2 = frame.cpu().numpy().astype("uint8")

        if self.sceneChange:
            if self.sceneChangeProcess.runNumpy(self.frame1, self.frame2):
                for _ in range(interpolateFactor - 1):
                    writeBuffer.write(frame)
                self.cacheFrame()
                return

        for i in range(interpolateFactor - 1):
            timestep = (i + 1) * 1 / interpolateFactor

            output = self.rife.process_cv2(self.frame1, self.frame2, timestep=timestep)

            output = torch.from_numpy(output).to(frame.device)
            writeBuffer.write(output)

        self.cacheFrame()



class SceneChange:
    def __init__(
        self,
        half,
        sceneChangeThreshold,
    ):
        self.half = half

        import onnxruntime as ort
        import cv2
        import numpy as np

        self.ort = ort
        self.cv2 = cv2
        self.np = np
        self.sceneChangeThreshold = sceneChangeThreshold

        self.loadModel()

    def loadModel(self):
        filename = modelsMap(
            "scenechange",
            half=self.half,
        )

        if not os.path.exists(os.path.join(weightsDir, "scenechange", filename)):
            modelPath = downloadModels(
                "scenechange",
                half=self.half,
            )

        else:
            modelPath = os.path.join(weightsDir, "scenechange", filename)

        logging.info(f"Loading scenechange detection model from {modelPath}")

        providers = self.ort.get_available_providers()
        if "DmlExecutionProvider" in providers:
            logging.info(
                "DirectML provider available for scenechange detection. Defaulting to DirectML"
            )
            self.model = self.ort.InferenceSession(
                modelPath, providers=["DmlExecutionProvider"]
            )
        else:
            logging.info(
                "DirectML provider not available for scenechange detection, falling back to CPU, expect significantly worse performance, ensure that your drivers are up to date and your GPU supports DirectX 12"
            )
            self.model = self.ort.InferenceSession(
                modelPath, providers=["CPUExecutionProvider"]
            )

        self.I0 = None
        self.I1 = None

    @torch.inference_mode()
    def processFrameTorch(self, frame):
        return (
            F.interpolate(
                frame.half() if self.half else frame.float(),
                size=(224, 224),
                mode="bilinear",
            )
            .contiguous()
            .squeeze_(0)
            .cpu()
            .numpy()
        )

    def processFrameNumpy(self, frame):
        return (
            self.cv2.resize(frame, (224, 224)).transpose(2, 0, 1).astype(self.np.float16)
            if self.half
            else self.cv2.resize(frame, (224, 224))
            .transpose(2, 0, 1)
            .astype(self.np.float32)
        )

    def cacheFrame(self):
        #self.I0 = self.I1
        pass

    @torch.inference_mode()
    def run(self, I0, I1):
        #if self.I0 is None:
        #    self.I0 = self.processFrameTorch(I1)
        #    return False

        self.I0 = self.processFrameTorch(I0)
        self.I1 = self.processFrameTorch(I1)
        inputs = self.np.concatenate((self.I0, self.I1), 0)
        return (
            self.model.run(None, {"input": inputs})[0][0][0] > self.sceneChangeThreshold
        )

    def runNumpy(self, frame1, frame2):
        if self.I0 is None:
            self.I0 = self.processFrameNumpy(frame1)
            return False

        self.I1 = self.processFrameNumpy(frame2)

        inputs = self.np.ascontiguousarray(self.np.concatenate((self.I0, self.I1), 0))
        return self.model.run(None, {"input": inputs})[0][0][0] > self.sceneChangeThreshold


class SceneChangeTensorRT:
    def __init__(self, half, sceneChangeThreshold=0.85):
        self.half = half
        self.sceneChangeThreshold = sceneChangeThreshold
        
        from polygraphy.backend.trt import (
            TrtRunner,
            engine_from_network,
            network_from_onnx_path,
            CreateConfig,
            Profile,
            EngineFromBytes,
            SaveEngine,
        )
        from polygraphy.backend.common import BytesFromPath

        self.TrtRunner = TrtRunner
        self.engine_from_network = engine_from_network
        self.network_from_onnx_path = network_from_onnx_path
        self.CreateConfig = CreateConfig
        self.Profile = Profile
        self.EngineFromBytes = EngineFromBytes
        self.SaveEngine = SaveEngine
        self.BytesFromPath = BytesFromPath
        

        self.handleModel()

    def handleModel(self):
        filename = modelsMap(
            "scenechange",
            half=self.half,
        )

        if not os.path.exists(os.path.join(weightsDir, "scenechange", filename)):
            modelPath = downloadModels(
                "scenechange",
                half=self.half,
            )

        else:
            modelPath = os.path.join(weightsDir, "scenechange", filename)

        if self.half:
            trtEngineModelPath = modelPath.replace(".onnx", "_fp16.engine")
        else:
            trtEngineModelPath = modelPath.replace(".onnx", "_fp32.engine")

        if not os.path.exists(trtEngineModelPath):
            toPrint = f"Engine not found, creating dynamic engine for model: {modelPath}, this may take a while, but it is worth the wait..."
            print(yellow(toPrint))
            logging.info(toPrint)

            profile = [
                self.Profile().add(
                    "input",
                    min=(6, 224, 224),
                    opt=(6, 224, 224),
                    max=(6, 224, 224),
                )
            ]

            self.config = self.CreateConfig(
                fp16=self.half,
                profiles=profile,
                preview_features=[],
            )

            self.engine = self.engine_from_network(
                self.network_from_onnx_path(modelPath),
                config=self.config,
            )
            self.engine = self.SaveEngine(self.engine, trtEngineModelPath)
            self.engine.__call__()

        with open(trtEngineModelPath, "rb") as f, trt.Runtime(
            trt.Logger(trt.Logger.INFO)
        ) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()

        self.dType = torch.float16 if self.half else torch.float32
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stream = torch.cuda.Stream()
        self.dummyInput = torch.zeros(
            (6, 224, 224),
            device=self.device,
            dtype=self.dType,
        )
        self.dummyOutput = torch.zeros(
            (1, 2),
            device=self.device,
            dtype=self.dType,
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
            for _ in range(50):
                self.dummyInput.copy_(
                    torch.zeros(6, 224, 224, device=self.device, dtype=self.dType)
                )
                self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
                self.stream.synchronize()

        self.I0 = torch.zeros(3, 224, 224, device=self.device, dtype=self.dType)
        self.I1 = torch.zeros(3, 224, 224, device=self.device, dtype=self.dType)

    @torch.inference_mode()
    def processFrame(self, frame):
        return (
            F.interpolate(
                frame.half() if self.half else frame.float(),
                size=(224, 224),
                mode="bilinear",
            )
            .contiguous()
            .squeeze(0)
        )

    @torch.inference_mode()
    def cacheFrame(self):
        self.I0.copy_(self.I1, non_blocking=True)

    @torch.inference_mode()
    def run(self, I0, I1):
        with torch.cuda.stream(self.stream):
            #if self.I0 is None:
            #    self.I0 = self.processFrame(I1)
            #    return False
            
            self.I0.copy_(self.processFrame(I0), non_blocking=True)
            self.I1.copy_(self.processFrame(I1), non_blocking=True)

            self.dummyInput.copy_(
                torch.cat([self.I0, self.I1], dim=0), non_blocking=True
            )
            self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
            self.stream.synchronize()
            return self.dummyOutput[0][0].item() > self.sceneChangeThreshold
