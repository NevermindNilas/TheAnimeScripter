import os
import torch
import logging
import torch.nn.functional as F
import math
import numpy as np

from src.model.downloadModels import downloadModels, weightsDir, modelsMap, resolveWeightPath
from src.infra.isCudaInit import CudaChecker
from src.infra.logAndPrint import logAndPrint

from src.constants import ADOBE

if ADOBE:
    from src.server.aeComms import progressState


checker = CudaChecker()

torch.set_float32_matmul_precision("medium")


_RIFE_V1 = {
    "rife":           ("IFNet425",      "IFNet_rife425"),
    "rife4.25":       ("IFNet425",      "IFNet_rife425"),
    "rife4.25-heavy": ("IFNet425Heavy", "IFNet_rife425heavy"),
    "rife4.25-lite":  ("IFNet425Lite",  "IFNet_rife425lite"),
    "rife4.22":       ("IFNet422",      "IFNet_rife422"),
    "rife4.22-lite":  ("IFNet422Lite",  "IFNet_rife422lite"),
    "rife4.21":       ("IFNet421",      "IFNet_rife421"),
    "rife4.20":       ("IFNet420",      "IFNet_rife420"),
    "rife4.18":       ("IFNet418",      "IFNet_rife418"),
    "rife4.17":       ("IFNet417",      "IFNet_rife417"),
    "rife4.15-lite":  ("IFNet415Lite",  "IFNet_rife415lite"),
    "rife4.16-lite":  ("IFNet416Lite",  "IFNet_rife416lite"),
    "rife4.6":        ("IFNet46",       "IFNet_rife46"),
    "rife_elexor":    (None,            "IFNet_elexor_cuda"),
}


def _loadV1(method, half):
    fastName, baseMod = _RIFE_V1[method]
    if half and fastName:
        from src.rifearches import rife_fast
        return getattr(rife_fast, fastName)
    mod = __import__(f"src.rifearches.{baseMod}", fromlist=["IFNet"])
    return mod.IFNet


def importRifeArch(interpolateMethod, version, half=True):
    match version:
        case "v1":
            return _loadV1(interpolateMethod, half)

        case "v3":
            match interpolateMethod:
                case "rife4.25-heavy-tensorrt":
                    from src.rifearches.Rife425_heavy_v3 import IFNet

                    Head = True

                case "rife4.25-lite-tensorrt":
                    from src.rifearches.Rife425_lite_v3 import IFNet

                    Head = True
                case "rife4.25-tensorrt":
                    from src.rifearches.Rife425_v3 import IFNet

                    Head = True
                case "rife4.22-tensorrt":
                    from src.rifearches.Rife422_v3 import IFNet

                    Head = True
                case "rife4.22-lite-tensorrt":
                    from src.rifearches.Rife422_lite_v3 import IFNet

                    Head = True
                case "rife4.21-tensorrt":
                    from src.rifearches.Rife422_v3 import IFNet

                    Head = True
                case "rife4.20-tensorrt":
                    from src.rifearches.Rife420_v3 import IFNet

                    Head = True
                case "rife4.18-tensorrt":
                    from src.rifearches.Rife415_v3 import IFNet

                    Head = True
                case "rife4.17-tensorrt":
                    from src.rifearches.Rife415_v3 import IFNet

                    Head = True
                case "rife4.15-tensorrt":
                    from src.rifearches.Rife415_v3 import IFNet

                    Head = True
                case "rife4.6-tensorrt":
                    from src.rifearches.Rife46_v3 import IFNet

                    Head = False
                case "rife4.6-directml" | "rife4.6-openvino":
                    from src.rifearches.Rife_directml import IFNet_46 as IFNet

                    Head = False
                case "rife4.22-directml" | "rife4.22-openvino":
                    from src.rifearches.Rife_directml import IFNet_422 as IFNet

                    Head = True
                case (
                    "rife4.15-directml"
                    | "rife4.17-directml"
                    | "rife4.18-directml"
                    | "rife4.15-openvino"
                    | "rife4.17-openvino"
                    | "rife4.18-openvino"
                ):
                    from src.rifearches.Rife_directml import IFNet_415 as IFNet

                    Head = True
                case (
                    "rife4.20-directml"
                    | "rife4.21-directml"
                    | "rife4.20-openvino"
                    | "rife4.21-openvino"
                ):
                    from src.rifearches.Rife_directml import IFNet_420 as IFNet

                    Head = True
                case "rife4.22-lite-directml" | "rife4.22-lite-openvino":
                    from src.rifearches.Rife_directml import IFNet_422_lite as IFNet

                    Head = True
                case "rife4.25-directml" | "rife4.25-openvino":
                    from src.rifearches.Rife_directml import IFNet_425 as IFNet

                    Head = True
                case "rife4.25-lite-directml" | "rife4.25-lite-openvino":
                    from src.rifearches.Rife_directml import IFNet_425_lite as IFNet

                    Head = True
                case "rife4.25-heavy-directml" | "rife4.25-heavy-openvino":
                    from src.rifearches.Rife_directml import IFNet_425_heavy as IFNet

                    Head = True
                case "rife_elexor-tensorrt":
                    from src.rifearches.IFNet_elexor_tensorrt import IFNet

                    Head = True
            return IFNet, Head


class DistilDRBACuda:
    def __init__(
        self,
        half: bool,
        width: int,
        height: int,
        interpolateMethod: str,
        interpolateFactor: int = 2,
        compileMode: str = None,
    ):
        """
        Initialize the DistilDRBA model.

        Args:
            half: Use half precision (fp16)
            width: Frame width
            height: Frame height
            interpolateMethod: "distildrba" (full) or "distildrba-lite"
            interpolateFactor: Interpolation factor
            compileMode: Optional torch.compile mode
        """
        self.half = half
        self.width = width
        self.height = height
        self.interpolateMethod = interpolateMethod
        self.interpolateFactor = interpolateFactor
        self.compileMode = compileMode
        self.lite = "lite" in interpolateMethod

        if width > 1920 or height > 1080:
            self.scale = 0.5
        else:
            self.scale = 1.0

        self.dType = torch.float16 if self.half else torch.float32
        self.device = checker.device

        self.padSize = 64
        ph = ((self.height - 1) // self.padSize + 1) * self.padSize
        pw = ((self.width - 1) // self.padSize + 1) * self.padSize
        self.padding = (0, pw - self.width, 0, ph - self.height)
        self.paddedHeight = ph
        self.paddedWidth = pw

        self.f0 = None
        self.f1 = None
        self.f2 = None

        self.I0 = None

        self.firstRun = True
        self.stream = torch.cuda.Stream() if checker.cudaAvailable else None
        self.normStream = torch.cuda.Stream() if checker.cudaAvailable else None

        self.loadModel()

    def loadModel(self):
        """Load the DistilDRBA model weights."""
        from src.rifearches.IFNet_distildrba import IFNet

        self.filename = modelsMap(self.interpolateMethod)
        folderPath = os.path.join(weightsDir, self.interpolateMethod)
        if not os.path.exists(os.path.join(folderPath, self.filename)):
            modelPath = downloadModels(model=self.interpolateMethod)
        else:
            modelPath = os.path.join(folderPath, self.filename)

        self.model = IFNet(lite=self.lite, scale=self.scale)

        stateDict = torch.load(modelPath, map_location="cpu", weights_only=True)

        if "model" in stateDict:
            stateDict = stateDict["model"]

        self.model.load_state_dict(stateDict, strict=False)
        del stateDict

        if self.half:
            self.model = self.model.half()
        else:
            self.model = self.model.float()

        self.model = self.model.eval()
        self.model = self.model.to(self.device)
        self.model = self.model.to(memory_format=torch.channels_last)
        if checker.cudaAvailable:
            torch.cuda.empty_cache()

        logging.info(
            f"DistilDRBA {'Lite' if self.lite else 'Full'} model loaded, "
            f"scale={self.scale}, half={self.half}"
        )

    def padFrame(self, frame: torch.Tensor) -> torch.Tensor:
        """Pad frame to required dimensions using replicate padding."""
        if self.padding != (0, 0, 0, 0):
            return F.pad(frame, self.padding, mode="replicate")
        return frame

    def prepareFrame(self, frame: torch.Tensor) -> torch.Tensor:
        """Prepare a frame for inference (transfer, cast, pad)."""
        frame = frame.to(device=self.device, dtype=self.dType, non_blocking=True)
        frame = self.padFrame(frame)
        return frame.to(memory_format=torch.channels_last)

    @torch.inference_mode()
    def cacheFrame(self, frame: torch.Tensor):
        """
        Update the frame cache without producing output.
        Used when a frame needs to be cached but not interpolated (e.g., after dedup).
        """
        if self.normStream:
            with torch.cuda.stream(self.normStream):
                self.I0 = self.prepareFrame(frame)
            self.normStream.synchronize()
        else:
            self.I0 = self.prepareFrame(frame)

        self.f0 = self.f1
        self.f1 = self.f2
        self.f2 = None

    @torch.inference_mode()
    def cacheFrameReset(self, frame: torch.Tensor):
        """
        Reset feature cache on scene change.
        Called when scene detection triggers a scene change.
        """
        self.f0 = None
        self.f1 = None
        self.f2 = None
        # Reset frame buffers - start fresh with this frame as I0
        self.I0 = self.prepareFrame(frame)
        self.firstRun = True

    @torch.inference_mode()
    def __call__(
        self,
        frame: torch.Tensor,
        nextFrame: torch.Tensor,
        interpQueue,
        framesToInsert: int = 2,
        timesteps=None,
    ):
        """
        Interpolate frames using 3-frame DistilDRBA model.

        Args:
            frame: Current frame tensor [B, C, H, W] - this becomes I1
            nextFrame: Next frame from peek() [B, C, H, W] - this becomes I2
            interpQueue: Queue to put interpolated frames
            framesToInsert: Number of frames to insert between consecutive frames
            timesteps: Optional list of timestep values (in TAS range 0-1)

        Frame mapping using peek():
            - I0 = previous frame (cached from last call)
            - I1 = current frame (passed as `frame`)
            - I2 = next frame (passed as `nextFrame` from readBuffer.peek())

        We interpolate between I0 and I1, using I2 as future context.
        """
        # Prepare current frame (I1)
        if self.normStream:
            with torch.cuda.stream(self.normStream):
                I1 = self.prepareFrame(frame)
            self.normStream.synchronize()
        else:
            I1 = self.prepareFrame(frame)

        # Prepare next frame (I2) - handle size mismatch if upscale was applied to frame
        if nextFrame is not None:
            # Check if nextFrame needs to be resized to match frame dimensions
            if nextFrame.shape[-2:] != frame.shape[-2:]:
                nextFrame = F.interpolate(
                    nextFrame,
                    size=frame.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            if self.normStream:
                with torch.cuda.stream(self.normStream):
                    I2 = self.prepareFrame(nextFrame)
                self.normStream.synchronize()
            else:
                I2 = self.prepareFrame(nextFrame)
        else:
            I2 = I1

        if self.firstRun:
            self.I0 = I1
            self.firstRun = False
            return

        if self.I0.shape[-2:] != I1.shape[-2:]:
            self.I0 = F.interpolate(
                self.I0, size=I1.shape[-2:], mode="bilinear", align_corners=False
            )
            self.f0 = None
            self.f1 = None
            self.f2 = None

        for i in range(framesToInsert):
            if timesteps is not None and i < len(timesteps):
                t = timesteps[i]
            else:
                t = (i + 1) / (framesToInsert + 1)

            tDrba = 0.5 + t * 0.5 - 0.0001  # Small epsilon to avoid exactly 1.0

            if self.stream:
                with torch.cuda.stream(self.stream):
                    output, self.f0, self.f1, self.f2 = self.model(
                        self.I0,
                        I1,
                        I2,
                        self.f0,
                        self.f1,
                        self.f2,
                        tDrba,
                    )
                self.stream.synchronize()
            else:
                output, self.f0, self.f1, self.f2 = self.model(
                    self.I0,
                    I1,
                    I2,
                    self.f0,
                    self.f1,
                    self.f2,
                    tDrba,
                )

            output = output[:, :, : self.height, : self.width]
            interpQueue.put(output)

        self.f0 = self.f1
        self.f1 = self.f2
        self.f2 = None
        self.I0 = I1


class DistilDRBATensorRT:
    def __init__(
        self,
        half: bool,
        width: int,
        height: int,
        interpolateMethod: str,
        interpolateFactor: int = 2,
    ):
        """
        Initialize DistilDRBA TensorRT model.

        Args:
            half: Use half precision (fp16)
            width: Frame width
            height: Frame height
            interpolateMethod: "distildrba-tensorrt" or "distildrba-lite-tensorrt"
            interpolateFactor: Interpolation factor
        """
        import tensorrt as trt
        from src.model.trtHandler import (
            tensorRTEngineCreator,
            tensorRTEngineLoader,
        )

        self.tensorRTEngineCreator = tensorRTEngineCreator
        self.tensorRTEngineLoader = tensorRTEngineLoader
        self.trt = trt

        self.half = half
        self.width = width
        self.height = height
        self.interpolateMethod = interpolateMethod
        self.interpolateFactor = interpolateFactor
        self.lite = "lite" in interpolateMethod

        if width > 1920 or height > 1080:
            self.scale = 0.5
        else:
            self.scale = 1.0

        self.dtype = torch.float16 if self.half else torch.float32
        self.device = checker.device

        self.padSize = 64
        self.ph = ((self.height - 1) // self.padSize + 1) * self.padSize
        self.pw = ((self.width - 1) // self.padSize + 1) * self.padSize
        self.padding = (0, self.pw - self.width, 0, self.ph - self.height)

        self.handleModel()

    def handleModel(self):
        """Initialize TensorRT engine for DistilDRBA."""
        from src.rifearches.IFNet_distildrba import IFNet

        baseMethod = self.interpolateMethod.replace("-tensorrt", "")
        self.filename = modelsMap(baseMethod)
        folderPath = os.path.join(weightsDir, baseMethod)

        if not os.path.exists(os.path.join(folderPath, self.filename)):
            modelPath = downloadModels(model=baseMethod)
        else:
            modelPath = os.path.join(folderPath, self.filename)

        self.model = IFNet(lite=self.lite, scale=self.scale)
        stateDict = torch.load(modelPath, map_location="cpu", weights_only=True)
        if "model" in stateDict:
            stateDict = stateDict["model"]
        self.model.load_state_dict(stateDict, strict=False)
        del stateDict

        if self.half:
            self.model = self.model.half()
        else:
            self.model = self.model.float()
        self.model = self.model.eval()
        self.model = self.model.to(self.device)
        if checker.cudaAvailable:
            torch.cuda.empty_cache()

        enginePrecision = "fp16" if self.half else "fp32"
        engineName = f"_{enginePrecision}_{self.height}x{self.width}.engine"
        enginePath = modelPath.replace(".pkl", "") + engineName

        self.engine, self.context = self.tensorRTEngineLoader(enginePath)

        if (
            self.engine is None
            or self.context is None
            or not os.path.exists(enginePath)
        ):
            self.createEngine(modelPath, enginePath)

        self.setupTensors()
        self.initCudaGraph()

    def createEngine(self, modelPath, enginePath):
        """Create TensorRT engine from ONNX export."""
        from src.rifearches.IFNet_distildrba_tensorrt import IFNetLiteTRT, IFNetFullTRT

        if self.lite:
            trtModel = IFNetLiteTRT(scale=self.scale)
        else:
            trtModel = IFNetFullTRT(scale=self.scale)
        srcStateDict = self.model.state_dict()
        trtModel.load_state_dict(srcStateDict, strict=False)
        del srcStateDict

        if self.half:
            trtModel.half()
        else:
            trtModel.float()
        trtModel.eval()
        trtModel = trtModel.to(self.device)
        if checker.cudaAvailable:
            torch.cuda.empty_cache()

        dummyImg0 = torch.zeros(
            1, 3, self.ph, self.pw, dtype=self.dtype, device=self.device
        )
        dummyImg1 = torch.zeros(
            1, 3, self.ph, self.pw, dtype=self.dtype, device=self.device
        )
        dummyImg2 = torch.zeros(
            1, 3, self.ph, self.pw, dtype=self.dtype, device=self.device
        )
        dummyTimestep = torch.full(
            (1, 1, self.ph, self.pw), 0.75, dtype=self.dtype, device=self.device
        )

        onnxPath = modelPath.replace(".pkl", "_trt.onnx")

        inputList = [dummyImg0, dummyImg1, dummyImg2, dummyTimestep]
        inputNames = ["img0", "img1", "img2", "timestep"]
        outputNames = ["output"]
        dynamicAxes = {
            "img0": {2: "height", 3: "width"},
            "img1": {2: "height", 3: "width"},
            "img2": {2: "height", 3: "width"},
            "timestep": {2: "height", 3: "width"},
            "output": {2: "height", 3: "width"},
        }

        torch.onnx.export(
            trtModel,
            tuple(inputList),
            onnxPath,
            input_names=inputNames,
            output_names=outputNames,
            dynamic_axes=dynamicAxes,
            opset_version=22,
            optimize=False,
            dynamo=False,
        )

        inputs = [
            [1, 3, self.ph, self.pw],
            [1, 3, self.ph, self.pw],
            [1, 3, self.ph, self.pw],
            [1, 1, self.ph, self.pw],
        ]

        # Cleanup
        del trtModel, dummyImg0, dummyImg1, dummyImg2, dummyTimestep
        import gc

        gc.collect()
        torch.cuda.empty_cache()

        inputsMin = inputsOpt = inputsMax = inputs

        logging.info(
            f"Creating TensorRT engine for DistilDRBA-{'Lite' if self.lite else 'Full'}"
        )
        self.engine, self.context = self.tensorRTEngineCreator(
            modelPath=onnxPath,
            enginePath=enginePath,
            fp16=self.half,
            inputsMin=inputsMin,
            inputsOpt=inputsOpt,
            inputsMax=inputsMax,
            inputName=inputNames,
            isMultiInput=True,
        )

        try:
            os.remove(onnxPath)
        except Exception as e:
            logging.error(f"Error removing onnx model: {e}")

    def setupTensors(self):
        """Setup persistent tensors for TensorRT execution."""
        self.stream = torch.cuda.Stream()
        self.normStream = torch.cuda.Stream()
        self.outputStream = torch.cuda.Stream()

        self.tImg0 = torch.zeros(
            1, 3, self.ph, self.pw, dtype=self.dtype, device=self.device
        )
        self.tImg1 = torch.zeros(
            1, 3, self.ph, self.pw, dtype=self.dtype, device=self.device
        )
        self.tImg2 = torch.zeros(
            1, 3, self.ph, self.pw, dtype=self.dtype, device=self.device
        )
        self.tTimestep = torch.full(
            (1, 1, self.ph, self.pw), 0.75, dtype=self.dtype, device=self.device
        )

        self.tOutput = torch.zeros(
            1, 3, self.ph, self.pw, dtype=self.dtype, device=self.device
        )

        self.tensors = [
            self.tImg0,
            self.tImg1,
            self.tImg2,
            self.tTimestep,
            self.tOutput,
        ]
        self.bindings = [tensor.data_ptr() for tensor in self.tensors]

        for i in range(self.engine.num_io_tensors):
            tensorName = self.engine.get_tensor_name(i)
            self.context.set_tensor_address(tensorName, self.bindings[i])
            if self.engine.get_tensor_mode(tensorName) == self.trt.TensorIOMode.INPUT:
                self.context.set_input_shape(tensorName, self.tensors[i].shape)

        self.I0 = None
        self.firstRun = True

    def initCudaGraph(self):
        """Initialize CUDA graph for optimized execution."""
        self.cudaGraph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.cudaGraph, stream=self.stream):
            self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        self.stream.synchronize()

    def padFrame(self, frame: torch.Tensor) -> torch.Tensor:
        """Pad frame to required dimensions."""
        if self.padding != (0, 0, 0, 0):
            return F.pad(frame, self.padding, mode="replicate")
        return frame

    @torch.inference_mode()
    def processFrame(self, frame: torch.Tensor, name: str):
        """Copy frame data to TensorRT input tensors."""
        with torch.cuda.stream(self.normStream):
            padded = self.padFrame(frame.to(dtype=self.dtype))
            match name:
                case "img0":
                    self.tImg0.copy_(padded, non_blocking=True)
                case "img1":
                    self.tImg1.copy_(padded, non_blocking=True)
                case "img2":
                    self.tImg2.copy_(padded, non_blocking=True)
                case "timestep":
                    self.tTimestep.copy_(frame, non_blocking=True)
        self.normStream.synchronize()

    @torch.inference_mode()
    def cacheFrame(self, frame: torch.Tensor):
        """Update frame cache without producing output."""
        self.I0 = frame.to(dtype=self.dtype, device=self.device, non_blocking=True)
        self.tImg0.copy_(self.tImg1, non_blocking=True)
        torch.cuda.synchronize()

    @torch.inference_mode()
    def cacheFrameReset(self, frame: torch.Tensor):
        """Reset cache on scene change."""
        self.I0 = frame.to(dtype=self.dtype, device=self.device, non_blocking=True)
        self.processFrame(frame, "img0")
        self.firstRun = True

    @torch.inference_mode()
    def __call__(
        self,
        frame: torch.Tensor,
        nextFrame: torch.Tensor,
        interpQueue,
        framesToInsert: int = 2,
        timesteps=None,
    ):
        """
        Interpolate frames using 3-frame DistilDRBA TensorRT model.

        Args:
            frame: Current frame tensor [B, C, H, W] - this becomes I1
            nextFrame: Next frame from peek() [B, C, H, W] - this becomes I2
            interpQueue: Queue to put interpolated frames
            framesToInsert: Number of frames to insert
            timesteps: Optional list of timestep values (in TAS range 0-1)
        """
        if nextFrame is not None and nextFrame.shape[-2:] != frame.shape[-2:]:
            nextFrame = F.interpolate(
                nextFrame, size=frame.shape[-2:], mode="bilinear", align_corners=False
            )

        if self.firstRun:
            self.I0 = frame.to(dtype=self.dtype, device=self.device, non_blocking=True)
            self.processFrame(frame, "img0")
            self.firstRun = False
            return

        if self.I0 is not None and self.I0.shape[-2:] != frame.shape[-2:]:
            self.I0 = F.interpolate(
                self.I0, size=frame.shape[-2:], mode="bilinear", align_corners=False
            )
            self.processFrame(self.I0, "img0")

        self.processFrame(frame, "img1")

        I2 = nextFrame if nextFrame is not None else frame
        self.processFrame(I2, "img2")

        for i in range(framesToInsert):
            if timesteps is not None and i < len(timesteps):
                t = timesteps[i]
            else:
                t = (i + 1) / (framesToInsert + 1)

            tDrba = 0.5 + t * 0.5 - 0.0001

            with torch.cuda.stream(self.stream):
                # Fill the timestep on the SAME stream that replays the graph so
                # the replay is ordered after the write; filling on the default
                # stream races the replay and can read a stale timestep.
                self.tTimestep.fill_(tDrba)
                self.cudaGraph.replay()
            self.stream.synchronize()

            with torch.cuda.stream(self.outputStream):
                output = (
                    self.tOutput[:, :, : self.height, : self.width].clone().detach()
                )
            self.outputStream.synchronize()
            interpQueue.put(output)

        self.I0 = frame.to(dtype=self.dtype, device=self.device, non_blocking=True)
        self.tImg0.copy_(self.tImg1, non_blocking=True)
        torch.cuda.synchronize()
