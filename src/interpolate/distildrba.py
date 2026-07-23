import logging
import os

import numpy as np
import torch
import torch.nn.functional as F

from src.constants import ADOBE
from src.infra.isCudaInit import CudaChecker
from src.infra.logAndPrint import logAndPrint, logWarning
from src.infra.providerCheck import warnIfProviderMissing
from src.model.download import downloadModels
from src.model.registry import modelsMap, weightsDir

if ADOBE:
    from src.server.aeComms import progressState


checker = CudaChecker()

torch.set_float32_matmul_precision("medium")


def _padSizeFor(scale: float) -> int:
    """Frame padding multiple the DistilDRBA pyramid needs.

    The coarsest level runs at 1/(16/scale) of the frame, and every IFBlock's
    conv0 floors the spatial size by 4 before lastconv upsamples it back by 4.
    So the padded frame has to stay divisible by 4 at the coarsest level:
    mod-64 at scale 1.0, mod-128 at scale 0.5. Padding to 64 at scale 0.5 (as
    this file used to, unconditionally) leaves the pyramid off by a level --
    distildrba-lite raises "Sizes of tensors must match" and distildrba emits a
    wrong-width tensor -- for every >1080p input whose mod-64 size is not also
    mod-128, e.g. 2560x1440 (pads to 1472, and 1472 % 128 == 64). 4K only
    escaped because 3840x2176 happens to be mod-128 on both axes.
    """
    return 64 if scale == 1.0 else 128


class DistilDRBACuda:
    # 3-frame arch: I0 is cached internally, I2 must be handed in.
    temporalWindow = (0, 1)

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

        self.padSize = _padSizeFor(self.scale)
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
        Scene-cut reset: anchor I0 = frame and clear the 3-frame feature cache
        so the next interpolation starts fresh in the new scene. firstRun stays
        False: the loop already emitted the held frames, and the next __call__
        must interpolate I0(=this frame) <-> next frame (a firstRun path would
        instead swallow that frame's interpolation).
        """
        self.f0 = None
        self.f1 = None
        self.f2 = None
        self.I0 = self.prepareFrame(frame)
        self.firstRun = False

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
            nextFrame: Next frame from the pipeline's FrameWindow [B, C, H, W] - this becomes I2
            interpQueue: Queue to put interpolated frames
            framesToInsert: Number of frames to insert between consecutive frames
            timesteps: Optional list of timestep values (in TAS range 0-1)

        Frame mapping:
            - I0 = previous frame (cached from last call)
            - I1 = current frame (passed as `frame`)
            - I2 = next frame (supplied by the window this class declares via
              `temporalWindow`; None on the final frame)

        We interpolate between I0 and I1, using I2 as future context.
        """
        # Prepare current frame (I1)
        if self.normStream:
            with torch.cuda.stream(self.normStream):
                I1 = self.prepareFrame(frame)
            self.normStream.synchronize()
        else:
            I1 = self.prepareFrame(frame)

        # Prepare next frame (I2). The pipeline hands us a neighbour from the
        # same domain as `frame`, so a size mismatch is a wiring bug, not
        # something to paper over: bilinearly upsampling I2 to match an upscaled
        # I1 used to feed the model a blurred future frame and silently degrade
        # the flow estimate.
        if nextFrame is not None:
            if nextFrame.shape[-2:] != frame.shape[-2:]:
                raise RuntimeError(
                    f"DistilDRBA got a next frame of {tuple(nextFrame.shape[-2:])} "
                    f"against a current frame of {tuple(frame.shape[-2:])}; the "
                    f"caller must supply the neighbour in the same domain."
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
    # 3-frame arch: I0 is cached internally, I2 must be handed in.
    temporalWindow = (0, 1)

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
        from src.model.trtHandler import (
            tensorRTEngineCreator,
            tensorRTEngineLoader,
        )
        from src.utils.tensorrt_import import trt

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

        self.padSize = _padSizeFor(self.scale)
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
        from src.rifearches.IFNet_distildrba_tensorrt import IFNetFullTRT, IFNetLiteTRT

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
        """Scene-cut reset: anchor I0 = frame (I0 python state + tImg0 binding).
        firstRun stays False so the next __call__ interpolates this frame <->
        the next frame instead of swallowing it (see DistilDRBACuda)."""
        self.I0 = frame.to(dtype=self.dtype, device=self.device, non_blocking=True)
        self.processFrame(frame, "img0")
        self.firstRun = False

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
            nextFrame: Next frame from the pipeline's FrameWindow [B, C, H, W] - this becomes I2
            interpQueue: Queue to put interpolated frames
            framesToInsert: Number of frames to insert
            timesteps: Optional list of timestep values (in TAS range 0-1)
        """
        # See DistilDRBACuda.__call__: the neighbour arrives in the same domain as
        # `frame`, so a mismatch is a caller bug rather than something to resize away.
        if nextFrame is not None and nextFrame.shape[-2:] != frame.shape[-2:]:
            raise RuntimeError(
                f"DistilDRBA got a next frame of {tuple(nextFrame.shape[-2:])} "
                f"against a current frame of {tuple(frame.shape[-2:])}; the "
                f"caller must supply the neighbour in the same domain."
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


class DistilDRBADirectML:
    # 3-frame arch: I0 is cached internally, I2 must be handed in.
    temporalWindow = (0, 1)

    def __init__(
        self,
        half: bool,
        width: int,
        height: int,
        interpolateMethod: str,
        interpolateFactor: int = 2,
    ):
        """
        Initialize DistilDRBA on DirectML (or OpenVINO, which rides the same
        ONNX Runtime session).

        The exported graph is the same shape as the TensorRT one -- img0, img1,
        img2, timestep in, one frame out, no recurrent state -- so the driver
        only has to keep img0 across calls. The arch it exports comes from
        DistilDRBA_directml, whose warps are decomposed into gathers because
        DirectML has no GridSample kernel.

        Args:
            half: Use half precision (fp16)
            width: Frame width
            height: Frame height
            interpolateMethod: "distildrba[-lite]-{directml,openvino}"
            interpolateFactor: Interpolation factor
        """
        import onnxruntime as ort

        if "openvino" in interpolateMethod:
            logAndPrint(
                "OpenVINO backend is an experimental feature, please report any issues you encounter.",
                "yellow",
            )
            import openvino  # noqa: F401

        self.ort = ort

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

        if self.scale == 0.5 and self.half:
            logAndPrint(
                "UHD and fp16 are not compatible with DistilDRBA, defaulting to fp32",
                "yellow",
            )
            self.half = False

        self.deviceType = "cpu"
        self.device = torch.device(self.deviceType)
        self.dtype = torch.float16 if self.half else torch.float32
        self.numpyDType = np.float16 if self.half else np.float32

        self.padSize = _padSizeFor(self.scale)
        self.ph = ((self.height - 1) // self.padSize + 1) * self.padSize
        self.pw = ((self.width - 1) // self.padSize + 1) * self.padSize
        self.padding = (0, self.pw - self.width, 0, self.ph - self.height)

        self.handleModel()

    def handleModel(self):
        baseMethod = self.interpolateMethod.replace("-directml", "").replace(
            "-openvino", ""
        )
        self.filename = modelsMap(baseMethod)
        folderPath = os.path.join(weightsDir, baseMethod)

        if not os.path.exists(os.path.join(folderPath, self.filename)):
            modelPath = downloadModels(model=baseMethod)
        else:
            modelPath = os.path.join(folderPath, self.filename)

        onnxPath = modelPath.replace(
            ".pkl",
            f"_{self.width}x{self.height}_"
            f"{'fp16' if self.half else 'fp32'}_directml_nocache.onnx",
        )

        if not os.path.exists(onnxPath):
            self.exportOnnx(modelPath, onnxPath)

        self.createSession(onnxPath)
        self.setupTensors()

    def exportOnnx(self, modelPath, onnxPath):
        from src.rifearches.DistilDRBA_directml import IFNetFullDML, IFNetLiteDML

        if ADOBE:
            progressState.update(
                {"status": f"Exporting {self.interpolateMethod} to ONNX."}
            )
        logAndPrint("Exporting model to ONNX", "green")

        model = (
            IFNetLiteDML(scale=self.scale)
            if self.lite
            else IFNetFullDML(scale=self.scale)
        )
        stateDict = torch.load(modelPath, map_location="cpu", weights_only=True)
        if "model" in stateDict:
            stateDict = stateDict["model"]
        model.load_state_dict(stateDict, strict=False)
        del stateDict

        model = model.half() if self.half else model.float()
        model = model.eval().to(self.device)

        dummyImg = torch.zeros(
            1, 3, self.ph, self.pw, dtype=self.dtype, device=self.device
        )
        dummyTimestep = torch.full(
            (1, 1, self.ph, self.pw), 0.75, dtype=self.dtype, device=self.device
        )

        logging.info(f"Exporting model to {onnxPath}")

        torch.onnx.export(
            model,
            (dummyImg, dummyImg.clone(), dummyImg.clone(), dummyTimestep),
            onnxPath,
            input_names=["img0", "img1", "img2", "timestep"],
            output_names=["output"],
            dynamic_axes={
                "img0": {2: "height", 3: "width"},
                "img1": {2: "height", 3: "width"},
                "img2": {2: "height", 3: "width"},
                "timestep": {2: "height", 3: "width"},
                "output": {2: "height", 3: "width"},
            },
            opset_version=20,
            optimize=False,
            dynamo=False,
        )

        del model, dummyImg, dummyTimestep

    def createSession(self, onnxPath):
        providers = self.ort.get_available_providers()
        logging.info(f"Available providers: {providers}")

        if "DmlExecutionProvider" in providers or "OpenVINOExecutionProvider" in (
            providers
        ):
            if "directml" in self.interpolateMethod:
                logging.info("DirectML provider available. Defaulting to DirectML")
                self.model = self.ort.InferenceSession(
                    onnxPath, providers=["DmlExecutionProvider"]
                )
                warnIfProviderMissing(
                    self.model, "DmlExecutionProvider", "DirectML interpolate"
                )
            else:
                logging.info("Using OpenVINO model")
                self.model = self.ort.InferenceSession(
                    onnxPath,
                    providers=[
                        ("OpenVINOExecutionProvider", {"device_type": "AUTO:GPU,CPU"})
                    ],
                )
                warnIfProviderMissing(
                    self.model, "OpenVINOExecutionProvider", "OpenVINO interpolate"
                )
        else:
            logWarning(
                "DirectML/OpenVINO provider not available, falling back to CPU, expect significantly worse performance, ensure that your drivers are up to date and your GPU supports DirectX 12"
            )
            self.model = self.ort.InferenceSession(
                onnxPath, providers=["CPUExecutionProvider"]
            )

        self.needsOutputSync = any(
            provider in self.model.get_providers()
            for provider in ("DmlExecutionProvider", "OpenVINOExecutionProvider")
        )

    def setupTensors(self):
        def buffer(channels, fill=0.0):
            return torch.full(
                (1, channels, self.ph, self.pw),
                fill,
                dtype=self.dtype,
                device=self.device,
            ).contiguous()

        self.tImg0 = buffer(3)
        self.tImg1 = buffer(3)
        self.tImg2 = buffer(3)
        self.tTimestep = buffer(1, 0.75)
        # The arch returns the padded frame; the crop happens on the way out.
        self.tOutput = buffer(3)

        self.IoBinding = self.model.io_binding()

        # Only the OUTPUT binding survives across runs: it is a destination
        # pointer ORT writes at run time. Inputs are snapshotted at bind_input()
        # time, so they are (re)bound in __call__ -- see bind().
        self.IoBinding.bind_output(
            name="output",
            device_type=self.deviceType,
            device_id=0,
            element_type=self.numpyDType,
            shape=self.tOutput.shape,
            buffer_ptr=self.tOutput.data_ptr(),
        )

        self.firstRun = True

    def bind(self, name, tensor):
        """Bind one input.

        ORT reads the buffer at bind_input() time, not at run time. Binding once
        at setup and mutating the tensor in place afterwards feeds the model
        whatever the buffer held during setup -- for this driver that is zeros,
        and DistilDRBA on three zero frames returns a zero frame, so the whole
        pipeline emits black. Every input that changes has to be rebound before
        the run that should see it.
        """
        self.IoBinding.bind_input(
            name=name,
            device_type=self.deviceType,
            device_id=0,
            element_type=self.numpyDType,
            shape=tensor.shape,
            buffer_ptr=tensor.data_ptr(),
        )

    @torch.inference_mode()
    def processFrame(self, frame: torch.Tensor, name: str):
        padded = F.pad(
            frame.to(device=self.device, dtype=self.dtype),
            self.padding,
            mode="replicate",
        )
        match name:
            case "img0":
                self.tImg0.copy_(padded, non_blocking=False)
            case "img1":
                self.tImg1.copy_(padded, non_blocking=False)
            case "img2":
                self.tImg2.copy_(padded, non_blocking=False)

    @torch.inference_mode()
    def cacheFrame(self, frame: torch.Tensor):
        """Anchor img0 = frame without producing output.

        Nothing in the pipeline calls this today (main.py drives DistilDRBA
        through __call__ and cacheFrameReset), but DistilDRBACuda exposes it and
        anchors on the frame it is handed, so this does the same rather than
        leaving a subtly different meaning behind for whoever wires it up.
        """
        self.processFrame(frame, "img0")

    @torch.inference_mode()
    def cacheFrameReset(self, frame: torch.Tensor):
        """Scene-cut reset: anchor img0 = frame. firstRun stays False so the next
        __call__ interpolates this frame <-> the next one instead of swallowing
        it (see DistilDRBACuda). The graph carries no recurrent feature state, so
        img0 is the whole cache."""
        self.processFrame(frame, "img0")
        self.firstRun = False

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
        Interpolate using the 3-frame DistilDRBA model.

        Args:
            frame: Current frame [B, C, H, W] - becomes img1
            nextFrame: Window's next frame [B, C, H, W] - becomes img2, None at
                the end of the stream or across a cut
            interpQueue: Queue to put interpolated frames
            framesToInsert: Number of frames to insert
            timesteps: Optional list of timestep values (in TAS range 0-1)
        """
        # See DistilDRBACuda.__call__: the neighbour arrives in the same domain
        # as `frame`, so a mismatch is a caller bug, not something to resize away.
        if nextFrame is not None and nextFrame.shape[-2:] != frame.shape[-2:]:
            raise RuntimeError(
                f"DistilDRBA got a next frame of {tuple(nextFrame.shape[-2:])} "
                f"against a current frame of {tuple(frame.shape[-2:])}; the "
                f"caller must supply the neighbour in the same domain."
            )

        if self.firstRun:
            self.processFrame(frame, "img0")
            self.firstRun = False
            return

        self.processFrame(frame, "img1")
        self.processFrame(nextFrame if nextFrame is not None else frame, "img2")

        self.bind("img0", self.tImg0)
        self.bind("img1", self.tImg1)
        self.bind("img2", self.tImg2)

        for i in range(framesToInsert):
            if timesteps is not None and i < len(timesteps):
                t = timesteps[i]
            else:
                t = (i + 1) / (framesToInsert + 1)

            tDrba = 0.5 + t * 0.5 - 0.0001  # Small epsilon to avoid exactly 1.0
            self.tTimestep.fill_(tDrba)

            # Rebind after every fill_ (see bind): otherwise all inserted frames
            # in a gap reuse the timestep captured before the loop, collapsing
            # factor>2 interpolation into duplicate frames.
            self.bind("timestep", self.tTimestep)

            self.model.run_with_iobinding(self.IoBinding)
            if self.needsOutputSync:
                self.IoBinding.synchronize_outputs()

            interpQueue.put(self.tOutput[:, :, : self.height, : self.width].clone())

        self.tImg0.copy_(self.tImg1, non_blocking=False)
