import os
import torch
import logging

from src.utils.modelOptimizer import ModelOptimizer
from .utils.downloadModels import downloadModels, weightsDir, modelsMap, resolveWeightPath
from .utils.isCudaInit import CudaChecker
from src.utils.logAndPrint import logAndPrint
from src.constants import ADOBE

if ADOBE:
    from src.utils.aeComms import progressState

checker = CudaChecker()

torch.set_float32_matmul_precision("medium")


def calculatePadding(width, height, multiple=4):
    padW = (multiple - (width % multiple)) % multiple
    padH = (multiple - (height % multiple)) % multiple
    return (0, padW, 0, padH)


class UniversalPytorch:
    def __init__(
        self,
        upscaleMethod: str = "shufflecugan",
        upscaleFactor: int = 2,
        half: bool = False,
        width: int = 1920,
        height: int = 1080,
        customModel: str = None,
        compileMode: str = "default",
    ):
        """
        Initialize the upscaler with the desired model

        Args:
            upscaleMethod (str): The method to use for upscaling
            upscaleFactor (int): The factor to upscale by
            half (bool): Whether to use half precision
            width (int): The width of the input frame
            height (int): The height of the input frame
            customModel (str): The path to a custom model file
            compileMode: (str): The compile mode to use for the model
        """
        self.upscaleMethod = upscaleMethod
        self.upscaleFactor = upscaleFactor
        self.half = half
        self.width = width
        self.height = height
        self.customModel = customModel
        self.compileMode: str = compileMode

        self.handleModel()

    def handleModel(self):
        """
        Load the desired model
        """
        if ADOBE:
            progressState.update(
                {"status": f"Loading upscale model: {self.upscaleMethod}..."}
            )

        from src.spandrelCompat import (
            ImageModelDescriptor,
            ModelLoader,
            UnsupportedDtypeError,
        )

        if not self.customModel:
            self.filename = modelsMap(
                self.upscaleMethod, self.upscaleFactor, modelType="pth"
            )
            modelPath = resolveWeightPath(
                self.upscaleMethod,
                self.filename,
                upscaleFactor=self.upscaleFactor,
            )
        else:
            if os.path.isfile(self.customModel):
                modelPath = self.customModel
            else:
                raise FileNotFoundError(
                    f"Custom model file {self.customModel} not found"
                )

        if self.upscaleMethod == "saryn":
            from src.extraArches.RTMoSR import RTMoSR

            self.model = RTMoSR()
            stateDict = torch.load(modelPath, map_location="cpu")
            self.model.load_state_dict(stateDict)
            del stateDict
        elif self.upscaleMethod == "figsr":
            from src.extraArches.figsr import FIGSR

            stateDict = torch.load(modelPath, map_location="cpu", weights_only=False)
            self.model = FIGSR(scale=self.upscaleFactor, dim=32)

            self.model.load_state_dict(stateDict, strict=False)
            del stateDict
        elif self.upscaleMethod == "smosr":
            # SMoSR is a registered spandrel arch; ModelLoader reads the
            # .safetensors, auto-detects dim/depth/rep, and reparameterizes
            # (fuses) the rep branches when .eval() runs below.
            self.model = ModelLoader().load_from_file(modelPath)
            if not isinstance(self.model, ImageModelDescriptor):
                raise TypeError(
                    f"SMoSR model {modelPath} did not resolve to an image model descriptor"
                )
            self.model = self.model.model
        elif self.upscaleMethod == "gauss":
            from src.extraArches.DIS import DIS
            from safetensors.torch import load_file

            self.model = DIS(scale=2, num_features=32, num_blocks=12)
            stateDict = load_file(modelPath)
            self.model.load_state_dict(stateDict)
            del stateDict
        else:
            if self.customModel:
                self.model = ModelLoader().load_from_file(modelPath)
                if not isinstance(self.model, ImageModelDescriptor):
                    raise TypeError(
                        f"Custom upscale model {modelPath} did not resolve to an image model descriptor"
                    )
            else:
                self.model = torch.load(modelPath, map_location="cpu", weights_only=False)

                if isinstance(self.model, dict):
                    self.model = ModelLoader().load_from_state_dict(self.model)

            if self.customModel:
                assert isinstance(self.model, ImageModelDescriptor)

            try:
                # SPANDREL HAXX
                self.model = self.model.model
            except Exception:
                pass

        self.model = self.model.eval()

        if self.half and checker.cudaAvailable:
            try:
                self.model = self.model.half()
            except UnsupportedDtypeError as e:
                logging.error(f"Model does not support half precision: {e}")
                self.model = self.model.float()
                self.half = False
            except Exception as e:
                logging.error(f"Error converting model to half precision: {e}")
                self.model = self.model.float()
                self.half = False

        if checker.cudaAvailable:
            self.model = self.model.cuda()
            torch.cuda.empty_cache()

        self.model = ModelOptimizer(
            self.model,
            torch.float16 if self.half else torch.float32,
            memoryFormat=torch.channels_last,
        ).optimizeModel()

        if self.compileMode != "default":
            try:
                if self.compileMode == "max":
                    self.model.compile(mode="max-autotune")
                elif self.compileMode == "max-graphs":
                    self.model.compile(
                        mode="max-autotune-no-cudagraphs", fullgraph=True
                    )
            except Exception as e:
                logging.error(
                    f"Error compiling model {self.upscaleMethod} with mode {self.compileMode}: {e}"
                )
                logAndPrint(
                    f"Error compiling model {self.upscaleMethod} with mode {self.compileMode}: {e}",
                    "red",
                )

            self.compileMode = "default"

        self.dummyInput = (
            torch.zeros(
                (1, 3, self.height, self.width),
                device=checker.device,
                dtype=torch.float16 if self.half else torch.float32,
            )
            .contiguous()
            .to(memory_format=torch.channels_last)
        )

        self.dummyOutput = (
            torch.zeros(
                (
                    1,
                    3,
                    self.height * self.upscaleFactor,
                    self.width * self.upscaleFactor,
                ),
                device=checker.device,
                dtype=torch.float16 if self.half else torch.float32,
            )
            .contiguous()
            .to(memory_format=torch.channels_last)
        )

        self.stream = torch.cuda.Stream()

        with torch.cuda.stream(self.stream):
            # 3 warmup iters before CUDA-graph capture: cudnn.benchmark is False
            # (no algo autotune), so 3 is sufficient to JIT cudnn kernels +
            # allocate workspace before capture (PyTorch-documented minimum).
            for _ in range(3):
                self.model(self.dummyInput)
                self.stream.synchronize()

        self.normStream = torch.cuda.Stream()
        self.outputStream = torch.cuda.Stream()

        if not self.compileMode != "default":
            self.cudaGraph = torch.cuda.CUDAGraph()
            self.initTorchCudaGraph()

    @torch.inference_mode()
    def initTorchCudaGraph(self):
        with torch.cuda.graph(self.cudaGraph, stream=self.stream):
            self.dummyOutput = self.model(self.dummyInput)
        self.stream.synchronize()

    @torch.inference_mode()
    def processFrame(self, frame):
        with torch.cuda.stream(self.normStream):
            self.dummyInput.copy_(
                frame.to(dtype=self.dummyInput.dtype),
                non_blocking=True,
            )
        self.normStream.synchronize()

    @torch.inference_mode()
    def __call__(self, frame: torch.tensor, nextFrame: None) -> torch.tensor:
        self.processFrame(frame)
        with torch.cuda.stream(self.stream):
            if self.compileMode == "default":
                self.cudaGraph.replay()
            else:
                self.dummyOutput.copy_(
                    self.model(self.dummyInput),
                    non_blocking=True,
                )
            output = self.dummyOutput.clone()
        self.stream.synchronize()

        return output

class UniversalPytorchMPS:
    """
    Apple Silicon (MPS) PyTorch upscaler. Mirrors UniversalPytorch but without
    CUDA streams / CUDA graphs, since MPS does not support them. Shares .pth
    weights with the CUDA path — the "-mps" suffix on upscaleMethod is stripped
    before resolving model filenames.
    """

    def __init__(
        self,
        upscaleMethod: str = "shufflecugan-mps",
        upscaleFactor: int = 2,
        half: bool = False,
        width: int = 1920,
        height: int = 1080,
        customModel: str = None,
        compileMode: str = "default",
    ):
        self.upscaleMethod = upscaleMethod
        self.baseMethod = upscaleMethod.replace("-mps", "")
        self.upscaleFactor = upscaleFactor
        self.half = half
        self.width = width
        self.height = height
        self.customModel = customModel
        self.compileMode = compileMode
        self.device = torch.device("mps")
        self.dtype = torch.float16 if self.half else torch.float32

        self.handleModel()

    def handleModel(self):
        if ADOBE:
            progressState.update(
                {"status": f"Loading MPS upscale model: {self.upscaleMethod}..."}
            )

        from src.spandrelCompat import (
            ImageModelDescriptor,
            ModelLoader,
            UnsupportedDtypeError,
        )

        if not self.customModel:
            self.filename = modelsMap(
                self.baseMethod, self.upscaleFactor, modelType="pth"
            )
            modelPath = resolveWeightPath(
                self.baseMethod,
                self.filename,
                upscaleFactor=self.upscaleFactor,
            )
        else:
            if not os.path.isfile(self.customModel):
                raise FileNotFoundError(
                    f"Custom model file {self.customModel} not found"
                )
            modelPath = self.customModel

        if self.baseMethod == "saryn":
            from src.extraArches.RTMoSR import RTMoSR

            self.model = RTMoSR()
            stateDict = torch.load(modelPath, map_location="cpu")
            self.model.load_state_dict(stateDict)
            del stateDict
        elif self.baseMethod == "figsr":
            from src.extraArches.figsr import FIGSR

            stateDict = torch.load(modelPath, map_location="cpu", weights_only=False)
            self.model = FIGSR(scale=self.upscaleFactor, dim=32)
            self.model.load_state_dict(stateDict, strict=False)
            del stateDict
        elif self.baseMethod == "smosr":
            self.model = ModelLoader().load_from_file(modelPath)
            if not isinstance(self.model, ImageModelDescriptor):
                raise TypeError(
                    f"SMoSR model {modelPath} did not resolve to an image model descriptor"
                )
            self.model = self.model.model
        elif self.baseMethod == "gauss":
            from src.extraArches.DIS import DIS
            from safetensors.torch import load_file

            self.model = DIS(scale=2, num_features=32, num_blocks=12)
            stateDict = load_file(modelPath)
            self.model.load_state_dict(stateDict)
            del stateDict
        else:
            if self.customModel:
                self.model = ModelLoader().load_from_file(modelPath)
                if not isinstance(self.model, ImageModelDescriptor):
                    raise TypeError(
                        f"Custom upscale model {modelPath} did not resolve to an image model descriptor"
                    )
            else:
                self.model = torch.load(
                    modelPath, map_location="cpu", weights_only=False
                )
                if isinstance(self.model, dict):
                    self.model = ModelLoader().load_from_state_dict(self.model)

            try:
                # SPANDREL HAXX
                self.model = self.model.model
            except Exception:
                pass

        self.model = self.model.eval()

        if self.half:
            try:
                self.model = self.model.half()
            except UnsupportedDtypeError as e:
                logging.error(f"Model does not support half precision on MPS: {e}")
                self.model = self.model.float()
                self.half = False
                self.dtype = torch.float32
            except Exception as e:
                logging.error(f"Error converting model to half precision on MPS: {e}")
                self.model = self.model.float()
                self.half = False
                self.dtype = torch.float32

        self.model = self.model.to(self.device)

        if self.compileMode != "default":
            # torch.compile on MPS is unstable as of torch 2.11 — warn and skip.
            logAndPrint(
                f"compileMode '{self.compileMode}' ignored on MPS backend (unsupported).",
                "yellow",
            )
            self.compileMode = "default"

        # Warm-up to force MPS kernel compilation before the first real frame.
        with torch.inference_mode():
            dummy = torch.zeros(
                (1, 3, self.height, self.width),
                device=self.device,
                dtype=self.dtype,
            )
            for _ in range(2):
                _ = self.model(dummy)
            torch.mps.synchronize()

    @torch.inference_mode()
    def __call__(self, frame: torch.Tensor, nextFrame=None) -> torch.Tensor:
        if frame.device != self.device or frame.dtype != self.dtype:
            frame = frame.to(device=self.device, dtype=self.dtype)
        output = self.model(frame)
        torch.mps.synchronize()
        return output

class UniversalTensorRT:
    def __init__(
        self,
        upscaleMethod: str = "shufflecugan-tensorrt",
        upscaleFactor: int = 2,
        half: bool = False,
        width: int = 1920,
        height: int = 1080,
        customModel: str = None,
        forceStatic: bool = False,
    ):
        """
        Initialize the upscaler with the desired model

        Args:
            upscaleMethod (str): The method to use for upscaling
            upscaleFactor (int): The factor to upscale by
            half (bool): Whether to use half precision
            width (int): The width of the input frame
            height (int): The height of the input frame
            customModel (str): The path to a custom model file
        """
        import tensorrt as trt
        from .utils.trtHandler import (
            tensorRTEngineCreator,
            tensorRTEngineLoader,
            tensorRTEngineNameHandler,
        )

        self.trt = trt
        self.tensorRTEngineCreator = tensorRTEngineCreator
        self.tensorRTEngineLoader = tensorRTEngineLoader
        self.tensorRTEngineNameHandler = tensorRTEngineNameHandler

        self.upscaleMethod = upscaleMethod
        self.upscaleFactor = upscaleFactor
        self.half = half
        self.width = width
        self.height = height
        self.customModel = customModel
        self.forceStatic = forceStatic

        self.handleModel()

    def handleModel(self):
        if ADOBE:
            progressState.update(
                {"status": f"Loading TensorRT upscale model: {self.upscaleMethod}..."}
            )

        if self.width > 1920 or self.height > 1080:
            self.forceStatic = True
            logAndPrint(
                message="Forcing static engine due to resolution higher than 1920x1080p",
                colorFunc="yellow",
            )

        if not self.customModel:
            self.filename = modelsMap(
                self.upscaleMethod,
                self.upscaleFactor,
                modelType="onnx",
                half=self.half,
            )
            folderName = self.upscaleMethod.replace("-tensorrt", "-onnx")
            self.modelPath = resolveWeightPath(
                folderName,
                self.filename,
                downloadModel=self.upscaleMethod,
                upscaleFactor=self.upscaleFactor,
                half=self.half,
                modelType="onnx",
            )
        else:
            self.modelPath = self.customModel
            if not os.path.exists(self.customModel):
                raise FileNotFoundError(
                    f"Custom model file {self.customModel} not found"
                )

        self.dtype = torch.float16 if self.half else torch.float32
        enginePath = self.tensorRTEngineNameHandler(
            modelPath=self.modelPath,
            fp16=self.half,
            optInputShape=[1, 3, self.height, self.width],
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
                inputsMin=[1, 3, 8, 8],
                inputsOpt=[1, 3, self.height, self.width],
                inputsMax=[1, 3, 1080, 1920],
                forceStatic=self.forceStatic,
            )

        self.stream = torch.cuda.Stream()
        self.dummyInput = torch.zeros(
            (1, 3, self.height, self.width),
            device=checker.device,
            dtype=torch.float16 if self.half else torch.float32,
        )

        self.dummyOutput = torch.zeros(
            (1, 3, self.height * self.upscaleFactor, self.width * self.upscaleFactor),
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

        with torch.cuda.stream(self.stream):
            for _ in range(5):
                self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
                self.stream.synchronize()

        self.normStream = torch.cuda.Stream()
        self.outputStream = torch.cuda.Stream()

        self.cudaGraph = torch.cuda.CUDAGraph()
        self.initTorchCudaGraph()

    @torch.inference_mode()
    def initTorchCudaGraph(self):
        with torch.cuda.graph(self.cudaGraph, stream=self.stream):
            self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        self.stream.synchronize()

    @torch.inference_mode()
    def processFrame(self, frame):
        with torch.cuda.stream(self.normStream):
            self.dummyInput.copy_(
                frame.to(dtype=self.dtype),
                non_blocking=True,
            )
        self.normStream.synchronize()

    @torch.inference_mode()
    def __call__(self, frame, nextFrame: None) -> torch.tensor:
        self.processFrame(frame)

        with torch.cuda.stream(self.stream):
            self.cudaGraph.replay()
            output = self.dummyOutput.clone()
        self.stream.synchronize()

        return output

# NVIDIA Works Notice (required by NVIDIA Software License Agreement
# AI Product-Specific Terms §1.7.1 for distributed source files using
# the NVIDIA Video Effects SDK / Maxine VSR):
#
#   This software contains source code provided by NVIDIA Corporation.
#
class NvidiaVSR:
    """
    NVIDIA Maxine Video Super Resolution (RTX VSR).

    Quality preset is parsed from `upscaleMethod` suffix, e.g.
    `maxine-ultra` -> ULTRA. Default `maxine` -> HIGH.
    Supported: BICUBIC, LOW, MEDIUM, HIGH, ULTRA, HIGHBITRATE_{LOW..ULTRA}.

    API hard limits (Maxine 1.2.0):
      - Input dtype must be float32 (no fp16/bf16)
      - Input shape must be (3, H, W) (no batch, no channels_last)
      - Scale factor must be 1..4

    """

    _VALID_QUALITIES = {
        "BICUBIC", "LOW", "MEDIUM", "HIGH", "ULTRA",
        "HIGHBITRATE_LOW", "HIGHBITRATE_MEDIUM",
        "HIGHBITRATE_HIGH", "HIGHBITRATE_ULTRA",
    }

    def __init__(
        self,
        upscaleMethod: str = "maxine-high",
        upscaleFactor: int = 2,
        half: bool = False,
        width: int = 1920,
        height: int = 1080,
        customModel: str = None,
        compileMode: str = "default",
    ):
        self.upscaleMethod = upscaleMethod
        self.upscaleFactor = upscaleFactor
        self.half = half
        self.width = width
        self.height = height
        self.customModel = customModel
        self.compileMode = compileMode

        if not checker.cudaAvailable:
            raise RuntimeError("NvidiaVSR requires a CUDA-capable NVIDIA GPU")

        if self.half:
            logAndPrint(
                "NVIDIA VSR API requires float32 input; forcing half=False.",
                "yellow",
            )
            self.half = False

        if self.customModel:
            logAndPrint(
                "NVIDIA VSR does not support custom models (bundled in wheel); "
                "ignoring customModel.",
                "yellow",
            )
            self.customModel = None

        if self.compileMode != "default":
            logAndPrint(
                f"compileMode '{self.compileMode}' ignored on NVIDIA VSR backend.",
                "yellow",
            )
            self.compileMode = "default"

        self.qualityName = self._parseQuality(self.upscaleMethod)

        self.handleModel()

    @classmethod
    def _parseQuality(cls, method: str) -> str:
        suffix = method.lower().replace("maxine", "", 1).lstrip("-")
        name = suffix.upper() if suffix else "HIGH"
        if name not in cls._VALID_QUALITIES:
            raise ValueError(
                f"Unknown Maxine VSR quality '{name}' in upscaleMethod "
                f"'{method}'. Valid: {sorted(cls._VALID_QUALITIES)}"
            )
        return name

    def handleModel(self):
        if ADOBE:
            progressState.update(
                {"status": f"Loading NVIDIA VSR ({self.qualityName})..."}
            )

        import nvvfx
        from nvvfx import VideoSuperRes

        self.nvvfx = nvvfx

        quality = VideoSuperRes.QualityLevel[self.qualityName]
        deviceIdx = (
            checker.device.index
            if checker.device.index is not None
            else 0
        )

        self.model = VideoSuperRes(quality=quality, device=deviceIdx)
        self.model.output_width = self.width * self.upscaleFactor
        self.model.output_height = self.height * self.upscaleFactor

        try:
            self.model.load()
        except Exception as e:
            logging.error(f"NVIDIA VSR load failed: {e}")
            raise

        self.dummyInput = torch.zeros(
            (3, self.height, self.width),
            device=checker.device,
            dtype=torch.float32,
        ).contiguous()

        self.dummyOutput = torch.zeros(
            (
                1,
                3,
                self.height * self.upscaleFactor,
                self.width * self.upscaleFactor,
            ),
            device=checker.device,
            dtype=torch.float32,
        ).contiguous()

        for _ in range(5):
            _out = self.model.run(self.dummyInput, stream_ptr=0)
            _ = torch.from_dlpack(_out.image).clone()
        self.dummyInput.uniform_(0.0, 1.0)
        for _ in range(5):
            _out = self.model.run(self.dummyInput, stream_ptr=0)
            _ = torch.from_dlpack(_out.image).clone()
        self.dummyInput.zero_()
        torch.cuda.synchronize()

    @torch.inference_mode()
    def __call__(
        self, frame: torch.Tensor, nextFrame: torch.Tensor = None
    ) -> torch.Tensor:
        src = frame.squeeze(0) # We are always using 4 dim throughout the process, but Maxine API expects 3 dim (C,H,W), so remove batch dim here.
        self.dummyInput.copy_(src.contiguous(), non_blocking=False)
        outCapsule = self.model.run(self.dummyInput, stream_ptr=0)
        upscaled = torch.from_dlpack(outCapsule.image)  # (3, H', W')
        self.dummyOutput[0].copy_(upscaled, non_blocking=False)
        output = self.dummyOutput.clone()
        torch.cuda.synchronize()
        return output

class UniversalDirectML:
    def __init__(
        self,
        upscaleMethod: str,
        upscaleFactor: int,
        half: bool,
        width: int,
        height: int,
        customModel: str,
    ):
        """
        Initialize the upscaler with the desired model

        Args:
            upscaleMethod (str): The method to use for upscaling
            upscaleFactor (int): The factor to upscale by
            half (bool): Whether to use half precision
            width (int): The width of the input frame
            height (int): The height of the input frame
            customModel (str): The path to a custom model file
        """

        import onnxruntime as ort
        import numpy as np

        if "openvino" in upscaleMethod:
            logAndPrint(
                "OpenVINO backend is an experimental feature, please report any issues you encounter.",
                "yellow",
            )
            import openvino  # noqa: F401

        self.ort = ort
        self.np = np
        self.ort.set_default_logger_severity(3)

        self.upscaleMethod = upscaleMethod
        self.upscaleFactor = upscaleFactor
        self.half = half
        self.width = width
        self.height = height
        self.customModel = customModel

        self.handleModel()

    def handleModel(self):
        """
        Load the desired model
        """
        if ADOBE:
            progressState.update(
                {"status": f"Loading DirectML upscale model: {self.upscaleMethod}..."}
            )

        if not self.customModel:
            method = self.upscaleMethod
            if "openvino" in self.upscaleMethod:
                method = method.replace("openvino", "directml")

            self.filename = modelsMap(method, self.upscaleFactor, modelType="onnx")
            if "-directml" in self.upscaleMethod:
                folderName = self.upscaleMethod.replace("-directml", "-onnx")
            elif "-openvino" in self.upscaleMethod:
                folderName = self.upscaleMethod.replace("-openvino", "-onnx")

            modelPath = resolveWeightPath(
                folderName,
                self.filename,
                downloadModel=method,
                upscaleFactor=self.upscaleFactor,
                modelType="onnx",
                half=self.half,
            )
        else:
            logging.info(
                f"Using custom model: {self.customModel}, this is an experimental feature, expect potential issues"
            )
            if os.path.isfile(self.customModel) and self.customModel.endswith(".onnx"):
                modelPath = self.customModel
            else:
                if not self.customModel.endswith(".onnx"):
                    raise FileNotFoundError(
                        f"Custom model file {self.customModel} is not an ONNX file"
                    )
                else:
                    raise FileNotFoundError(
                        f"Custom model file {self.customModel} not found"
                    )

        providers = self.ort.get_available_providers()

        if (
            "DmlExecutionProvider" in providers
            or "OpenVINOExecutionProvider" in providers
        ):
            if "directml" in self.upscaleMethod:
                logging.info("DirectML provider available. Defaulting to DirectML")
                self.model = self.ort.InferenceSession(
                    modelPath, providers=["DmlExecutionProvider"]
                )
            elif "openvino" in self.upscaleMethod:
                logging.info("Using OpenVINO model")
                self.model = self.ort.InferenceSession(
                    modelPath,
                    providers=[
                        ("OpenVINOExecutionProvider", {"device_type": "AUTO:GPU,CPU"})
                    ],
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
            self.numpyDType = self.np.float16
            self.torchDType = torch.float16
        else:
            self.numpyDType = self.np.float32
            self.torchDType = torch.float32

        self.IoBinding = self.model.io_binding()
        self.dummyInput = torch.zeros(
            (1, 3, self.height, self.width),
            device=self.deviceType,
            dtype=self.torchDType,
        ).contiguous()

        self.dummyOutput = torch.zeros(
            (1, 3, self.height * self.upscaleFactor, self.width * self.upscaleFactor),
            device=self.deviceType,
            dtype=self.torchDType,
        ).contiguous()



        self.usingCpuFallback = False
        self.modelPath = modelPath

    def _fallbackToCpu(self):
        """Reinitialize model with CPU provider after DirectML/OpenVINO failure."""
        logAndPrint(
            "DirectML/OpenVINO encountered an error, falling back to CPU. Performance will be slower.",
            "yellow",
        )

        self.model = self.ort.InferenceSession(
            self.modelPath, providers=["CPUExecutionProvider"]
        )

        self.IoBinding = self.model.io_binding()
        self.IoBinding.bind_output(
            name="output",
            device_type=self.deviceType,
            device_id=0,
            element_type=self.numpyDType,
            shape=self.dummyOutput.shape,
            buffer_ptr=self.dummyOutput.data_ptr(),
        )
        self.IoBinding.bind_input(
            name="input",
            device_type=self.deviceType,
            device_id=0,
            element_type=self.numpyDType,
            shape=self.dummyInput.shape,
            buffer_ptr=self.dummyInput.data_ptr(),
        )

        self.usingCpuFallback = True

    @torch.inference_mode()
    def __call__(self, frame: torch.tensor, nextFrame: None) -> torch.tensor:
        """
        Run the model on the input frame
        """
        try:
            if self.half:
                frame = frame.half()
            else:
                frame = frame.float()

            self.dummyInput.copy_(frame.contiguous(), non_blocking=False)


            self.IoBinding.bind_input(
                name="input",
                device_type=self.deviceType,
                device_id=0,
                element_type=self.numpyDType,
                shape=self.dummyInput.shape,
                buffer_ptr=self.dummyInput.data_ptr(),
            )

            self.IoBinding.bind_output(
                name="output",
                device_type=self.deviceType,
                device_id=0,
                element_type=self.numpyDType,
                shape=self.dummyOutput.shape,
                buffer_ptr=self.dummyOutput.data_ptr(),
            )


            self.model.run_with_iobinding(self.IoBinding)
            frame = self.dummyOutput.contiguous()

            return frame

        except Exception as e:
            if not self.usingCpuFallback:
                logging.warning(
                    f"DirectML/OpenVINO inference failed ({e}); falling back to CPU provider."
                )
                self._fallbackToCpu()
                return self.__call__(frame, nextFrame)
            else:
                logging.exception(
                    f"Something went wrong while processing the frame, {e}"
                )
                raise

class AnimeSRDirectML:
    """
    AnimeSR DirectML/OpenVINO implementation with 5-input architecture.
    This handles the multi-input/multi-output nature of AnimeSR.
    """

    def __init__(
        self,
        upscaleMethod: str,
        half: bool,
        width: int,
        height: int,
    ):
        import onnxruntime as ort
        import numpy as np

        if "openvino" in upscaleMethod:
            logAndPrint(
                "OpenVINO backend is an experimental feature, please report any issues you encounter.",
                "yellow",
            )
            import openvino  # noqa: F401

        self.ort = ort
        self.np = np
        self.ort.set_default_logger_severity(3)

        self.upscaleMethod = upscaleMethod
        self.half = half
        self.width = width
        self.height = height

        self.padding = calculatePadding(width, height, 4)
        self.paddedHeight = self.padding[3] + height + self.padding[2]
        self.paddedWidth = self.padding[1] + width + self.padding[0]

        self.handleModel()

    def handleModel(self):
        method = self.upscaleMethod
        if "openvino" in self.upscaleMethod:
            method = method.replace("openvino", "directml")

        filename = modelsMap(method, modelType="onnx")
        if "-directml" in self.upscaleMethod:
            folderName = self.upscaleMethod.replace("-directml", "-onnx")
        elif "-openvino" in self.upscaleMethod:
            folderName = self.upscaleMethod.replace("-openvino", "-onnx")

        modelPath = resolveWeightPath(
            folderName,
            filename,
            downloadModel=method,
            modelType="onnx",
            half=self.half,
        )

        providers = self.ort.get_available_providers()

        if "DmlExecutionProvider" in providers and "directml" in self.upscaleMethod:
            logging.info("DirectML provider available. Defaulting to DirectML")
            self.model = self.ort.InferenceSession(
                modelPath, providers=["DmlExecutionProvider"]
            )
        elif (
            "OpenVINOExecutionProvider" in providers
            and "openvino" in self.upscaleMethod
        ):
            logging.info("Using OpenVINO model")
            self.model = self.ort.InferenceSession(
                modelPath,
                providers=[
                    ("OpenVINOExecutionProvider", {"device_type": "AUTO:GPU,CPU"})
                ],
            )
        else:
            logging.info(
                "DirectML/OpenVINO provider not available, falling back to CPU"
            )
            self.model = self.ort.InferenceSession(
                modelPath, providers=["CPUExecutionProvider"]
            )

        self.deviceType = "cpu"
        self.device = torch.device(self.deviceType)

        if self.half:
            self.numpyDType = self.np.float16
            self.torchDType = torch.float16
        else:
            self.numpyDType = self.np.float32
            self.torchDType = torch.float32

        self.prevFrame = torch.zeros(
            (1, 3, self.paddedHeight, self.paddedWidth),
            device=self.deviceType,
            dtype=self.torchDType,
        ).contiguous()
        self.currFrame = torch.zeros(
            (1, 3, self.paddedHeight, self.paddedWidth),
            device=self.deviceType,
            dtype=self.torchDType,
        ).contiguous()
        self.nextFrame = torch.zeros(
            (1, 3, self.paddedHeight, self.paddedWidth),
            device=self.deviceType,
            dtype=self.torchDType,
        ).contiguous()
        self.fb = torch.zeros(
            (1, 3, self.paddedHeight * 4, self.paddedWidth * 4),
            device=self.deviceType,
            dtype=self.torchDType,
        ).contiguous()
        self.state = torch.zeros(
            (1, 64, self.height, self.width),
            device=self.deviceType,
            dtype=self.torchDType,
        ).contiguous()

        self.outImg = torch.zeros(
            (1, 3, self.paddedHeight * 4, self.paddedWidth * 4),
            device=self.deviceType,
            dtype=self.torchDType,
        ).contiguous()
        self.outState = torch.zeros(
            (1, 64, self.height, self.width),
            device=self.deviceType,
            dtype=self.torchDType,
        ).contiguous()

        self._ortInputs = {
            "prev_frame": self.prevFrame.numpy(),
            "curr_frame": self.currFrame.numpy(),
            "next_frame": self.nextFrame.numpy(),
            "fb": self.fb.numpy(),
            "state": self.state.numpy(),
        }

        self.firstRun = True
        self.modelPath = modelPath

    def padFrame(self, frame: torch.tensor) -> torch.tensor:
        return torch.nn.functional.pad(frame, self.padding, mode="reflect")

    def __call__(self, frame: torch.tensor, nextFrame: torch.tensor) -> torch.tensor:
        frame = frame.half() if self.half else frame.float()
        paddedFrame = self.padFrame(frame).cpu()
        self.currFrame.copy_(paddedFrame)

        if self.firstRun:
            self.prevFrame.copy_(paddedFrame)
            self.firstRun = False

        if nextFrame is None:
            self.nextFrame.copy_(paddedFrame)
        else:
            nextFrame = nextFrame.half() if self.half else nextFrame.float()
            self.nextFrame.copy_(self.padFrame(nextFrame).cpu())

        outputs = self.model.run(["out_img", "out_state"], self._ortInputs)

        self.outImg = torch.from_numpy(outputs[0])
        self.outState = torch.from_numpy(outputs[1])

        self.state.copy_(self.outState)
        self.fb.copy_(self.outImg)
        self.prevFrame.copy_(paddedFrame)

        return torch.nn.functional.interpolate(
            self.outImg,
            size=(self.height * 2, self.width * 2),
            mode="bicubic",
            align_corners=False,
        )


class UniversalNCNN:
    def __init__(self, upscaleMethod, upscaleFactor):
        self.upscaleMethod = upscaleMethod
        self.upscaleFactor = upscaleFactor

        from upscale_ncnn_py import UPSCALE

        self.filename = modelsMap(
            self.upscaleMethod,
            modelType="ncnn",
        )

        if self.filename.endswith("-ncnn.zip"):
            self.filename = self.filename[:-9]
        elif self.filename.endswith("-ncnn"):
            self.filename = self.filename[:-5]

        modelPath = resolveWeightPath(
            self.upscaleMethod,
            self.filename,
            modelType="ncnn",
        )

        if modelPath.endswith("-ncnn.zip"):
            modelPath = modelPath[:-9]
        elif modelPath.endswith("-ncnn"):
            modelPath = modelPath[:-5]

        lastSlash = os.path.basename(modelPath)
        modelPath = os.path.join(modelPath, lastSlash)

        self.model = UPSCALE(
            gpuid=0,
            tta_mode=False,
            tilesize=0,
            model_str=modelPath,
            num_threads=2,
        )

    def __call__(self, frame, nextFrame: None) -> torch.tensor:
        iniFrameDtype = frame.dtype
        frame = self.model.process_torch(
            frame.mul(255).clamp(0, 255).to(torch.uint8).squeeze(0).permute(1, 2, 0).cpu()
        )

        frame = frame.to(iniFrameDtype).mul(1 / 255).permute(2, 0, 1).unsqueeze(0)
        return frame

class AnimeSR:
    def __init__(
        self,
        upscaleFactor: int = 2,
        half: bool = False,
        width: int = 1920,
        height: int = 1080,
        compileMode: str = "default",
    ):
        self.upscaleFactor = upscaleFactor
        self.half = half
        self.width = width
        self.height = height
        self.compileMode: str = compileMode

        self.handleModel()

    def handleModel(self):
        """
        Load the desired model
        """
        self.filename = modelsMap("animesr", self.upscaleFactor, modelType="pth")
        modelPath = resolveWeightPath(
            "animesr",
            self.filename,
            upscaleFactor=self.upscaleFactor,
        )

        from src.extraArches.AnimeSR import MSRSWVSR

        self.model = MSRSWVSR(num_feat=64, num_block=[5, 3, 2], netscale=4)

        stateDict = torch.load(modelPath, map_location="cpu")
        self.model.load_state_dict(stateDict)
        del stateDict

        self.model = self.model.eval()

        if self.half and checker.cudaAvailable:
            try:
                self.model = self.model.half()
            except Exception as e:
                logging.error(f"Error converting model to half precision: {e}")
                self.model = self.model.float()
                self.half = False

        if checker.cudaAvailable:
            self.model = self.model.cuda()
            torch.cuda.empty_cache()

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
                    f"Error compiling model {self.upscaleMethod} with mode {self.compileMode}: {e}"
                )
                logAndPrint(
                    f"Error compiling model {self.upscaleMethod} with mode {self.compileMode}: {e}",
                    "red",
                )

            self.compileMode = "default"

        # padding related logic
        ph = (4 - self.height % 4) % 4
        pw = (4 - self.width % 4) % 4
        self.padding = (0, pw, 0, ph)

        # The arch requires 3 inputs, so we create dummy inputs for the other two
        self.prevFrame = torch.zeros(
            (
                1,
                3,
                self.padding[3] + self.height + self.padding[2],
                self.padding[1] + self.width + self.padding[0],
            ),
            device=checker.device,
            dtype=torch.float16 if self.half else torch.float32,
        ).to(memory_format=torch.channels_last)
        self.nextFrame = torch.zeros(
            (
                1,
                3,
                self.padding[3] + self.height + self.padding[2],
                self.padding[1] + self.width + self.padding[0],
            ),
            device=checker.device,
            dtype=torch.float16 if self.half else torch.float32,
        ).to(memory_format=torch.channels_last)

        self.dummyOutput = self.prevFrame.new_zeros(
            1, 3, self.height * 4, self.width * 4
        ).to(memory_format=torch.channels_last)

        # The model has some caching functionality that requires a state
        self.state = self.prevFrame.new_zeros(1, 64, self.height, self.width)

        self.stream = torch.cuda.Stream()
        self.normStream = torch.cuda.Stream()
        self.outputStream = torch.cuda.Stream()

        self.firstRun = True

    def padFrame(self, frame: torch.tensor) -> torch.tensor:
        return torch.nn.functional.pad(frame, self.padding, mode="reflect")

    @torch.inference_mode()
    def __call__(self, frame: torch.tensor, nextFrame: torch.tensor) -> torch.tensor:
        if self.firstRun:
            with torch.cuda.stream(self.normStream):
                self.prevFrame.copy_(
                    frame.to(dtype=frame.dtype).to(memory_format=torch.channels_last),
                    non_blocking=False,
                )
                if nextFrame is None:
                    self.nextFrame.copy_(
                        frame.to(dtype=frame.dtype).to(
                            memory_format=torch.channels_last
                        ),
                        non_blocking=False,
                    )
                else:
                    self.nextFrame.copy_(
                        nextFrame.to(dtype=frame.dtype).to(
                            memory_format=torch.channels_last
                        ),
                        non_blocking=False,
                    )
            self.normStream.synchronize()

            self.firstRun = False
        else:
            with torch.cuda.stream(self.normStream):
                if nextFrame is None:
                    self.nextFrame.copy_(
                        frame.to(dtype=frame.dtype).to(
                            memory_format=torch.channels_last
                        ),
                        non_blocking=False,
                    )
                else:
                    self.nextFrame.copy_(
                        nextFrame.to(dtype=frame.dtype).to(
                            memory_format=torch.channels_last
                        ),
                        non_blocking=False,
                    )
            self.normStream.synchronize()

        # preparing that mofo
        with torch.cuda.stream(self.normStream):
            frame = self.padFrame(frame)
        self.normStream.synchronize()

        with torch.cuda.stream(self.outputStream):
            self.dummyOutput, state = self.model(
                self.prevFrame,
                frame,
                self.nextFrame,
                self.dummyOutput,
                self.state,
            )

            self.state = state
        self.outputStream.synchronize()

        with torch.cuda.stream(self.normStream):
            self.prevFrame.copy_(frame, non_blocking=False)
        self.normStream.synchronize()

        # resize the output to self.height*2 and self.width * 2
        with torch.cuda.stream(self.outputStream):
            output = torch.nn.functional.interpolate(
                self.dummyOutput,
                size=(self.height * 2, self.width * 2),
                mode="bicubic",
                align_corners=False,
            )
        self.outputStream.synchronize()

        return output


class AnimeSRTensorRT:
    def __init__(
        self,
        upscaleFactor: int = 2,
        half: bool = False,
        width: int = 1920,
        height: int = 1080,
    ):
        """
        Initialize the upscaler with the desired model

        Args:
            upscaleFactor (int): The factor to upscale by
            half (bool): Whether to use half precision
            width (int): The width of the input frame
            height (int): The height of the input frame
        """
        import tensorrt as trt
        from .utils.trtHandler import (
            tensorRTEngineCreator,
            tensorRTEngineLoader,
            tensorRTEngineNameHandler,
        )

        self.trt = trt
        self.tensorRTEngineCreator = tensorRTEngineCreator
        self.tensorRTEngineLoader = tensorRTEngineLoader
        self.tensorRTEngineNameHandler = tensorRTEngineNameHandler

        self.upscaleFactor = upscaleFactor
        self.half = half
        self.width = width
        self.height = height

        self.handleModel()

    def handleModel(self):
        """
        Load the desired model
        """

        if self.width > 1920 or self.height > 1080:
            self.forceStatic = True
            logAndPrint(
                message="Forcing static engine due to resolution higher than 1920x1080p",
                colorFunc="yellow",
            )

        self.filename = modelsMap(
            "animesr-tensorrt",
            self.upscaleFactor,
            modelType="onnx",
            half=self.half,
        )
        self.modelPath = resolveWeightPath(
            "animesr-onnx",
            self.filename,
            downloadModel="animesr-tensorrt",
            upscaleFactor=self.upscaleFactor,
            half=self.half,
            modelType="onnx",
        )

        self.dtype = torch.float16 if self.half else torch.float32
        enginePath = self.tensorRTEngineNameHandler(
            modelPath=self.modelPath,
            fp16=self.half,
            optInputShape=[1, 3, self.height, self.width],
        )

        ph = (4 - self.height % 4) % 4
        pw = (4 - self.width % 4) % 4
        self.padding = (0, pw, 0, ph)

        # Padded dimensions for x and fb
        self.paddedHeight = self.padding[3] + self.height + self.padding[2]
        self.paddedWidth = self.padding[1] + self.width + self.padding[0]

        self.engine, self.context = self.tensorRTEngineLoader(enginePath)
        if (
            self.engine is None
            or self.context is None
            or not os.path.exists(enginePath)
        ):
            # 3 separate frame inputs instead of concatenated x (9 channels)
            # This allows TensorRT to fuse the concatenation operation
            inputs = [
                [1, 3, self.paddedHeight, self.paddedWidth],  # prev_frame
                [1, 3, self.paddedHeight, self.paddedWidth],  # curr_frame
                [1, 3, self.paddedHeight, self.paddedWidth],  # next_frame
                [1, 3, self.paddedHeight * 4, self.paddedWidth * 4],  # fb
                [1, 64, self.height, self.width],  # state
            ]

            inputsMin = inputsOpt = inputsMax = inputs
            inputNames = ["prev_frame", "curr_frame", "next_frame", "fb", "state"]

            self.engine, self.context = self.tensorRTEngineCreator(
                modelPath=self.modelPath,
                enginePath=enginePath,
                fp16=self.half,
                inputsMin=inputsMin,
                inputsOpt=inputsOpt,
                inputsMax=inputsMax,
                inputName=inputNames,
                isMultiInput=True,
            )

        self.prevFrame = torch.zeros(
            (1, 3, self.paddedHeight, self.paddedWidth),
            device=checker.device,
            dtype=torch.float16 if self.half else torch.float32,
        ).contiguous()
        self.currFrame = torch.zeros(
            (1, 3, self.paddedHeight, self.paddedWidth),
            device=checker.device,
            dtype=torch.float16 if self.half else torch.float32,
        ).contiguous()
        self.nextFrame = torch.zeros(
            (1, 3, self.paddedHeight, self.paddedWidth),
            device=checker.device,
            dtype=torch.float16 if self.half else torch.float32,
        ).contiguous()

        self.dummyOutput = torch.zeros(
            (1, 3, self.paddedHeight * 4, self.paddedWidth * 4),
            device=checker.device,
            dtype=torch.float16 if self.half else torch.float32,
        ).contiguous()

        self.state = torch.zeros(
            (1, 64, self.height, self.width),
            device=checker.device,
            dtype=torch.float16 if self.half else torch.float32,
        )
        self.stateOutput = torch.zeros(
            (1, 64, self.height, self.width),
            device=checker.device,
            dtype=torch.float16 if self.half else torch.float32,
        )

        self.bindings = {
            "prev_frame": self.prevFrame.data_ptr(),
            "curr_frame": self.currFrame.data_ptr(),
            "next_frame": self.nextFrame.data_ptr(),
            "fb": self.dummyOutput.data_ptr(),
            "state": self.state.data_ptr(),
            "out_img": self.dummyOutput.data_ptr(),
            "out_state": self.stateOutput.data_ptr(),
        }

        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)

            if tensor_name == "prev_frame":
                self.context.set_tensor_address(
                    tensor_name, self.bindings["prev_frame"]
                )
                self.context.set_input_shape(tensor_name, self.prevFrame.shape)
            elif tensor_name == "curr_frame":
                self.context.set_tensor_address(
                    tensor_name, self.bindings["curr_frame"]
                )
                self.context.set_input_shape(tensor_name, self.currFrame.shape)
            elif tensor_name == "next_frame":
                self.context.set_tensor_address(
                    tensor_name, self.bindings["next_frame"]
                )
                self.context.set_input_shape(tensor_name, self.nextFrame.shape)
            elif tensor_name == "fb":
                self.context.set_tensor_address(tensor_name, self.bindings["fb"])
                self.context.set_input_shape(tensor_name, self.dummyOutput.shape)
            elif tensor_name == "state":
                self.context.set_tensor_address(tensor_name, self.bindings["state"])
                self.context.set_input_shape(tensor_name, self.state.shape)
            elif tensor_name == "out_img":
                self.context.set_tensor_address(tensor_name, self.bindings["out_img"])
            elif tensor_name == "out_state":
                self.context.set_tensor_address(tensor_name, self.bindings["out_state"])

        self.stream = torch.cuda.Stream()
        self.normStream = torch.cuda.Stream()
        self.outputStream = torch.cuda.Stream()

        self.firstRun = True

    def padFrame(self, frame: torch.tensor) -> torch.tensor:
        return torch.nn.functional.pad(frame, self.padding, mode="reflect")

    @torch.inference_mode()
    def __call__(self, frame: torch.tensor, nextFrame: torch.tensor) -> torch.tensor:
        with torch.cuda.stream(self.normStream):
            paddedFrame = self.padFrame(frame)
            # Copy current frame to currFrame buffer
            self.currFrame.copy_(
                paddedFrame.to(dtype=self.dtype),
                non_blocking=False,
            )

            if self.firstRun:
                # On first run, prevFrame = currFrame
                self.prevFrame.copy_(
                    paddedFrame.to(dtype=self.dtype),
                    non_blocking=False,
                )
                self.firstRun = False

            if nextFrame is None:
                self.nextFrame.copy_(
                    paddedFrame.to(dtype=self.dtype),
                    non_blocking=False,
                )
            else:
                paddedNextFrame = self.padFrame(nextFrame)
                self.nextFrame.copy_(
                    paddedNextFrame.to(dtype=self.dtype),
                    non_blocking=False,
                )
        self.normStream.synchronize()

        # TensorRT execution - no torch.cat needed, TRT receives separate buffers
        with torch.cuda.stream(self.outputStream):
            self.context.execute_async_v3(stream_handle=self.outputStream.cuda_stream)
        self.outputStream.synchronize()

        with torch.cuda.stream(self.normStream):
            self.state.copy_(self.stateOutput, non_blocking=False)
            self.prevFrame.copy_(paddedFrame, non_blocking=False)
        self.normStream.synchronize()

        with torch.cuda.stream(self.outputStream):
            output = torch.nn.functional.interpolate(
                self.dummyOutput,
                size=(self.height * 2, self.width * 2),
                mode="bicubic",
                align_corners=False,
            )
        self.outputStream.synchronize()

        return output


class _ArtCNNLumaMixin:
    """
    Shared BT.709 luma split / recombine for ArtCNN.

    ArtCNN upscales the luma (Y) channel only — its ONNX models take a single
    Y plane (1, 1, H, W) and emit a 2x Y plane (1, 1, 2H, 2W). To keep the
    surrounding RGB pipeline untouched, we split the incoming RGB frame into
    Y'CbCr (BT.709, full-range, gamma-encoded R'G'B' to match decoded video
    frames), run the model on Y, bicubic-upscale the two chroma planes, then
    recombine to RGB.

    Chroma planes stay centered at 0 (range ~[-0.5, 0.5]) throughout — there is
    no need for a +0.5 offset since they are never quantized to 8-bit here.
    """

    _KR = 0.2126
    _KB = 0.0722
    _KG = 1.0 - _KR - _KB  # 0.7152

    @classmethod
    def _rgbToYCbCr(cls, frame: torch.Tensor):
        r = frame[:, 0:1]
        g = frame[:, 1:2]
        b = frame[:, 2:3]
        y = cls._KR * r + cls._KG * g + cls._KB * b
        cb = (b - y) / (2.0 * (1.0 - cls._KB))
        cr = (r - y) / (2.0 * (1.0 - cls._KR))
        return y, cb, cr

    @classmethod
    def _yCbCrToRgb(
        cls, y: torch.Tensor, cb: torch.Tensor, cr: torch.Tensor
    ) -> torch.Tensor:
        r = y + (2.0 * (1.0 - cls._KR)) * cr
        b = y + (2.0 * (1.0 - cls._KB)) * cb
        g = (y - cls._KR * r - cls._KB * b) / cls._KG
        return torch.cat((r, g, b), dim=1).clamp_(0.0, 1.0)

    @staticmethod
    def _upscaleChroma(plane: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.interpolate(
            plane,
            scale_factor=2,
            mode="bicubic",
            align_corners=False,
        )


class ArtCNNTensorRT(_ArtCNNLumaMixin):
    """
    ArtCNN luma-only 2x upscaler (TensorRT).

    Wraps a single-channel ArtCNN engine with BT.709 Y'CbCr split/recombine
    (see _ArtCNNLumaMixin). The engine is fed only the Y plane; chroma is
    bicubic-upscaled on the GPU and merged back.

    The fp16/fp32 ONNX is selected by `modelsMap` from `half`; the shared engine
    builder is strongly-typed, so engine precision follows the ONNX graph.
    """

    def __init__(
        self,
        upscaleMethod: str = "artcnn_c4f16-tensorrt",
        upscaleFactor: int = 2,
        half: bool = False,
        width: int = 1920,
        height: int = 1080,
        customModel: str = None,
        forceStatic: bool = False,
    ):
        import tensorrt as trt
        from .utils.trtHandler import (
            tensorRTEngineCreator,
            tensorRTEngineLoader,
            tensorRTEngineNameHandler,
        )

        self.trt = trt
        self.tensorRTEngineCreator = tensorRTEngineCreator
        self.tensorRTEngineLoader = tensorRTEngineLoader
        self.tensorRTEngineNameHandler = tensorRTEngineNameHandler

        self.upscaleMethod = upscaleMethod
        self.upscaleFactor = 2  # ArtCNN models are fixed 2x luma doublers.
        self.half = half
        self.width = width
        self.height = height
        self.customModel = customModel
        self.forceStatic = forceStatic

        self.handleModel()

    def handleModel(self):
        if ADOBE:
            progressState.update(
                {"status": f"Loading TensorRT upscale model: {self.upscaleMethod}..."}
            )

        if self.width > 1920 or self.height > 1080:
            self.forceStatic = True
            logAndPrint(
                message="Forcing static engine due to resolution higher than 1920x1080p",
                colorFunc="yellow",
            )

        if not self.customModel:
            self.filename = modelsMap(
                self.upscaleMethod,
                self.upscaleFactor,
                modelType="onnx",
                half=self.half,
            )
            folderName = self.upscaleMethod.replace("-tensorrt", "-onnx")
            self.modelPath = resolveWeightPath(
                folderName,
                self.filename,
                downloadModel=self.upscaleMethod,
                upscaleFactor=self.upscaleFactor,
                half=self.half,
                modelType="onnx",
            )
        else:
            self.modelPath = self.customModel
            if not os.path.exists(self.customModel):
                raise FileNotFoundError(
                    f"Custom model file {self.customModel} not found"
                )

        self.dtype = torch.float16 if self.half else torch.float32
        enginePath = self.tensorRTEngineNameHandler(
            modelPath=self.modelPath,
            fp16=self.half,
            optInputShape=[1, 1, self.height, self.width],
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
                inputsMin=[1, 1, 8, 8],
                inputsOpt=[1, 1, self.height, self.width],
                inputsMax=[1, 1, 1080, 1920],
                forceStatic=self.forceStatic,
            )

        self.stream = torch.cuda.Stream()
        # Single-channel luma buffers fed to / read from the engine.
        self.dummyInput = torch.zeros(
            (1, 1, self.height, self.width),
            device=checker.device,
            dtype=self.dtype,
        )
        self.dummyOutput = torch.zeros(
            (1, 1, self.height * self.upscaleFactor, self.width * self.upscaleFactor),
            device=checker.device,
            dtype=self.dtype,
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

        self.cudaGraph = torch.cuda.CUDAGraph()
        self.initTorchCudaGraph()

    @torch.inference_mode()
    def initTorchCudaGraph(self):
        with torch.cuda.graph(self.cudaGraph, stream=self.stream):
            self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        self.stream.synchronize()

    @torch.inference_mode()
    def __call__(self, frame: torch.tensor, nextFrame: None) -> torch.tensor:
        frame = frame.to(dtype=self.dtype)
        y, cb, cr = self._rgbToYCbCr(frame)

        with torch.cuda.stream(self.normStream):
            self.dummyInput.copy_(y, non_blocking=True)
        self.normStream.synchronize()

        with torch.cuda.stream(self.stream):
            self.cudaGraph.replay()
            yUp = self.dummyOutput.clone()
        self.stream.synchronize()

        with torch.cuda.stream(self.outputStream):
            cbUp = self._upscaleChroma(cb)
            crUp = self._upscaleChroma(cr)
            output = self._yCbCrToRgb(yUp, cbUp, crUp)
        self.outputStream.synchronize()

        return output

class ArtCNNDirectML(_ArtCNNLumaMixin):
    """
    ArtCNN luma-only 2x upscaler (DirectML / OpenVINO via onnxruntime).

    DirectML / OpenVINO counterpart of ArtCNNTensorRT. Runs the single-channel
    ArtCNN ONNX through onnxruntime and wraps it with BT.709 Y'CbCr
    split/recombine. The fp16/fp32 ONNX is selected by `modelsMap` from `half`.
    """

    def __init__(
        self,
        upscaleMethod: str,
        upscaleFactor: int,
        half: bool,
        width: int,
        height: int,
        customModel: str,
    ):
        import onnxruntime as ort
        import numpy as np

        if "openvino" in upscaleMethod:
            logAndPrint(
                "OpenVINO backend is an experimental feature, please report any issues you encounter.",
                "yellow",
            )
            import openvino  # noqa: F401

        self.ort = ort
        self.np = np
        self.ort.set_default_logger_severity(3)

        self.upscaleMethod = upscaleMethod
        self.upscaleFactor = 2  # ArtCNN models are fixed 2x luma doublers.
        self.half = half
        self.width = width
        self.height = height
        self.customModel = customModel

        self.handleModel()

    def handleModel(self):
        if ADOBE:
            progressState.update(
                {"status": f"Loading DirectML upscale model: {self.upscaleMethod}..."}
            )

        if not self.customModel:
            # OpenVINO reuses the -directml ONNX (mirrors UniversalDirectML).
            method = self.upscaleMethod
            if "openvino" in self.upscaleMethod:
                method = method.replace("openvino", "directml")

            self.filename = modelsMap(
                method, self.upscaleFactor, modelType="onnx", half=self.half
            )
            folderName = self.upscaleMethod.replace("-directml", "-onnx").replace(
                "-openvino", "-onnx"
            )
            modelPath = resolveWeightPath(
                folderName,
                self.filename,
                downloadModel=method,
                upscaleFactor=self.upscaleFactor,
                modelType="onnx",
                half=self.half,
            )
        else:
            if os.path.isfile(self.customModel) and self.customModel.endswith(".onnx"):
                modelPath = self.customModel
            else:
                raise FileNotFoundError(
                    f"Custom model file {self.customModel} is not a valid ONNX file"
                )

        providers = self.ort.get_available_providers()
        if "DmlExecutionProvider" in providers and "directml" in self.upscaleMethod:
            logging.info("DirectML provider available. Defaulting to DirectML")
            self.model = self.ort.InferenceSession(
                modelPath, providers=["DmlExecutionProvider"]
            )
        elif (
            "OpenVINOExecutionProvider" in providers
            and "openvino" in self.upscaleMethod
        ):
            logging.info("Using OpenVINO model")
            self.model = self.ort.InferenceSession(
                modelPath,
                providers=[
                    ("OpenVINOExecutionProvider", {"device_type": "AUTO:GPU,CPU"})
                ],
            )
        else:
            logging.info(
                "DirectML/OpenVINO provider not available, falling back to CPU, expect significantly worse performance"
            )
            self.model = self.ort.InferenceSession(
                modelPath, providers=["CPUExecutionProvider"]
            )

        self.deviceType = "cpu"
        self.device = torch.device(self.deviceType)
        self.numpyDType = self.np.float16 if self.half else self.np.float32
        self.torchDType = torch.float16 if self.half else torch.float32

        self.IoBinding = self.model.io_binding()
        self.dummyInput = torch.zeros(
            (1, 1, self.height, self.width),
            device=self.deviceType,
            dtype=self.torchDType,
        ).contiguous()
        self.dummyOutput = torch.zeros(
            (1, 1, self.height * self.upscaleFactor, self.width * self.upscaleFactor),
            device=self.deviceType,
            dtype=self.torchDType,
        ).contiguous()

    @torch.inference_mode()
    def __call__(self, frame: torch.tensor, nextFrame: None) -> torch.tensor:
        # Chroma split / bicubic / recombine run in fp32 (CPU half bicubic is
        # unsupported); only the model I/O buffers carry the requested dtype.
        srcDevice = frame.device
        fr = frame.float()
        y, cb, cr = self._rgbToYCbCr(fr)

        self.dummyInput.copy_(
            y.to(device=self.deviceType, dtype=self.torchDType).contiguous(),
            non_blocking=False,
        )

        self.IoBinding.bind_input(
            name="input",
            device_type=self.deviceType,
            device_id=0,
            element_type=self.numpyDType,
            shape=self.dummyInput.shape,
            buffer_ptr=self.dummyInput.data_ptr(),
        )
        self.IoBinding.bind_output(
            name="output",
            device_type=self.deviceType,
            device_id=0,
            element_type=self.numpyDType,
            shape=self.dummyOutput.shape,
            buffer_ptr=self.dummyOutput.data_ptr(),
        )

        self.model.run_with_iobinding(self.IoBinding)

        yUp = self.dummyOutput.to(device=srcDevice, dtype=torch.float32)
        cbUp = self._upscaleChroma(cb)
        crUp = self._upscaleChroma(cr)
        return self._yCbCrToRgb(yUp, cbUp, crUp).to(dtype=self.torchDType)
