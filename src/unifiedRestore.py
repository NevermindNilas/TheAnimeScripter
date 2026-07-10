import logging
import os

import torch

from src.constants import ADOBE

from .infra.isCudaInit import CudaChecker
from .infra.logAndPrint import logAndPrint, logWarning
from .infra.providerCheck import warnIfProviderMissing
from .model.download import resolveWeightPath
from .model.registry import modelsMap

if ADOBE:
    from src.server.aeComms import progressState

checker = CudaChecker()


class UnifiedRestoreCuda:
    def __init__(
        self,
        model: str = "scunet",
        half: bool = True,
    ):
        """
        Initialize the denoiser with the desired model

        Args:
            model (str): The model to use for denoising
            width (int): The width of the input frame
            height (int): The height of the input frame
            half (bool): Whether to use half precision
            customModel (str): The path to a custom model file
        """

        self.model = model
        self.half = half
        self.CHANNELSLAST = True
        self.handleModel()

    def handleModel(self):
        """
        Load the Model
        """
        if ADOBE:
            progressState.update({"status": f"Loading restore model: {self.model}..."})

        from src.spandrelCompat import ModelLoader

        if self.model in ["deepdeband-f"]:
            self.half = False
            print(
                f"{self.model} does not support half precision, using float32 instead"
            )

        self.filename = modelsMap(self.model)
        modelPath = resolveWeightPath(self.model, self.filename)

        customLoaders = ("gater3", "deepdeband-f")
        if self.model not in customLoaders:
            try:
                self.model = ModelLoader().load_from_file(path=modelPath)
                if isinstance(self.model, dict):
                    self.model = ModelLoader().load_from_state_dict(self.model)
            except Exception as e:
                logging.error(f"Error loading model: {e}")
        else:
            if self.model == "gater3":
                from safetensors.torch import load_file

                from src.extraArches.gaterv3 import GateRV3

                self.CHANNELSLAST = False

                self.model = GateRV3()

                stateDict = load_file(modelPath)
                self.model.load_state_dict(stateDict)
                del stateDict
            elif self.model == "deepdeband-f":
                from src.extraArches.deepdeband import DeepDebandF

                self.CHANNELSLAST = False

                wrapper = DeepDebandF()
                stateDict = torch.load(modelPath, map_location="cpu")
                wrapper.net.load_state_dict(stateDict)
                self.model = wrapper
                del stateDict

        try:
            # Weird spandrel hack to bypass ModelDecriptor
            self.model = self.model.model
        except Exception:
            pass

        self.model = self.model.eval()

        if self.half:
            self.model = self.model.half()
            self.dType = torch.float16
        else:
            self.model = self.model.float()  # Sanity check, should not be needed
            self.dType = torch.float32

        if checker.cudaAvailable:
            self.model = self.model.cuda()
            torch.cuda.empty_cache()

        self.stream = torch.cuda.Stream()

        if self.CHANNELSLAST:
            self.model.to(memory_format=torch.channels_last)
        else:
            self.model.to(memory_format=torch.contiguous_format)

    @torch.inference_mode()
    def __call__(self, frame: torch.tensor) -> torch.tensor:
        with torch.cuda.stream(self.stream):
            if self.CHANNELSLAST:
                frame = frame.to(
                    checker.device,
                    non_blocking=True,
                    dtype=self.dType,
                    memory_format=torch.channels_last,
                )
            else:
                frame = frame.to(checker.device, non_blocking=True, dtype=self.dType)
            frame = self.model(frame)
        self.stream.synchronize()
        return frame


class AutoCAS:
    """
    AutoCAS — AMD FidelityFX Contrast Adaptive Sharpening (PyTorch port) with
    automatic strength detection. Deterministic signal-processing filter: no
    weights, nothing to download. Replaces the old FFmpeg `cas` --sharpen path
    with a frame-tensor restore stage that auto-tunes its sharpening amount per
    frame from the image's own blur statistics.

    Runs on the input frame's own device (CUDA / CPU / MPS). The blur/contrast
    statistics are computed in fp32 internally, so fp16 input is safe.
    """

    def __init__(self, half: bool = True):
        if ADOBE:
            progressState.update({"status": "Loading restore model: autocas..."})

        from src.extraArches.cas import contrast_adaptive_sharpening

        self._cas = contrast_adaptive_sharpening
        self.half = half
        self.dType = torch.float16 if half else torch.float32
        self.stream = torch.cuda.Stream() if checker.cudaAvailable else None

    @torch.inference_mode()
    def __call__(self, frame: torch.Tensor) -> torch.Tensor:
        # amount=None -> per-frame auto-tune from the frame's own blur signal.
        # Cast + filter run inside the restore stream (mirrors UnifiedRestoreCuda)
        # so the dtype cast is visible to the filter without a cross-stream race.
        if frame.is_cuda and self.stream is not None:
            with torch.cuda.stream(self.stream):
                frame = self._cas(frame.to(dtype=self.dType), amount=None)
            self.stream.synchronize()
            return frame
        return self._cas(frame.to(dtype=self.dType), amount=None)


# NVIDIA Works Notice (required by NVIDIA Software License Agreement
# AI Product-Specific Terms §1.7.1 for distributed source files using
# the NVIDIA Video Effects SDK / Maxine):
#
#   This software contains source code provided by NVIDIA Corporation.
#
class MaxineRestore:
    """
    NVIDIA Maxine Video Effects same-size enhancement (DENOISE / DEBLUR).

    Quality parsed from method name:
      maxine-denoise_{low,medium,high,ultra}  -> DENOISE_*
      maxine-deblur_{low,medium,high,ultra}   -> DEBLUR_*

    API hard limits (Maxine 1.2.0):
      - Input dtype must be float32 (no fp16/bf16)
      - Input shape must be (3, H, W) (no batch, no channels_last)

    """

    _VALID_QUALITIES = {
        "DENOISE_LOW",
        "DENOISE_MEDIUM",
        "DENOISE_HIGH",
        "DENOISE_ULTRA",
        "DEBLUR_LOW",
        "DEBLUR_MEDIUM",
        "DEBLUR_HIGH",
        "DEBLUR_ULTRA",
    }

    def __init__(
        self,
        method: str = "maxine-denoise_high",
        half: bool = False,
        width: int = 1920,
        height: int = 1080,
    ):
        self.method = method
        self.half = half
        self.width = width
        self.height = height

        if not checker.cudaAvailable:
            raise RuntimeError("MaxineRestore requires a CUDA-capable NVIDIA GPU")

        if self.half:
            logAndPrint(
                "NVIDIA Maxine API requires float32 input; forcing half=False.",
                "yellow",
            )
            self.half = False

        self.qualityName = self._parseQuality(self.method)

        self.handleModel()

    @classmethod
    def _parseQuality(cls, method: str) -> str:
        suffix = method.lower().replace("maxine", "", 1).lstrip("-")
        name = suffix.upper()
        if name not in cls._VALID_QUALITIES:
            raise ValueError(
                f"Unknown Maxine restore quality '{name}' in method "
                f"'{method}'. Valid: {sorted(cls._VALID_QUALITIES)}"
            )
        return name

    def handleModel(self):
        if ADOBE:
            progressState.update(
                {"status": f"Loading Maxine restore ({self.qualityName})..."}
            )

        import nvvfx
        from nvvfx import VideoSuperRes

        self.nvvfx = nvvfx

        quality = VideoSuperRes.QualityLevel[self.qualityName]
        deviceIdx = checker.device.index if checker.device.index is not None else 0

        self.sdkModel = VideoSuperRes(quality=quality, device=deviceIdx)
        self.sdkModel.output_width = self.width
        self.sdkModel.output_height = self.height

        try:
            self.sdkModel.load()
        except Exception as e:
            logging.error(f"Maxine restore load failed: {e}")
            raise

        self.dummyInput = torch.zeros(
            (3, self.height, self.width),
            device=checker.device,
            dtype=torch.float32,
        ).contiguous()

        self.dummyOutput = torch.zeros(
            (1, 3, self.height, self.width),
            device=checker.device,
            dtype=torch.float32,
        ).contiguous()

        # No custom streams — default stream throughout. See NvidiaVSR for
        # rationale: multi-stream gave no measurable gain and added
        # cross-stream visibility risk against upstream/downstream stages.

        for _ in range(5):
            _out = self.sdkModel.run(self.dummyInput, stream_ptr=0)
            _ = torch.from_dlpack(_out.image).clone()
        self.dummyInput.uniform_(0.0, 1.0)
        for _ in range(5):
            _out = self.sdkModel.run(self.dummyInput, stream_ptr=0)
            _ = torch.from_dlpack(_out.image).clone()
        self.dummyInput.zero_()
        torch.cuda.synchronize()

    @torch.inference_mode()
    def __call__(self, frame: torch.Tensor) -> torch.Tensor:
        src = frame.squeeze(
            0
        )  # We are always using 4 dim throughout the process, but Maxine API expects 3 dim (C,H,W), so remove batch dim here.
        self.dummyInput.copy_(src.contiguous(), non_blocking=False)
        outCapsule = self.sdkModel.run(self.dummyInput, stream_ptr=0)
        restored = torch.from_dlpack(outCapsule.image)  # (3, H, W)
        self.dummyOutput[0].copy_(restored, non_blocking=False)
        output = self.dummyOutput.clone()
        torch.cuda.synchronize()
        return output


class UnifiedRestoreMPS:
    """
    Apple Silicon (MPS) restore. Mirrors UnifiedRestoreCuda but drops
    torch.cuda.Stream. Shares weights with the CUDA path: the "-mps" suffix
    on the method name is stripped before resolving model files.
    """

    def __init__(
        self,
        model: str = "scunet-mps",
        half: bool = True,
    ):
        self.model = model.replace("-mps", "")
        self.half = half
        self.CHANNELSLAST = True
        self.device = torch.device("mps")
        self.handleModel()

    def handleModel(self):
        if ADOBE:
            progressState.update(
                {"status": f"Loading MPS restore model: {self.model}..."}
            )

        from src.spandrelCompat import ModelLoader

        self.filename = modelsMap(self.model)
        modelPath = resolveWeightPath(self.model, self.filename)

        if self.model not in ["gater3"]:
            try:
                self.model = ModelLoader().load_from_file(path=modelPath)
                if isinstance(self.model, dict):
                    self.model = ModelLoader().load_from_state_dict(self.model)
            except Exception as e:
                logging.error(f"Error loading model: {e}")
        else:
            from safetensors.torch import load_file

            if self.model == "gater3":
                from src.extraArches.gaterv3 import GateRV3

                self.CHANNELSLAST = False
                self.model = GateRV3()
                stateDict = load_file(modelPath)
                self.model.load_state_dict(stateDict)
                del stateDict

        try:
            # Weird spandrel hack to bypass ModelDescriptor
            self.model = self.model.model
        except Exception:
            pass

        self.model = self.model.eval()

        if self.half:
            self.model = self.model.half()
            self.dType = torch.float16
        else:
            self.model = self.model.float()
            self.dType = torch.float32

        self.model = self.model.to(self.device)

        if self.CHANNELSLAST:
            self.model.to(memory_format=torch.channels_last)
        else:
            self.model.to(memory_format=torch.contiguous_format)

    @torch.inference_mode()
    def __call__(self, frame: torch.tensor) -> torch.tensor:
        moved = frame.to(self.device, dtype=self.dType)
        if self.CHANNELSLAST:
            moved = moved.to(memory_format=torch.channels_last)
        output = self.model(moved)
        torch.mps.synchronize()
        return output


class UnifiedRestoreTensorRT:
    def __init__(
        self,
        restoreMethod: str = "anime1080fixer-tensorrt",
        half: bool = False,
        width: int = 1920,
        height: int = 1080,
        forceStatic: bool = False,
    ):
        """
        Initialize the upscaler with the desired model

        Args:
            restoreMethod (str): The method to use for upscaling
            upscaleFactor (int): The factor to upscale by
            half (bool): Whether to use half precision
            width (int): The width of the input frame
            height (int): The height of the input frame
            customModel (str): The path to a custom model file
        """

        # Attempt to lazy load for faster startup

        from src.utils.tensorrt_import import trt

        from .model.trtHandler import (
            tensorRTEngineCreator,
            tensorRTEngineLoader,
            tensorRTEngineNameHandler,
        )

        self.trt = trt
        self.tensorRTEngineCreator = tensorRTEngineCreator
        self.tensorRTEngineLoader = tensorRTEngineLoader
        self.tensorRTEngineNameHandler = tensorRTEngineNameHandler

        self.restoreMethod = restoreMethod
        self.half = half
        self.width = width
        self.height = height
        self.forceStatic = forceStatic

        self.handleModel()

    def handleModel(self):
        if ADOBE:
            progressState.update(
                {"status": f"Loading TensorRT restore model: {self.restoreMethod}..."}
            )

        self.originalHeight = self.height
        self.originalWidth = self.width

        if self.restoreMethod in ["scunet-tensorrt"]:
            if self.forceStatic is not True:
                self.forceStatic = True
                logAndPrint(
                    "Forcing static engine due to SCUNET limitations.",
                    "yellow",
                )
            # padding to 64x64
            self.height = (self.height + 63) // 64 * 64
            self.width = (self.width + 63) // 64 * 64

        if self.width >= 1920 and self.height >= 1080:
            if self.forceStatic is not True:
                self.forceStatic = True
                logAndPrint(
                    "Forcing static engine due to resolution being equal or greater than 1080p.",
                    "yellow",
                )
            if self.restoreMethod in ["scunet-tensorrt"]:
                logAndPrint(
                    "!WARNING:! SCUNET requires more than 24GB of VRAM for 1920x1080 resolutions or higher.",
                    "red",
                )

        self.filename = modelsMap(
            self.restoreMethod,
            modelType="onnx",
            half=self.half,
        )
        folderName = self.restoreMethod.replace("-tensorrt", "-onnx")
        self.modelPath = resolveWeightPath(
            folderName,
            self.filename,
            downloadModel=self.restoreMethod,
            modelType="onnx",
            half=self.half,
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
                inputsMax=[1, 3, self.height, self.width],
                forceStatic=self.forceStatic,
            )

        self.stream = torch.cuda.Stream()
        self.dummyInput = torch.zeros(
            (1, 3, self.height, self.width),
            device=checker.device,
            dtype=torch.float16 if self.half else torch.float32,
        )

        self.dummyOutput = torch.zeros(
            (1, 3, self.height, self.width),
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
            if self.originalHeight != self.height or self.originalWidth != self.width:
                frame = torch.nn.functional.interpolate(
                    frame,
                    size=(self.height, self.width),
                    mode="bilinear",
                    align_corners=False,
                )
            self.dummyInput.copy_(
                frame.to(dtype=self.dtype),
                non_blocking=False,
            )
        self.normStream.synchronize()

    @torch.inference_mode()
    def processOutput(self):
        with torch.cuda.stream(self.outputStream):
            output = self.dummyOutput[
                :, :, : self.originalHeight, : self.originalWidth
            ].clamp(0, 1)
        self.outputStream.synchronize()

        return output

    @torch.inference_mode()
    def __call__(self, frame):
        self.processFrame(frame)

        with torch.cuda.stream(self.stream):
            self.cudaGraph.replay()
        self.stream.synchronize()
        return self.processOutput()


class UnifiedRestoreDirectML:
    def __init__(
        self,
        restoreMethod: str = "anime1080fixer-tensorrt",
        half: bool = False,
        width: int = 1920,
        height: int = 1080,
    ):
        """
        Initialize the upscaler with the desired model

        Args:
            restoreMethod (str): The method to use for upscaling
            half (bool): Whether to use half precision
            width (int): The width of the input frame
            height (int): The height of the input frame
        """
        if "openvino" in restoreMethod:
            logAndPrint(
                "OpenVINO backend is an experimental feature, please report any issues you encounter.",
                "yellow",
            )
            import openvino  # noqa: F401

        import numpy as np
        import onnxruntime as ort

        self.ort = ort
        self.np = np
        self.ort.set_default_logger_severity(3)

        self.restoreMethod = restoreMethod
        self.half = half
        self.width = width
        self.height = height

        self.handleModel()

    def handleModel(self):
        """
        Load the desired model
        """
        if ADOBE:
            progressState.update(
                {"status": f"Loading DirectML restore model: {self.restoreMethod}..."}
            )

        method = self.restoreMethod
        if "openvino" in self.restoreMethod:
            method = method.replace("openvino", "directml")

        self.filename = modelsMap(method, modelType="onnx")
        if "-directml" in self.restoreMethod:
            folderName = self.restoreMethod.replace("-directml", "-onnx")
        elif "-openvino" in self.restoreMethod:
            folderName = self.restoreMethod.replace("-openvino", "-onnx")

        modelPath = resolveWeightPath(
            folderName,
            self.filename,
            downloadModel=method,
            modelType="onnx",
            half=self.half,
        )

        providers = self.ort.get_available_providers()
        logging.info(f"Available ONNX Runtime providers: {providers}")

        if (
            "DmlExecutionProvider" in providers
            or "OpenVINOExecutionProvider" in providers
        ):
            if "directml" in self.restoreMethod:
                logging.info("Using DirectML model")
                self.model = self.ort.InferenceSession(
                    modelPath, providers=["DmlExecutionProvider"]
                )
                warnIfProviderMissing(
                    self.model, "DmlExecutionProvider", "DirectML restore"
                )
            elif "openvino" in self.restoreMethod:
                self.model = self.ort.InferenceSession(
                    modelPath,
                    providers=[
                        ("OpenVINOExecutionProvider", {"device_type": "AUTO:GPU,CPU"})
                    ],
                )
                warnIfProviderMissing(
                    self.model, "OpenVINOExecutionProvider", "OpenVINO restore"
                )
        else:
            logWarning(
                f"{self.restoreMethod} provider not available, falling back to CPU, expect significantly worse performance, ensure that your drivers are up to date and your GPU supports DirectX 12"
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
            (1, 3, self.height, self.width),
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

        self.usingCpuFallback = True

    @torch.inference_mode()
    def __call__(self, frame: torch.tensor) -> torch.tensor:
        """
        Run the model on the input frame
        """
        try:
            if self.half:
                frame = frame.half()
            else:
                frame = frame.float()

            self.dummyInput.copy_(frame.contiguous())

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
                logWarning(
                    f"DirectML/OpenVINO inference failed, falling back to CPU: {e}"
                )
                self._fallbackToCpu()
                return self.__call__(frame)
            else:
                logging.exception(
                    f"Something went wrong while processing the frame, {e}"
                )
                raise
