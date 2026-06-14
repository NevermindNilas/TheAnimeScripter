import logging
import os

import torch

from src.constants import ADOBE
from src.utils.downloadModels import (
    modelsMap,
    resolveWeightPath,
)
from src.utils.isCudaInit import CudaChecker
from src.utils.logAndPrint import logAndPrint

if ADOBE:
    from src.utils.aeComms import progressState

checker = CudaChecker()

torch.set_float32_matmul_precision("medium")


def calculatePadding(width, height, multiple=4):
    padW = (multiple - (width % multiple)) % multiple
    padH = (multiple - (height % multiple)) % multiple
    return (0, padW, 0, padH)


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

        from src.utils.trtHandler import (
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
        import numpy as np
        import onnxruntime as ort

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
