import os
import torch
import logging

from .utils.downloadModels import downloadModels, weightsDir, modelsMap
from .utils.isCudaInit import CudaChecker
from .utils.logAndPrint import logAndPrint

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
        from src.spandrel import ModelLoader

        if self.model in ["nafnet"]:
            self.half = False
            print("NAFNet does not support half precision, using float32 instead")

        self.filename = modelsMap(self.model)
        if not os.path.exists(os.path.join(weightsDir, self.model, self.filename)):
            modelPath = downloadModels(model=self.model)
        else:
            modelPath = os.path.join(weightsDir, self.model, self.filename)

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

        try:
            # Weird spandrel hack to bypass ModelDecriptor
            self.model = self.model.model
        except Exception:
            pass

        self.model = (
            self.model.eval().cuda() if checker.cudaAvailable else self.model.eval()
        )

        if self.half:
            self.model.half()
            self.dType = torch.float16
        else:
            self.model.float()  # Sanity check, should not be needed
            self.dType = torch.float32
        self.stream = torch.cuda.Stream()

        if self.CHANNELSLAST:
            self.model.to(memory_format=torch.channels_last)
        else:
            self.model.to(memory_format=torch.contiguous_format)

    @torch.inference_mode()
    def __call__(self, frame: torch.tensor) -> torch.tensor:
        with torch.cuda.stream(self.stream):
            frame = self.model(
                frame.to(checker.device, non_blocking=True, dtype=self.dType).to(
                    memory_format=torch.channels_last
                )
                if self.CHANNELSLAST
                else frame.to(checker.device, non_blocking=True, dtype=self.dType)
            )
        self.stream.synchronize()
        return frame


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

        self.restoreMethod = restoreMethod
        self.half = half
        self.width = width
        self.height = height
        self.forceStatic = forceStatic

        self.handleModel()

    def handleModel(self):
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
        elif self.restoreMethod in ["codeformer-tensorrt"]:
            if self.forceStatic is not True:
                self.forceStatic = True
                logAndPrint(
                    "Forcing static engine due to Codeformer's limitations.",
                    "yellow",
                )

            self.width = 512
            self.height = 512

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
        if not os.path.exists(os.path.join(weightsDir, folderName, self.filename)):
            self.modelPath = downloadModels(
                model=self.restoreMethod,
                half=self.half,
                modelType="onnx",
            )
        else:
            self.modelPath = os.path.join(weightsDir, folderName, self.filename)

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
            import openvino # noqa: F401

        import onnxruntime as ort
        import numpy as np

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

        method = self.restoreMethod
        if "openvino" in self.restoreMethod:
            method = method.replace("openvino", "directml")

        self.filename = modelsMap(method, modelType="onnx")
        if "directml" in self.restoreMethod:
            folderName = self.restoreMethod.replace("directml", "-onnx")
        elif "openvino" in self.restoreMethod:
            folderName = self.restoreMethod.replace("openvino", "-onnx")

        if not os.path.exists(os.path.join(weightsDir, folderName, self.filename)):
            modelPath = downloadModels(
                model=method,
                modelType="onnx",
                half=self.half,
            )
        else:
            modelPath = os.path.join(weightsDir, folderName, self.filename)

        providers = self.ort.get_available_providers()
        logging.info(f"Available ONNX Runtime providers: {providers}")

        if "DmlExecutionProvider" or "OpenVINOExecutionProvider" in providers:
            if "directml" in self.restoreMethod:
                logging.info("Using DirectML model")
                self.model = self.ort.InferenceSession(
                    modelPath, providers=["DmlExecutionProvider"]
                )
            elif "openvino" in self.restoreMethod:
                self.model = self.ort.InferenceSession(
                    modelPath, providers=["OpenVINOExecutionProvider"]
                )
        else:
            logging.info(
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
        self.IoBinding.bind_output(
            name="output",
            device_type=self.deviceType,
            device_id=0,
            element_type=self.numpyDType,
            shape=self.dummyOutput.shape,
            buffer_ptr=self.dummyOutput.data_ptr(),
        )

        self.usingCpuFallback = True

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

            self.model.run_with_iobinding(self.IoBinding)
            frame = self.dummyOutput.contiguous()

            return frame

        except UnicodeDecodeError as e:
            if not self.usingCpuFallback:
                logging.warning(f"DirectML/OpenVINO UnicodeDecodeError: {e}")
                self._fallbackToCpu()
                return self.__call__(frame)
            else:
                logging.exception(f"Something went wrong while processing the frame, {e}")
                raise


