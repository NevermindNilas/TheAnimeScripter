import os
import torch
import logging

from .utils.downloadModels import downloadModels, weightsDir, modelsMap


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
        self.handleModel()

    def handleModel(self):
        """
        Load the Model
        """
        from spandrel import ModelLoader

        if self.model in ["nafnet"]:
            self.half = False
            print("NAFNet does not support half precision, using float32 instead")

        self.filename = modelsMap(self.model)
        if not os.path.exists(os.path.join(weightsDir, self.model, self.filename)):
            modelPath = downloadModels(model=self.model)
        else:
            modelPath = os.path.join(weightsDir, self.model, self.filename)

        try:
            self.model = ModelLoader().load_from_file(path=modelPath)

        except Exception as e:
            logging.error(f"Error loading model: {e}")

        self.isCudaAvailable = torch.cuda.is_available()
        self.model = (
            self.model.eval().cuda() if self.isCudaAvailable else self.model.eval()
        )

        if self.half:
            self.model.model.half()
            self.dType = torch.float16
        else:
            self.dType = torch.float32

        self.device = torch.device("cuda" if self.isCudaAvailable else "cpu")
        self.stream = torch.cuda.Stream()

        self.model.model.to(memory_format=torch.channels_last)

    @torch.inference_mode()
    def __call__(self, frame: torch.tensor) -> torch.tensor:
        with torch.cuda.stream(self.stream):
            frame = self.model(
                frame.to(self.device, non_blocking=True, dtype=self.dType).to(
                    memory_format=torch.channels_last
                )
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
            nt (int): The number of threads to use
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
        if self.width > 1920 and self.height > 1080:
            self.forceStatic = True
            logging.info(
                "Forcing static engine due to resolution higher than 1080p, wtf are you restoring?"
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
        self.isCudaAvailable = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.isCudaAvailable else "cpu")
        if self.isCudaAvailable:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            if self.half:
                torch.set_default_dtype(torch.float16)

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
            self.dummyInput.copy_(
                frame.to(dtype=self.dtype),
                non_blocking=False,
            )
        self.normStream.synchronize()

    @torch.inference_mode()
    def processOutput(self):
        with torch.cuda.stream(self.outputStream):
            output = self.dummyOutput.clamp(0, 1)
        self.outputStream.synchronize()

        return output

    @torch.inference_mode()
    def __call__(self, frame):
        self.processFrame(frame)

        with torch.cuda.stream(self.stream):
            self.cudaGraph.replay()
        self.stream.synchronize()
        output = self.processOutput()
        return output


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

        self.filename = modelsMap(self.restoreMethod, modelType="onnx")
        folderName = self.restoreMethod.replace("directml", "-onnx")
        if not os.path.exists(os.path.join(weightsDir, folderName, self.filename)):
            modelPath = downloadModels(
                model=self.restoreMethod,
                modelType="onnx",
                half=self.half,
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

    def __call__(self, frame: torch.tensor) -> torch.tensor:
        """
        Run the model on the input frame
        """
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

    def getSkippedCounter(self):
        return self.skippedCounter
