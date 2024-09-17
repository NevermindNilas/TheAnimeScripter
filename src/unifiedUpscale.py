import os
import torch
import logging

from spandrel import ImageModelDescriptor, ModelLoader
from .downloadModels import downloadModels, weightsDir, modelsMap

torch.set_float32_matmul_precision("medium")


class UniversalPytorch:
    def __init__(
        self,
        upscaleMethod: str = "shufflecugan",
        upscaleFactor: int = 2,
        half: bool = False,
        width: int = 1920,
        height: int = 1080,
        customModel: str = None,
        upscaleSkip: bool | None = None,
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
            nt (int): The number of threads to use
            trt (bool): Whether to use tensorRT
        """
        self.upscaleMethod = upscaleMethod
        self.upscaleFactor = upscaleFactor
        self.half = half
        self.width = width
        self.height = height
        self.customModel = customModel
        self.upscaleSkip = upscaleSkip

        self.handleModel()

    def handleModel(self):
        """
        Load the desired model
        """
        if not self.customModel:
            self.filename = modelsMap(
                self.upscaleMethod, self.upscaleFactor, modelType="pth"
            )
            if not os.path.exists(
                os.path.join(weightsDir, self.upscaleMethod, self.filename)
            ):
                modelPath = downloadModels(
                    model=self.upscaleMethod,
                    upscaleFactor=self.upscaleFactor,
                )
            else:
                modelPath = os.path.join(weightsDir, self.upscaleMethod, self.filename)
        else:
            if os.path.isfile(self.customModel):
                modelPath = self.customModel
            else:
                raise FileNotFoundError(
                    f"Custom model file {self.customModel} not found"
                )
        try:
            self.model = ModelLoader().load_from_file(modelPath)
        except Exception as e:
            logging.error(f"Error loading model: {e}")

        if self.customModel:
            assert isinstance(self.model, ImageModelDescriptor)

        self.isCudaAvailable = torch.cuda.is_available()
        self.model = (
            self.model.eval().cuda() if self.isCudaAvailable else self.model.eval()
        )

        self.device = torch.device("cuda" if self.isCudaAvailable else "cpu")
        if self.isCudaAvailable:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            if self.half:
                torch.set_default_dtype(torch.float16)
                self.model.half()

        self.stream = torch.cuda.Stream()
        if self.upscaleSkip is not None:
            self.skippedCounter = 0
            self.prevFrame = torch.zeros(
                (self.height * self.upscaleFactor, self.width * self.upscaleFactor, 3),
                device=self.device,
                dtype=torch.float16 if self.half else torch.float32,
            )

        self.model = self.model.model.to(memory_format=torch.channels_last)

    def __call__(self, frame: torch.tensor) -> torch.tensor:
        with torch.cuda.stream(self.stream):
            if self.upscaleSkip is not None:
                if self.upscaleSkip(frame):
                    self.skippedCounter += 1
                    return self.prevFrame

            frame = (
                frame.to(
                    self.device,
                    non_blocking=True,
                    dtype=torch.float16 if self.half else torch.float32,
                )
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(memory_format=torch.channels_last)
                .mul(1 / 255)
            )
            output = self.model(frame).squeeze(0).mul(255).permute(1, 2, 0)
            self.stream.synchronize()

            if self.upscaleSkip is not None:
                self.prevFrame.copy_(output, non_blocking=True)

            return output

    def getSkippedCounter(self):
        return self.skippedCounter


class UniversalTensorRT:
    def __init__(
        self,
        upscaleMethod: str = "shufflecugan-tensorrt",
        upscaleFactor: int = 2,
        half: bool = False,
        width: int = 1920,
        height: int = 1080,
        customModel: str = None,
        upscaleSkip: bool | None = None,
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
            nt (int): The number of threads to use
        """

        # Attempt to lazy load for faster startup

        import tensorrt as trt
        from .utils.trtHandler import (
            TensorRTEngineCreator,
            TensorRTEngineLoader,
            TensorRTEngineNameHandler,
        )

        self.trt = trt
        self.TensorRTEngineCreator = TensorRTEngineCreator
        self.TensorRTEngineLoader = TensorRTEngineLoader
        self.TensorRTEngineNameHandler = TensorRTEngineNameHandler

        self.upscaleMethod = upscaleMethod
        self.upscaleFactor = upscaleFactor
        self.half = half
        self.width = width
        self.height = height
        self.customModel = customModel
        self.upscaleSkip = upscaleSkip
        self.forceStatic = forceStatic

        self.handleModel()

    def handleModel(self):
        if self.width > 1920 and self.height > 1080:
            self.forceStatic = True
            logging.info(
                "Forcing static engine due to resolution higher than 1080p, wtf are you upscaling?"
            )

        if not self.customModel:
            self.filename = modelsMap(
                self.upscaleMethod,
                self.upscaleFactor,
                modelType="onnx",
                half=self.half,
            )
            folderName = self.upscaleMethod.replace("-tensorrt", "-onnx")
            if not os.path.exists(os.path.join(weightsDir, folderName, self.filename)):
                self.modelPath = downloadModels(
                    model=self.upscaleMethod,
                    upscaleFactor=self.upscaleFactor,
                    half=self.half,
                    modelType="onnx",
                )
            else:
                self.modelPath = os.path.join(weightsDir, folderName, self.filename)
        else:
            self.modelPath = self.customModel
            if not os.path.exists(self.customModel):
                raise FileNotFoundError(
                    f"Custom model file {self.customModel} not found"
                )

        self.dtype = torch.float16 if self.half else torch.float32
        self.isCudaAvailable = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.isCudaAvailable else "cpu")
        if self.isCudaAvailable:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            if self.half:
                torch.set_default_dtype(torch.float16)

        enginePath = self.TensorRTEngineNameHandler(
            modelPath=self.modelPath,
            fp16=self.half,
            optInputShape=[1, 3, self.height, self.width],
        )

        self.engine, self.context = self.TensorRTEngineLoader(enginePath)
        if (
            self.engine is None
            or self.context is None
            or not os.path.exists(enginePath)
        ):
            self.engine, self.context = self.TensorRTEngineCreator(
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
        ).contiguous()

        self.dummyOutput = torch.zeros(
            (1, 3, self.height * self.upscaleFactor, self.width * self.upscaleFactor),
            device=self.device,
            dtype=torch.float16 if self.half else torch.float32,
        ).contiguous()

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

        if self.upscaleSkip is not None:
            self.skippedCounter = 0
            self.prevFrame = torch.zeros(
                (self.height * self.upscaleFactor, self.width * self.upscaleFactor, 3),
                device=self.device,
                dtype=torch.float16 if self.half else torch.float32,
            )

        self.normStream = torch.cuda.Stream()

    @torch.inference_mode()
    def processFrame(self, frame):
        with torch.cuda.stream(self.normStream):
            self.dummyInput.copy_(
                frame.to(dtype=self.dtype).permute(2, 0, 1).unsqueeze(0).mul(1 / 255),
                non_blocking=True,
            )

    @torch.inference_mode()
    def __call__(self, frame):
        if self.upscaleSkip is not None:
            if self.upscaleSkip(frame):
                self.skippedCounter += 1
                return self.prevFrame

        self.processFrame(frame)
        self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        output = self.dummyOutput.squeeze(0).permute(1, 2, 0).clamp(0, 1).mul(255).cpu()
        self.stream.synchronize()

        if self.upscaleSkip is not None:
            self.prevFrame.copy_(output, non_blocking=True)

        return output

    def getSkippedCounter(self):
        return self.skippedCounter


class UniversalDirectML:
    def __init__(
        self,
        upscaleMethod: str,
        upscaleFactor: int,
        half: bool,
        width: int,
        height: int,
        customModel: str,
        upscaleSkip: bool | None = None,
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
            nt (int): The number of threads to use
        """

        import onnxruntime as ort
        import numpy as np

        self.ort = ort
        self.np = np
        self.ort.set_default_logger_severity(3)

        self.upscaleMethod = upscaleMethod
        self.upscaleFactor = upscaleFactor
        self.half = half
        self.width = width
        self.height = height
        self.customModel = customModel
        self.upscaleSkip = upscaleSkip

        self.handleModel()

    def handleModel(self):
        """
        Load the desired model
        """

        if not self.customModel:
            self.filename = modelsMap(
                self.upscaleMethod, self.upscaleFactor, modelType="onnx"
            )
            folderName = self.upscaleMethod.replace("directml", "-onnx")
            if not os.path.exists(os.path.join(weightsDir, folderName, self.filename)):
                modelPath = downloadModels(
                    model=self.upscaleMethod,
                    upscaleFactor=self.upscaleFactor,
                    modelType="onnx",
                    half=self.half,
                )
            else:
                modelPath = os.path.join(weightsDir, folderName, self.filename)
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
            (1, 3, self.height * self.upscaleFactor, self.width * self.upscaleFactor),
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

        if self.upscaleSkip is not None:
            self.prevFrame = torch.zeros(
                (self.height * self.upscaleFactor, self.width * self.upscaleFactor, 3),
                device=self.device,
                dtype=self.torchDType,
            )
            self.skippedCounter = 0

    def __call__(self, frame: torch.tensor) -> torch.tensor:
        """
        Run the model on the input frame
        """
        if self.upscaleSkip is not None:
            if self.upscaleSkip.run(frame):
                self.skippedCounter += 1
                return self.prevFrame

        self.IoBinding.bind_input(
            name="input",
            device_type=self.deviceType,
            device_id=0,
            element_type=self.numpyDType,
            shape=self.dummyInput.shape,
            buffer_ptr=self.dummyInput.data_ptr(),
        )

        if self.half:
            frame = frame.permute(2, 0, 1).unsqueeze(0).half().mul(1 / 255)
        else:
            frame = frame.permute(2, 0, 1).unsqueeze(0).float().mul(1 / 255)

        self.dummyInput.copy_(frame.contiguous())

        self.model.run_with_iobinding(self.IoBinding)
        frame = self.dummyOutput.squeeze(0).permute(1, 2, 0).mul(255).contiguous()

        if self.upscaleSkip is not None:
            self.prevFrame.copy_(frame, non_blocking=True)

        return frame

    def getSkippedCounter(self):
        return self.skippedCounter


class UniversalNCNN:
    def __init__(self, upscaleMethod, upscaleFactor, upscaleSkip):
        self.upscaleMethod = upscaleMethod
        self.upscaleFactor = upscaleFactor
        self.upscaleSkip = upscaleSkip

        from upscale_ncnn_py import UPSCALE

        self.filename = modelsMap(
            self.upscaleMethod,
            modelType="ncnn",
        )

        if self.filename.endswith("-ncnn.zip"):
            self.filename = self.filename[:-9]
        elif self.filename.endswith("-ncnn"):
            self.filename = self.filename[:-5]

        if not os.path.exists(
            os.path.join(weightsDir, self.upscaleMethod, self.filename)
        ):
            modelPath = downloadModels(
                model=self.upscaleMethod,
                modelType="ncnn",
            )
        else:
            modelPath = os.path.join(weightsDir, self.upscaleMethod, self.filename)

        if modelPath.endswith("-ncnn.zip"):
            modelPath = modelPath[:-9]
        elif modelPath.endswith("-ncnn"):
            modelPath = modelPath[:-5]

        lastSlash = modelPath.split("\\")[-1]
        modelPath = modelPath + "\\" + lastSlash

        self.model = UPSCALE(
            gpuid=0,
            tta_mode=False,
            tilesize=0,
            model_str=modelPath,
            num_threads=2,
        )

        if self.upscaleSkip is not None:
            self.skippedCounter = 0
            self.prevFrame = None

    def __call__(self, frame):
        if self.upscaleSkip is not None:
            if self.upscaleSkip.run(frame):
                self.skippedCounter += 1
                return self.prevFrame

        frame = self.model.process_torch(frame)

        if self.upscaleSkip is not None:
            self.prevFrame = frame

        return frame

    def getSkippedCounter(self):
        return self.skippedCounter
