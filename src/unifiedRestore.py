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
            frame = (
                self.model(
                    frame.to(self.device, non_blocking=True, dtype=self.dType)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .to(memory_format=torch.channels_last)
                )
                .squeeze_(0)
                .permute(1, 2, 0)
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
            TensorRTEngineCreator,
            TensorRTEngineLoader,
            TensorRTEngineNameHandler,
        )

        self.trt = trt
        self.TensorRTEngineCreator = TensorRTEngineCreator
        self.TensorRTEngineLoader = TensorRTEngineLoader
        self.TensorRTEngineNameHandler = TensorRTEngineNameHandler

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

    @torch.inference_mode()
    def processFrame(self, frame):
        with torch.cuda.stream(self.normStream):
            self.dummyInput.copy_(
                frame.to(dtype=self.dtype).permute(2, 0, 1).unsqueeze(0),
                non_blocking=False,
            )
        self.normStream.synchronize()

    @torch.inference_mode()
    def processOutput(self):
        with torch.cuda.stream(self.outputStream):
            output = self.dummyOutput.squeeze(0).permute(1, 2, 0).clamp(0, 1)
        self.outputStream.synchronize()

        return output

    @torch.inference_mode()
    def __call__(self, frame):
        self.processFrame(frame)

        self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        self.stream.synchronize()
        output = self.processOutput()
        return output
