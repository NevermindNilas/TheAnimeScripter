import os
import torch

from src.model.download import resolveWeightPath
from src.model.registry import modelsMap
from src.infra.logAndPrint import logAndPrint
from src.constants import ADOBE

from ._shared import checker

if ADOBE:
    from src.server.aeComms import progressState


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
        from src.model.trtHandler import (
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
