import os
import torch
import logging

from src.utils.modelOptimizer import ModelOptimizer
from .utils.downloadModels import downloadModels, weightsDir, modelsMap
from .utils.isCudaInit import CudaChecker
from src.utils.logAndPrint import logAndPrint

checker = CudaChecker()

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
            trt (bool): Whether to use tensorRT
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
        from src.spandrel import (
            ImageModelDescriptor,
            ModelLoader,
            UnsupportedDtypeError,
        )

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

        self.model = (
            self.model.eval().cuda() if checker.cudaAvailable else self.model.eval()
        )

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

        self.model = ModelOptimizer(
            self.model,
            torch.float16 if self.half else torch.float32,
            memoryFormat=torch.channels_last,
        ).optimizeModel()

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
            for _ in range(5):
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
                frame.to(dtype=self.dummyInput.dtype).to(
                    memory_format=torch.channels_last
                ),
                non_blocking=False,
            )
        self.normStream.synchronize()

    @torch.inference_mode()
    def __call__(self, frame: torch.tensor, nextFrame: None) -> torch.tensor:
        self.processFrame(frame)
        if not self.compileMode != "default":
            with torch.cuda.stream(self.stream):
                self.cudaGraph.replay()
            self.stream.synchronize()
        else:
            with torch.cuda.stream(self.stream):
                self.dummyOutput.copy_(
                    self.model(self.dummyInput),
                    non_blocking=True,
                )
            self.stream.synchronize()

        with torch.cuda.stream(self.outputStream):
            output = self.dummyOutput.clone()
        self.outputStream.synchronize()

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

        # Experimental feature, may not work as expected
        with torch.cuda.stream(self.stream):
            self.cudaGraph.replay()
        self.stream.synchronize()

        with torch.cuda.stream(self.outputStream):
            output = self.dummyOutput.clone()
        self.outputStream.synchronize()

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

    def __call__(self, frame: torch.tensor, nextFrame: None) -> torch.tensor:
        """
        Run the model on the input frame
        """

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

        self.model.run_with_iobinding(self.IoBinding)
        frame = self.dummyOutput.contiguous()

        return frame


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

    def __call__(self, frame, nextFrame: None) -> torch.tensor:
        iniFrameDtype = frame.dtype
        frame = self.model.process_torch(
            frame.mul(255).to(torch.uint8).squeeze(0).permute(1, 2, 0).cpu()
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
        if not os.path.exists(os.path.join(weightsDir, "animesr", self.filename)):
            modelPath = downloadModels(
                model="animesr",
                upscaleFactor=self.upscaleFactor,
            )
        else:
            modelPath = os.path.join(weightsDir, "animesr", self.filename)

        from src.extraArches.AnimeSR import MSRSWVSR

        self.model = MSRSWVSR(num_feat=64, num_block=[5, 3, 2], netscale=4)

        self.model.load_state_dict(torch.load(modelPath))

        self.model = (
            self.model.eval().cuda() if checker.cudaAvailable else self.model.eval()
        )

        if self.half and checker.cudaAvailable:
            try:
                self.model = self.model.half()
            except Exception as e:
                logging.error(f"Error converting model to half precision: {e}")
                self.model = self.model.float()
                self.half = False

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

        self.firstIter = True

    def padFrame(self, frame: torch.tensor) -> torch.tensor:
        return torch.nn.functional.pad(frame, self.padding, mode="reflect")

    @torch.inference_mode()
    def __call__(self, frame: torch.tensor, nextFrame: torch.tensor) -> torch.tensor:
        if self.firstIter:
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

            self.firstIter = False
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
                torch.cat((self.prevFrame, frame, self.nextFrame), dim=1),
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
        folderName = "animesr-onnx"
        if not os.path.exists(os.path.join(weightsDir, folderName, self.filename)):
            self.modelPath = downloadModels(
                model="animesr-tensorrt",
                upscaleFactor=self.upscaleFactor,
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
            inputs = [
                [
                    1,
                    9,
                    self.paddedHeight,
                    self.paddedWidth,
                ],  # x: 3 padded frames concatenated
                [
                    1,
                    3,
                    self.paddedHeight * 4,
                    self.paddedWidth * 4,
                ],  # fb: padded output
                [
                    1,
                    64,
                    self.paddedHeight,
                    self.paddedWidth,
                ],  # state: uses padded dims too
            ]

            inputsMin = inputsOpt = inputsMax = inputs
            inputNames = ["x", "fb", "state"]

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
        ).to(memory_format=torch.channels_last)
        self.nextFrame = torch.zeros(
            (1, 3, self.paddedHeight, self.paddedWidth),
            device=checker.device,
            dtype=torch.float16 if self.half else torch.float32,
        ).to(memory_format=torch.channels_last)

        # Separate input feedback buffer and output buffer - MUST use padded dimensions
        self.feedbackBuffer = self.prevFrame.new_zeros(
            1, 3, self.paddedHeight * 4, self.paddedWidth * 4
        ).to(memory_format=torch.channels_last)

        self.dummyOutput = self.prevFrame.new_zeros(
            1, 3, self.paddedHeight * 4, self.paddedWidth * 4
        ).to(memory_format=torch.channels_last)

        # State tensors MUST use padded dimensions to match the engine
        self.state = self.prevFrame.new_zeros(
            1, 64, self.paddedHeight, self.paddedWidth
        )
        self.stateOutput = self.state.new_zeros(
            1, 64, self.paddedHeight, self.paddedWidth
        )

        self.dummyInput = torch.cat(
            (self.prevFrame, self.prevFrame, self.nextFrame), dim=1
        )

        # AnimeSR has 3 inputs (x, fb, state) and 2 outputs (out_img, out_state)
        self.bindings = {
            "x": self.dummyInput.data_ptr(),
            "fb": self.feedbackBuffer.data_ptr(),
            "state": self.state.data_ptr(),
            "out_img": self.dummyOutput.data_ptr(),
            "out_state": self.stateOutput.data_ptr(),
        }

        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)

            if tensor_name == "x":
                self.context.set_tensor_address(tensor_name, self.bindings["x"])
                self.context.set_input_shape(tensor_name, self.dummyInput.shape)
            elif tensor_name == "fb":
                self.context.set_tensor_address(tensor_name, self.bindings["fb"])
                self.context.set_input_shape(tensor_name, self.feedbackBuffer.shape)
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

        self.firstIter = True

    def padFrame(self, frame: torch.tensor) -> torch.tensor:
        return torch.nn.functional.pad(frame, self.padding, mode="reflect")

    @torch.inference_mode()
    def __call__(self, frame: torch.tensor, nextFrame: torch.tensor) -> torch.tensor:
        with torch.cuda.stream(self.normStream):
            frame = self.padFrame(frame)
        self.normStream.synchronize()

        if self.firstIter:
            with torch.cuda.stream(self.normStream):
                self.prevFrame.copy_(
                    frame.to(dtype=self.dtype).to(memory_format=torch.channels_last),
                    non_blocking=False,
                )
                if nextFrame is None:
                    self.nextFrame.copy_(
                        frame.to(dtype=self.dtype).to(
                            memory_format=torch.channels_last
                        ),
                        non_blocking=False,
                    )
                else:
                    paddedNextFrame = self.padFrame(nextFrame)
                    self.nextFrame.copy_(
                        paddedNextFrame.to(dtype=self.dtype).to(
                            memory_format=torch.channels_last
                        ),
                        non_blocking=False,
                    )
            self.normStream.synchronize()

            self.firstIter = False

        else:
            with torch.cuda.stream(self.normStream):
                if nextFrame is None:
                    self.nextFrame.copy_(
                        frame.to(dtype=self.dtype).to(
                            memory_format=torch.channels_last
                        ),
                        non_blocking=False,
                    )
                else:
                    paddedNextFrame = self.padFrame(nextFrame)
                    self.nextFrame.copy_(
                        paddedNextFrame.to(dtype=self.dtype).to(
                            memory_format=torch.channels_last
                        ),
                        non_blocking=False,
                    )
            self.normStream.synchronize()

        with torch.cuda.stream(self.normStream):
            self.dummyInput.copy_(
                torch.cat((self.prevFrame, frame, self.nextFrame), dim=1),
                non_blocking=False,
            )
        self.normStream.synchronize()

        with torch.cuda.stream(self.outputStream):
            self.context.execute_async_v3(stream_handle=self.outputStream.cuda_stream)
        self.outputStream.synchronize()

        with torch.cuda.stream(self.normStream):
            self.feedbackBuffer.copy_(self.dummyOutput, non_blocking=False)
            self.state.copy_(self.stateOutput, non_blocking=False)
            self.prevFrame.copy_(frame, non_blocking=False)
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
