import os
import torch
import numpy as np
import logging
import tensorrt as trt

from spandrel import ImageModelDescriptor, ModelLoader
from .downloadModels import downloadModels, weightsDir, modelsMap
from .coloredPrints import yellow

# Apparently this can improve performance slightly
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
        nt: int = 1,
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
        self.nt = nt

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
            # self.stream = [torch.cuda.Stream() for _ in range(self.nt)]
            # self.currentStream = 0
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            if self.half:
                torch.set_default_dtype(torch.float16)
                self.model.half()

    @torch.inference_mode()
    def run(self, frame: np.ndarray) -> np.ndarray:
        """
        Upscale a frame using a desired model, and return the upscaled frame
        Expects a numpy array of shape (height, width, 3) and dtype uint8
        """
        if self.half:
            frame = frame.permute(2, 0, 1).unsqueeze(0).to(self.device).half().mul_(1 / 255)
        else:
            frame = frame.permute(2, 0, 1).unsqueeze(0).to(self.device).float().mul_(1 / 255)

        """
        if self.isCudaAvailable:
            torch.cuda.synchronize(self.stream[self.currentStream])
            self.currentStream = (self.currentStream + 1) % len(self.stream)
        """

        return self.model(frame).squeeze(0).permute(1, 2, 0).mul_(255)


class UniversalTensorRT:
    def __init__(
        self,
        upscaleMethod: str = "shufflecugan-tensorrt",
        upscaleFactor: int = 2,
        half: bool = False,
        width: int = 1920,
        height: int = 1080,
        customModel: str = None,
        nt: int = 1,
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
        from polygraphy.backend.trt import (
            TrtRunner,
            engine_from_network,
            network_from_onnx_path,
            CreateConfig,
            Profile,
            EngineFromBytes,
            SaveEngine,
        )
        from polygraphy.backend.common import BytesFromPath

        self.TrtRunner = TrtRunner
        self.engine_from_network = engine_from_network
        self.network_from_onnx_path = network_from_onnx_path
        self.CreateConfig = CreateConfig
        self.Profile = Profile
        self.EngineFromBytes = EngineFromBytes
        self.SaveEngine = SaveEngine
        self.BytesFromPath = BytesFromPath

        self.upscaleMethod = upscaleMethod
        self.upscaleFactor = upscaleFactor
        self.half = half
        self.width = width
        self.height = height
        self.customModel = customModel
        self.nt = nt

        self.handleModel()

    def handleModel(self):
        if not self.customModel:
            # For some reason this runs out of VRAM on my f 3090, so we'll just use the alr existing one
            if self.upscaleMethod != "shufflecugan-tensorrt":
                modelType = "pth"
                self.upscaleMethod = self.upscaleMethod.replace("-tensorrt", "")
            else:
                modelType = "onnx"

            self.filename = modelsMap(
                self.upscaleMethod, self.upscaleFactor, modelType=modelType, half=self.half
            )
            if not os.path.exists(
                os.path.join(weightsDir, self.upscaleMethod, self.filename)
            ):
                modelPath = downloadModels(
                    model=self.upscaleMethod,
                    upscaleFactor=self.upscaleFactor,
                    half=self.half,
                    modelType=modelType,
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
        
        if modelType == "pth":
            try:
                self.model = ModelLoader().load_from_file(modelPath)
            except Exception as e:
                logging.error(f"Error loading model: {e}")

            if self.customModel:
                assert isinstance(self.model, ImageModelDescriptor)

        self.isCudaAvailable = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.isCudaAvailable else "cpu")
        if self.isCudaAvailable:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            if self.half:
                torch.set_default_dtype(torch.float16)

        enginePrecision = "fp16" if self.half else "fp32"
        if modelType == "pth":
            self.model = self.model.half() if self.half and self.isCudaAvailable else self.model
            self.model = self.model.eval().to(self.device).model

            if not os.path.exists(modelPath.replace(f".{modelType}", f"_{enginePrecision}.onnx")):
                torch.onnx.export(
                    self.model,
                    torch.zeros(1, 3, 256, 256, device=self.device, dtype=torch.float16 if self.half else torch.float32),
                    modelPath.replace(f".{modelType}", f"_{enginePrecision}.onnx"),
                    opset_version=19,
                    input_names=["input"],
                    output_names=["output"],
                    dynamic_axes={"input": {0: "batch", 2: "height", 3: "width"}, "output": {0: "batch", 2: "height", 3: "width"}},
                )

            modelPath = modelPath.replace(f".{modelType}", f"_{enginePrecision}.onnx")

        if not os.path.exists(modelPath.replace(f"_{enginePrecision}.onnx", f"_{enginePrecision}.engine")):
            toPrint = f"Model engine not found, creating engine for model: {modelPath}, this may take a while..."
            print(yellow(toPrint))
            logging.info(toPrint)
            profiles = [
                self.Profile().add(
                    "input",
                    min=(1, 3, 8, 8),
                    opt=(1, 3, self.height, self.width),
                    max=(1, 3, 1080, 1920),
                ),
            ]
            self.engine = self.engine_from_network(
                self.network_from_onnx_path(modelPath),
                config=self.CreateConfig(fp16=self.half, profiles=profiles),
            )
            self.engine = self.SaveEngine(
                self.engine, modelPath.replace(".onnx", ".engine")
            )

            with self.TrtRunner(self.engine) as runner:
                self.runner = runner

            """
        else:
            self.engine = self.EngineFromBytes(
                self.BytesFromPath(modelPath.replace(f"_{enginePrecision}.onnx", f"_{enginePrecision}.engine"))
            )
            """
            """
            with open(args.engine, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime,
                runtime.deserialize_cuda_engine(f.read()) as engine, engine.create_execution_context() as context:
            """

        with open(modelPath.replace(f"_{enginePrecision}.onnx", f"_{enginePrecision}.engine"), "rb") as f, trt.Runtime(trt.Logger(trt.Logger.INFO)) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read()) 
            self.context = self.engine.create_execution_context()
        
        #self.runner = self.TrtRunner(self.engine)
        #self.runner.activate()

        
        self.stream = torch.cuda.Stream()
        self.dummyInput = torch.zeros(
            (1, 3, self.height, self.width),
            device=self.device,
            dtype=torch.float16 if self.half else torch.float32,
        )

        self.dummyOutput = torch.zeros(
            (1, 3, self.height * self.upscaleFactor, self.width * self.upscaleFactor),
            device=self.device,
            dtype=torch.float16 if self.half else torch.float32,
        )

        self.bindings = [self.dummyInput.data_ptr(), self.dummyOutput.data_ptr()]

        for i in range(self.engine.num_io_tensors):
            self.context.set_tensor_address(self.engine.get_tensor_name(i), self.bindings[i])
            tensor_name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(tensor_name, self.dummyInput.shape)

        
        with torch.cuda.stream(self.stream):
            for _ in range(10):
                self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
                self.stream.synchronize()

                
        
        
    @torch.inference_mode()
    def run(self, frame):
        with torch.cuda.stream(self.stream):
            if self.half:
                self.dummyInput.copy_(frame.permute(2, 0, 1).unsqueeze(0).half().mul_(1 / 255))
            else:
                self.dummyInput.copy_(frame.permute(2, 0, 1).unsqueeze(0).float().mul_(1 / 255))
            self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
            self.stream.synchronize()
        
            return self.dummyOutput.squeeze(0).permute(1, 2, 0).mul_(255).clamp(0, 255)

        """
        return (
            self.runner.infer(
                {
                    "input": frame.contiguous(),
                },
                check_inputs=False,
            )["output"]
            .squeeze(0)
            .permute(1, 2, 0)
            .mul_(255)
            .clamp(0, 255)  # Sadge but it had to be done, I love TRT 10 <3 
        )
        """


class UniversalDirectML:
    def __init__(
        self,
        upscaleMethod: str,
        upscaleFactor: int,
        half: bool,
        width: int,
        height: int,
        customModel: str,
        nt: int,
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

        ort.set_default_logger_severity(3)

        self.ort = ort

        self.upscaleMethod = upscaleMethod
        self.upscaleFactor = upscaleFactor
        self.half = half
        self.width = width
        self.height = height
        self.customModel = customModel
        self.nt = nt

        self.handleModel()

    def handleModel(self):
        """
        Load the desired model
        """

        if not self.customModel:
            self.filename = modelsMap(
                self.upscaleMethod, self.upscaleFactor, modelType="onnx"
            )
            if not os.path.exists(
                os.path.join(weightsDir, self.upscaleMethod, self.filename)
            ):
                modelPath = downloadModels(
                    model=self.upscaleMethod,
                    upscaleFactor=self.upscaleFactor,
                    modelType="onnx",
                    half=self.half,
                )
            else:
                modelPath = os.path.join(weightsDir, self.upscaleMethod, self.filename)
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
            self.numpyDType = np.float16
            self.torchDType = torch.float16
        else:
            self.numpyDType = np.float32
            self.torchDType = torch.float32

        self.IoBinding = self.model.io_binding()
        self.dummyInput = torch.zeros(
            (1, 3, self.height, self.width),
            device=self.deviceType,
            dtype=self.torchDType,
        )
        self.dummyInput = self.dummyInput.contiguous()

        self.dummyOutput = torch.zeros(
            (1, 3, self.height * self.upscaleFactor, self.width * self.upscaleFactor),
            device=self.deviceType,
            dtype=self.torchDType,
        )

        self.IoBinding.bind_output(
            name="output",
            device_type=self.deviceType,
            device_id=0,
            element_type=self.numpyDType,
            shape=self.dummyOutput.shape,
            buffer_ptr=self.dummyOutput.data_ptr(),
        )

    def run(self, frame: torch.tensor) -> torch.tensor:
        if self.half:
            frame = (
                frame
                .permute(2, 0, 1)
                .unsqueeze(0)
                .half()
                .mul_(1 / 255)
            )
        else:
            frame = (
                frame
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                .mul_(1 / 255)
            )

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
        frame = (
            self.dummyOutput.squeeze(0)
            .permute(1, 2, 0)
            .mul_(255)
            .contiguous()
        )

        return frame