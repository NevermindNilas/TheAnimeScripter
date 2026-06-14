import os
import torch
import logging

from src.model.download import resolveWeightPath
from src.model.registry import modelsMap
from src.infra.logAndPrint import logAndPrint

from ._shared import calculatePadding, checker


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
        )
        self.nextFrame = torch.zeros(
            (
                1,
                3,
                self.padding[3] + self.height + self.padding[2],
                self.padding[1] + self.width + self.padding[0],
            ),
            device=checker.device,
            dtype=torch.float16 if self.half else torch.float32,
        )

        self.dummyOutput = self.prevFrame.new_zeros(
            1, 3, self.height * 4, self.width * 4
        )

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
                    frame.to(dtype=frame.dtype),
                    non_blocking=False,
                )
                if nextFrame is None:
                    self.nextFrame.copy_(
                        frame.to(dtype=frame.dtype),
                        non_blocking=False,
                    )
                else:
                    self.nextFrame.copy_(
                        nextFrame.to(dtype=frame.dtype),
                        non_blocking=False,
                    )
            self.normStream.synchronize()

            self.firstRun = False
        else:
            with torch.cuda.stream(self.normStream):
                if nextFrame is None:
                    self.nextFrame.copy_(
                        frame.to(dtype=frame.dtype),
                        non_blocking=False,
                    )
                else:
                    self.nextFrame.copy_(
                        nextFrame.to(dtype=frame.dtype),
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
        from src.model.trtHandler import (
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
