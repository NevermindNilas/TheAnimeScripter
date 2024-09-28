import os
import torch
import logging
import torch.nn.functional as F
import math

from .downloadModels import downloadModels, weightsDir, modelsMap
from .coloredPrints import yellow

torch.set_float32_matmul_precision("medium")


def importRifeArch(interpolateMethod, version):
    match version:
        case "v1":
            match interpolateMethod:
                case "rife" | "rife4.25":
                    from .rifearches.IFNet_rife425 import IFNet
                case "rife4.22-lite":
                    from .rifearches.IFNET_rife422lite import IFNet
                case "rife4.22":
                    from .rifearches.IFNET_rife422 import IFNet
                case "rife4.21":
                    from .rifearches.IFNet_rife421 import IFNet
                case "rife4.20":
                    from .rifearches.IFNet_rife420 import IFNet
                case "rife4.18":
                    from .rifearches.IFNet_rife418 import IFNet
                case "rife4.17":
                    from .rifearches.IFNet_rife417 import IFNet
                case "rife4.15-lite":
                    from .rifearches.IFNet_rife415lite import IFNet
                case "rife4.16-lite":
                    from .rifearches.IFNet_rife416lite import IFNet
                case "rife4.6":
                    from .rifearches.IFNet_rife46 import IFNet
            return IFNet

        case "v2":
            match interpolateMethod:
                case "rife4.25-tensorrt":
                    from src.rifearches.Rife425_v2 import IFNet, Head
                case "rife4.22-tensorrt":
                    from src.rifearches.Rife422_v2 import IFNet, Head
                case "rife4.22-lite-tensorrt":
                    from src.rifearches.Rife422_lite_v2 import IFNet, Head
                case "rife4.21-tensorrt":
                    from src.rifearches.Rife422_v2 import IFNet, Head
                case "rife4.20-tensorrt":
                    from src.rifearches.Rife420_v2 import IFNet, Head
                case "rife4.18-tensorrt":
                    from src.rifearches.Rife415_v2 import IFNet, Head
                case "rife4.17-tensorrt":
                    from src.rifearches.Rife415_v2 import IFNet, Head
                case "rife4.15-tensorrt":
                    from src.rifearches.Rife415_v2 import IFNet, Head
                case "rife4.6-tensorrt":
                    from src.rifearches.Rife46_v2 import IFNet

                    Head = None

            return IFNet, Head


class RifeCuda:
    def __init__(
        self,
        half,
        width,
        height,
        interpolateMethod,
        ensemble=False,
        interpolateFactor=2,
        inputFPS=30,
        interpolateSkip=False,
    ):
        """
        Initialize the RIFE model

        Args:
            half (bool): Whether to use half precision.
            width (int): The width of the input frame.
            height (int): The height of the input frame.
            UHD (bool): Whether to use UHD mode.
            interpolateMethod (str): The method to use for interpolation.
            ensemble (bool): Whether to use ensemble mode.
            nt (int): The number of streams to use, not available for now.
            interpolateFactor (int): The interpolation factor.
        """
        self.half = half
        self.scale = 1.0
        self.width = width
        self.height = height
        self.interpolateMethod = interpolateMethod
        self.ensemble = ensemble
        self.interpolateFactor = interpolateFactor
        self.inputFPS = inputFPS
        self.interpolateSkip = interpolateSkip

        if self.width > 1920 and self.height > 1080:
            self.scale = 0.5
            if self.half:
                print(
                    yellow(
                        "UHD and fp16 are not compatible with RIFE, defaulting to fp32"
                    )
                )
                logging.info(
                    "UHD and fp16 for rife are not compatible due to flickering issues, defaulting to fp32"
                )
                self.half = False

        self.handle_model()

    def handle_model(self):
        """
        Load the desired model
        """

        self.filename = modelsMap(self.interpolateMethod)
        if not os.path.exists(os.path.join(weightsDir, "rife", self.filename)):
            modelPath = downloadModels(model=self.interpolateMethod)
        else:
            modelPath = os.path.join(weightsDir, "rife", self.filename)

        IFNet = importRifeArch(self.interpolateMethod, "v1")

        self.model = IFNet(self.ensemble, self.scale, self.interpolateFactor)
        self.isCudaAvailable = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.isCudaAvailable else "cpu")
        self.dType = torch.float16 if self.half else torch.float32

        torch.set_grad_enabled(False)
        if self.isCudaAvailable:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            if self.half:
                torch.set_default_dtype(torch.float16)

        if self.isCudaAvailable and self.half:
            self.model.half()
        else:
            self.half = False
            self.model.float()

        self.model.load_state_dict(torch.load(modelPath, map_location=self.device))
        self.model.eval().cuda() if self.isCudaAvailable else self.model.eval()
        self.model.to(self.device).to(memory_format=torch.channels_last)

        ph = ((self.height - 1) // 64 + 1) * 64
        pw = ((self.width - 1) // 64 + 1) * 64
        self.padding = (0, pw - self.width, 0, ph - self.height)

        self.I0 = torch.zeros(
            1,
            3,
            self.height + self.padding[3],
            self.width + self.padding[1],
            dtype=self.dType,
            device=self.device,
        ).to(memory_format=torch.channels_last)

        self.I1 = torch.zeros(
            1,
            3,
            self.height + self.padding[3],
            self.width + self.padding[1],
            dtype=self.dType,
            device=self.device,
        ).to(memory_format=torch.channels_last)

        self.firstRun = True
        self.stream = torch.cuda.Stream()
        self.normStream = torch.cuda.Stream()

    @torch.inference_mode()
    def cacheFrameReset(self, frame):
        self.processFrame(frame, "cache")
        self.processFrame(self.I0, "model")

    @torch.inference_mode()
    def processFrame(self, frame, toNorm):
        match toNorm:
            case "I0":
                with torch.cuda.stream(self.normStream):
                    self.I0.copy_(
                        self.padFrame(
                            frame.to(
                                device=self.device,
                                dtype=self.dType,
                                non_blocking=True,
                            )
                            .permute(2, 0, 1)
                            .unsqueeze(0)
                            .mul(1 / 255),
                        )
                    ).to(memory_format=torch.channels_last)
                    self.normStream.synchronize()

            case "I1":
                with torch.cuda.stream(self.normStream):
                    self.I1.copy_(
                        self.padFrame(
                            frame.to(
                                device=self.device,
                                dtype=self.dType,
                                non_blocking=True,
                            )
                            .permute(2, 0, 1)
                            .unsqueeze(0)
                            .mul(1 / 255),
                        ),
                        non_blocking=True,
                    ).to(memory_format=torch.channels_last)
                    self.normStream.synchronize()

            case "cache":
                with torch.cuda.stream(self.normStream):
                    self.I0.copy_(
                        self.I1,
                        non_blocking=True,
                    )
                    self.model.cache()
                    self.normStream.synchronize()

            case "pad":
                with torch.cuda.stream(self.normStream):
                    output = self.padFrame(frame)
                    self.normStream.synchronize()

                return output

            case "infer":
                with torch.cuda.stream(self.stream):
                    output = self.model(self.I0, self.I1, frame)[
                        : self.height, : self.width, :
                    ]
                    self.stream.synchronize()
                return output

            case "model":
                self.model.cacheReset(frame)

    @torch.inference_mode()
    def padFrame(self, frame):
        return (
            F.pad(frame, [0, self.padding[1], 0, self.padding[3]])
            if self.padding != (0, 0, 0, 0)
            else frame
        )

    @torch.inference_mode()
    def __call__(self, frame, interpQueue):
        if self.firstRun:
            self.processFrame(frame, "I0")
            self.firstRun = False
            return

        self.processFrame(frame, "I1")

        for i in range(self.interpolateFactor - 1):
            timestep = torch.full(
                (1, 1, self.height + self.padding[3], self.width + self.padding[1]),
                (i + 1) * 1 / self.interpolateFactor,
                dtype=self.dType,
                device=self.device,
            )
            output = self.processFrame(timestep, "infer")
            interpQueue.put(output)

        self.processFrame(None, "cache")


class RifeTensorRT:
    def __init__(
        self,
        interpolateMethod: str = "rife4.25-tensorrt",
        interpolateFactor: int = 2,
        width: int = 0,
        height: int = 0,
        half: bool = True,
        ensemble: bool = False,
        interpolateSkip: bool | None = None,
    ):
        """
        Interpolates frames using TensorRT

        Args:
            interpolateMethod (str, optional): Interpolation method. Defaults to "rife415".
            interpolateFactor (int, optional): Interpolation factor. Defaults to 2.
            width (int, optional): Width of the frame. Defaults to 0.
            height (int, optional): Height of the frame. Defaults to 0.
            half (bool, optional): Half resolution. Defaults to True.
            ensemble (bool, optional): Ensemble. Defaults to False.
            nt (int, optional): Number of threads. Defaults to 1.
        """
        import tensorrt as trt
        from .utils.trtHandler import (
            TensorRTEngineCreator,
            TensorRTEngineLoader,
            TensorRTEngineNameHandler,
        )

        self.TensorRTEngineCreator = TensorRTEngineCreator
        self.TensorRTEngineLoader = TensorRTEngineLoader
        self.TensorRTEngineNameHandler = TensorRTEngineNameHandler
        self.trt = trt

        self.interpolateMethod = interpolateMethod
        self.interpolateFactor = interpolateFactor
        self.width = width
        self.height = height
        self.half = half
        self.ensemble = ensemble
        self.model = None
        self.interpolateSkip = interpolateSkip
        if self.width > 1920 and self.height > 1080:
            if self.half:
                print(
                    yellow(
                        "UHD and fp16 are not compatible with RIFE, defaulting to fp32"
                    )
                )
                logging.info(
                    "UHD and fp16 for rife are not compatible due to flickering issues, defaulting to fp32"
                )
                self.half = False
                self.scale = 0.5
        else:
            self.scale = 1.0
        self.handleModel()

    def handleModel(self):
        self.device = torch.device("cuda")
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        if self.half:
            torch.set_default_dtype(torch.float16)
        self.filename = modelsMap(
            self.interpolateMethod.replace("-tensorrt", ""),
            modelType="pth",
            half=self.half,
            ensemble=self.ensemble,
        )

        folderName = self.interpolateMethod.replace("-tensorrt", "")
        if not os.path.exists(os.path.join(weightsDir, folderName, self.filename)):
            self.modelPath = downloadModels(
                model=self.interpolateMethod.replace("-tensorrt", ""),
                modelType="pth",
                half=self.half,
                ensemble=self.ensemble,
            )
        else:
            self.modelPath = os.path.join(weightsDir, folderName, self.filename)

        self.dtype = torch.float16 if self.half else torch.float32
        tmp = max(64, int(64 / 1.0))
        self.pw = math.ceil(self.width / tmp) * tmp
        self.ph = math.ceil(self.height / tmp) * tmp
        self.padding = (0, self.pw - self.width, 0, self.ph - self.height)

        IFNet, Head = importRifeArch(self.interpolateMethod, "v2")
        self.norm = Head().cuda() if Head is not None else None

        enginePath = self.TensorRTEngineNameHandler(
            modelPath=self.modelPath,
            fp16=self.half,
            optInputShape=[1, 3, self.height, self.width],
            ensemble=self.ensemble,
        )

        self.model = IFNet(
            scale=self.scale,
            ensemble=self.ensemble,
            dtype=self.dtype,
            device=self.device,
            width=self.width,
            height=self.height,
        )

        self.model.to(self.device)
        if self.half:
            self.model.half()
        else:
            self.model.float()
        self.model.load_state_dict(torch.load(self.modelPath, map_location="cpu"))

        if self.interpolateMethod in ["rife4.25-tensorrt", "rife4.22-lite-tensorrt"]:
            channels = 4
        else:
            channels = 8

        self.engine, self.context = self.TensorRTEngineLoader(enginePath)
        if (
            self.engine is None
            or self.context is None
            or not os.path.exists(enginePath)
        ):
            dummyInput1 = torch.zeros(
                1, 3, self.ph, self.pw, dtype=self.dtype, device=self.device
            )
            dummyInput2 = torch.zeros(
                1, 3, self.ph, self.pw, dtype=self.dtype, device=self.device
            )
            dummyInput3 = torch.zeros(
                1, 1, self.ph, self.pw, dtype=self.dtype, device=self.device
            )

            if self.norm is not None:
                dummyInput4 = torch.zeros(
                    1,
                    channels,
                    self.ph,
                    self.pw,
                    dtype=self.dtype,
                    device=self.device,
                )

            self.modelPath = self.modelPath.replace(".pth", ".onnx")

            inputList = [dummyInput1, dummyInput2, dummyInput3]
            inputNames = ["img0", "img1", "timestep"]
            outputNames = ["output"]
            dynamicAxes = {
                "img0": {2: "height", 3: "width"},
                "img1": {2: "height", 3: "width"},
                "timestep": {2: "height", 3: "width"},
                "output": {1: "height", 2: "width"},
            }

            if self.norm is not None:
                inputList.append(dummyInput4)
                inputNames.append("f0")
                outputNames.append("f1")
                dynamicAxes["f0"] = {2: "height", 3: "width"}

            torch.onnx.export(
                self.model,
                tuple(inputList),
                self.modelPath,
                input_names=inputNames,
                output_names=outputNames,
                dynamic_axes=dynamicAxes,
                opset_version=19,
            )

            inputs = [
                [1, 3, self.ph, self.pw],
                [1, 3, self.ph, self.pw],
                [1, 1, self.ph, self.pw],
            ]

            if self.norm is not None:
                inputs.append([1, channels, self.ph, self.pw])

            inputsMin = inputsOpt = inputsMax = inputs

            logging.info("Loading engine failed, creating a new one")
            self.engine, self.context = self.TensorRTEngineCreator(
                modelPath=self.modelPath,
                enginePath=enginePath,
                fp16=self.half,
                inputsMin=inputsMin,
                inputsOpt=inputsOpt,
                inputsMax=inputsMax,
                inputName=inputNames,
                isMultiInput=True,
            )

            try:
                os.remove(self.modelPath)
            except Exception as e:
                logging.error(f"Error removing onnx model: {e}")

        self.dtype = torch.float16 if self.half else torch.float32
        self.stream = torch.cuda.Stream()
        self.I0 = torch.zeros(
            1,
            3,
            self.ph,
            self.pw,
            dtype=self.dtype,
            device=self.device,
        )

        self.I1 = torch.zeros(
            1,
            3,
            self.ph,
            self.pw,
            dtype=self.dtype,
            device=self.device,
        )

        if self.norm is not None:
            self.f0 = torch.zeros(
                1,
                channels,
                self.ph,
                self.pw,
                dtype=self.dtype,
                device=self.device,
            )

            self.f1 = torch.zeros(
                1,
                channels,
                self.ph,
                self.pw,
                dtype=self.dtype,
                device=self.device,
            )

        self.dummyTimeStep = torch.zeros(
            1, 1, self.ph, self.pw, dtype=self.dtype, device=self.device
        )

        self.dummyTimeSteps = [
            torch.full(
                (1, 1, self.ph, self.pw),
                (i + 1) * 1 / self.interpolateFactor,
                dtype=self.dtype,
                device=self.device,
            )
            for i in range(self.interpolateFactor - 1)
        ]

        self.dummyOutput = torch.zeros(
            (self.height, self.width, 3),
            device=self.device,
            dtype=self.dtype,
        )

        self.tensors = [
            self.I0,
            self.I1,
            self.dummyTimeStep,
        ]

        if self.norm is not None:
            self.tensors.extend([self.f0])

        self.tensors.extend([self.dummyOutput])

        if self.norm is not None:
            self.tensors.extend([self.f1])

        self.bindings = [tensor.data_ptr() for tensor in self.tensors]

        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            self.context.set_tensor_address(tensor_name, self.bindings[i])

            if self.engine.get_tensor_mode(tensor_name) == self.trt.TensorIOMode.INPUT:
                self.context.set_input_shape(tensor_name, self.tensors[i].shape)

        self.firstRun = True
        self.normalizationStream = torch.cuda.Stream()

        if self.interpolateSkip is not None:
            self.skippedCounter = 0

    @torch.inference_mode()
    def processFrame(self, frame, name=None):
        with torch.cuda.stream(self.normalizationStream):
            match name:
                case "I0":
                    self.I0.copy_(
                        F.pad(
                            frame.to(dtype=self.dtype, non_blocking=True)
                            .mul(1 / 255.0)
                            .permute(2, 0, 1)
                            .unsqueeze(0),
                            self.padding,
                        ),
                        non_blocking=True,
                    )
                case "I1":
                    self.I1.copy_(
                        F.pad(
                            frame.to(dtype=self.dtype, non_blocking=True)
                            .mul(1 / 255.0)
                            .permute(2, 0, 1)
                            .unsqueeze(0),
                            self.padding,
                        ),
                        non_blocking=True,
                    )
                case "f0":
                    self.f0.copy_(
                        self.norm(
                            F.pad(
                                frame.to(dtype=self.dtype, non_blocking=True)
                                .mul(1 / 255.0)
                                .permute(2, 0, 1)
                                .unsqueeze(0),
                                self.padding,
                            )
                        ),
                        non_blocking=True,
                    )
                case "f0-copy":
                    self.f0.copy_(self.f1, non_blocking=True)

                case "cache":
                    self.I0.copy_(self.I1, non_blocking=True)
                case "timestep":
                    self.dummyTimeStep.copy_(frame, non_blocking=True)

            self.normalizationStream.synchronize()

    @torch.inference_mode()
    def cacheFrameReset(self, frame):
        self.I0.copy_(self.processFrame(frame, "I0"), non_blocking=True)
        if self.norm is not None:
            self.f0.copy_(self.norm(self.processFrame(frame)), non_blocking=True)
        if self.interpolateSkip is not None:
            self.interpolateSkip.reset()

    @torch.inference_mode()
    def __call__(self, frame, interpQueue):
        if self.firstRun:
            if self.norm is not None:
                self.processFrame(frame, "f0")

            self.processFrame(frame, "I0")
            if self.interpolateSkip is not None:
                self.interpolateSkip(frame)

            self.firstRun = False
            return

        self.processFrame(frame, "I1")
        for timestep in self.dummyTimeSteps:
            self.processFrame(timestep, "timestep")
            self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)

            self.stream.synchronize()
            interpQueue.put(self.dummyOutput)

        self.processFrame(None, "cache")
        if self.norm is not None:
            self.processFrame(None, "f0-copy")

    def getSkippedCounter(self):
        return self.skippedCounter


class RifeNCNN:
    def __init__(
        self,
        interpolateMethod,
        ensemble=False,
        width=1920,
        height=1080,
        half=True,
        interpolateFactor=2,
    ):
        self.interpolateMethod = interpolateMethod
        self.height = height
        self.width = width
        self.ensemble = ensemble
        self.half = half
        self.interpolateFactor = interpolateFactor

        UHD = True if width >= 3840 or height >= 2160 else False
        scale = 2 if UHD else 1

        from rife_ncnn_vulkan_python import Rife

        match interpolateMethod:
            case "rife4.6-ncnn":
                self.interpolateMethod = "rife-v4.6-ncnn"
            case "rife4.15-lite-ncnn":
                self.interpolateMethod = "rife-v4.15-lite-ncnn"
            case "rife4.16-lite-ncnn":
                self.interpolateMethod = "rife-v4.16-lite-ncnn"
            case "rife4.17-ncnn":
                self.interpolateMethod = "rife-v4.17-ncnn"
            case "rife4.18-ncnn":
                self.interpolateMethod = "rife-v4.18-ncnn"

        self.filename = modelsMap(
            self.interpolateMethod,
            ensemble=self.ensemble,
            modelType="ncnn",
        )

        if self.filename.endswith("-ncnn.zip"):
            self.filename = self.filename[:-9]
        elif self.filename.endswith("-ncnn"):
            self.filename = self.filename[:-5]

        if not os.path.exists(
            os.path.join(weightsDir, self.interpolateMethod, self.filename)
        ):
            modelPath = downloadModels(
                model=self.interpolateMethod,
                ensemble=self.ensemble,
                modelType="ncnn",
            )
        else:
            modelPath = os.path.join(weightsDir, self.interpolateMethod, self.filename)

        if modelPath.endswith("-ncnn.zip"):
            modelPath = modelPath[:-9]
        elif modelPath.endswith("-ncnn"):
            modelPath = modelPath[:-5]

        self.rife = Rife(
            gpuid=0,
            model=modelPath,
            scale=scale,
            tta_mode=False,
            tta_temporal_mode=False,
            uhd_mode=UHD,
        )

        self.frame1 = None
        self.shape = (self.height, self.width)

    def cacheFrame(self):
        self.frame1 = self.frame2

    def cacheFrameReset(self, frame):
        self.frame1 = frame.cpu().numpy().astype("uint8")

    def __call__(self, frame, interpQueue):
        if self.frame1 is None:
            self.frame1 = frame.cpu().numpy().astype("uint8")
            return False

        self.frame2 = frame.cpu().numpy().astype("uint8")

        for i in range(self.interpolateFactor - 1):
            timestep = (i + 1) * 1 / self.interpolateFactor
            output = self.rife.process_cv2(self.frame1, self.frame2, timestep=timestep)
            output = torch.from_numpy(output).to(frame.device)
            interpQueue.put(output)

        self.cacheFrame()
