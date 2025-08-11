import os
import torch
import logging
import torch.nn.functional as F
import math
import numpy as np

from .utils.downloadModels import downloadModels, weightsDir, modelsMap
from .utils.isCudaInit import CudaChecker
from .utils.logAndPrint import logAndPrint

checker = CudaChecker()

torch.set_float32_matmul_precision("medium")


def importRifeArch(interpolateMethod, version):
    match version:
        case "v1":
            match interpolateMethod:
                case "rife4.25-heavy":
                    from .rifearches.IFNet_rife425heavy import IFNet
                case "rife4.25-lite":
                    from .rifearches.IFNet_rife425lite import IFNet
                case "rife" | "rife4.25":
                    from .rifearches.IFNet_rife425 import IFNet
                case "rife4.22-lite":
                    from .rifearches.IFNet_rife422lite import IFNet
                case "rife4.22":
                    from .rifearches.IFNet_rife422 import IFNet
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
                case "rife_elexor":
                    from .rifearches.IFNet_elexor_cuda import IFNet
            return IFNet

        case "v3":
            match interpolateMethod:
                case "rife4.25-heavy-tensorrt":
                    from src.rifearches.Rife425_heavy_v3 import IFNet

                    Head = True

                case "rife4.25-lite-tensorrt":
                    from src.rifearches.Rife425_lite_v3 import IFNet

                    Head = True
                case "rife4.25-tensorrt":
                    from src.rifearches.Rife425_v3 import IFNet

                    Head = True
                case "rife4.22-tensorrt":
                    from src.rifearches.Rife422_v3 import IFNet

                    Head = True
                case "rife4.22-lite-tensorrt":
                    from src.rifearches.Rife422_lite_v3 import IFNet

                    Head = True
                case "rife4.21-tensorrt":
                    from src.rifearches.Rife422_v3 import IFNet

                    Head = True
                case "rife4.20-tensorrt":
                    from src.rifearches.Rife420_v3 import IFNet

                    Head = True
                case "rife4.18-tensorrt":
                    from src.rifearches.Rife415_v3 import IFNet

                    Head = True
                case "rife4.17-tensorrt":
                    from src.rifearches.Rife415_v3 import IFNet

                    Head = True
                case "rife4.15-tensorrt":
                    from src.rifearches.Rife415_v3 import IFNet

                    Head = True
                case "rife4.6-tensorrt" | "rife4.6-directml":
                    from src.rifearches.Rife46_v3 import IFNet

                    Head = False
                case "rife_elexor-tensorrt":
                    from src.rifearches.IFNet_elexor_tensorrt import IFNet

                    Head = True
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
        dynamicScale=False,
        staticStep=False,
        compileMode: str = "default",
    ):
        """
        Initialize the RIFE model

        Args:
            half (bool): Half resolution
            width (int): Width of the frame
            height (int): Height of the frame
            interpolateMethod (str): Interpolation method
            ensemble (bool, optional): Ensemble. Defaults to False.
            interpolateFactor (int, optional): Interpolation factor. Defaults to 2.
            dynamicScale (bool, optional): Use Dynamic scale. Defaults to False.
            staticStep (bool, optional): Use static timestep. Defaults to False.
        """
        self.half = half
        self.scale = 1.0
        self.width = width
        self.height = height
        self.interpolateMethod = interpolateMethod
        self.ensemble = ensemble
        self.interpolateFactor = interpolateFactor
        self.dynamicScale = dynamicScale
        self.staticStep = staticStep
        self.compileMode = compileMode

        if self.width > 1920 and self.height > 1080:
            self.scale = 0.5
            if self.half:
                logAndPrint(
                    "UHD and fp16 are not compatible with RIFE, defaulting to fp32",
                    "yellow",
                )
                self.half = False

        self.handleModel()

    def handleModel(self):
        """
        Load the desired model
        """

        self.filename = modelsMap(self.interpolateMethod)
        if not os.path.exists(os.path.join(weightsDir, "rife", self.filename)):
            modelPath = downloadModels(model=self.interpolateMethod)
        else:
            modelPath = os.path.join(weightsDir, "rife", self.filename)

        self.dType = torch.float16 if self.half else torch.float32

        IFNet = importRifeArch(self.interpolateMethod, "v1")
        if self.interpolateMethod in ["rife_elexor"] and self.staticStep:
            self.staticStep = False
            logAndPrint(
                "Static step is not supported for rife_elexor, automatically disabling it",
                "yellow",
            )
        if (
            self.interpolateMethod not in ["rife4.6", "rife4.15", "rife4.15-lite"]
            and self.staticStep
        ):
            self.staticStep = False
            logAndPrint(
                "Static step is not supported for this interpolation model yet, automatically disabling it",
                "yellow",
            )
        if self.interpolateMethod in ["rife_elexor"]:
            self.model = IFNet(
                self.scale,
                self.ensemble,
                self.dType,
                checker.device,
                self.width,
                self.height,
                self.interpolateFactor,
            )
        else:
            if self.interpolateMethod in ["rife4.6", "rife4.15", "rife4.15-lite"]:
                self.model = IFNet(
                    self.ensemble,
                    self.dynamicScale,
                    self.scale,
                    self.interpolateFactor,
                    self.staticStep,
                )
            else:
                self.model = IFNet(
                    self.ensemble,
                    self.dynamicScale,
                    self.scale,
                    self.interpolateFactor,
                )

        if checker.cudaAvailable and self.half:
            self.model.half()
        else:
            self.half = False
            self.model.float()

        self.model.load_state_dict(torch.load(modelPath, map_location=checker.device))
        self.model.eval().cuda() if checker.cudaAvailable else self.model.eval()
        self.model = self.model.to(checker.device)
        self.model = self.model.to(memory_format=torch.channels_last)

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
                    f"Error compiling model {self.interpolateMethod} with mode {self.compileMode}: {e}"
                )
                logAndPrint(
                    f"Error compiling model {self.interpolateMethod} with mode {self.compileMode}: {e}",
                    "red",
                )

            self.compileMode = "default"

        if self.interpolateMethod in ["rife4.25", "rife4.25-heavy", "rife4.25-lite"]:
            ph = ((self.height - 1) // 128 + 1) * 128
            pw = ((self.width - 1) // 128 + 1) * 128
        else:
            ph = ((self.height - 1) // 64 + 1) * 64
            pw = ((self.width - 1) // 64 + 1) * 64
        self.padding = (0, pw - self.width, 0, ph - self.height)

        self.I0 = torch.zeros(
            1,
            3,
            self.height + self.padding[3],
            self.width + self.padding[1],
            dtype=self.dType,
            device=checker.device,
        ).to(memory_format=torch.channels_last)

        self.I1 = torch.zeros(
            1,
            3,
            self.height + self.padding[3],
            self.width + self.padding[1],
            dtype=self.dType,
            device=checker.device,
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
                    frame = frame.to(
                        device=checker.device,
                        dtype=self.dType,
                        non_blocking=False,
                    )
                    frame = self.padFrame(frame)
                    self.I0.copy_(
                        frame,
                        non_blocking=False,
                    ).to(memory_format=torch.channels_last)
                self.normStream.synchronize()

            case "I1":
                with torch.cuda.stream(self.normStream):
                    frame = frame.to(
                        device=checker.device,
                        dtype=self.dType,
                        non_blocking=False,
                    )
                    frame = self.padFrame(frame)
                    self.I1.copy_(
                        frame,
                        non_blocking=False,
                    ).to(memory_format=torch.channels_last)
                self.normStream.synchronize()
            case "cache":
                with torch.cuda.stream(self.normStream):
                    self.I0.copy_(
                        self.I1,
                        non_blocking=False,
                    )
                    self.model.cache()
                self.normStream.synchronize()
            case "infer":
                with torch.cuda.stream(self.normStream):
                    if self.staticStep:
                        output = self.model(self.I0, self.I1, frame).clone()
                    else:
                        output = self.model(self.I0, self.I1, frame)[
                            :, :, : self.height, : self.width
                        ].clone()
                self.normStream.synchronize()
                return output

            case "model":
                with torch.cuda.stream(self.normStream):
                    self.model.cacheReset(frame)
                self.normStream.synchronize()

    @torch.inference_mode()
    def padFrame(self, frame):
        return (
            F.pad(frame, [0, self.padding[1], 0, self.padding[3]])
            if self.padding != (0, 0, 0, 0)
            else frame
        )

    @torch.inference_mode()
    def __call__(self, frame, interpQueue, framesToInsert: int = 2, timesteps=None):
        if self.firstRun:
            self.processFrame(frame, "I0")
            self.firstRun = False
            return
        self.processFrame(frame, "I1")

        for i in range(framesToInsert):
            if timesteps is not None and i < len(timesteps):
                t = timesteps[i]
            else:
                t = (i + 1) * 1 / (framesToInsert + 1)

            timestep = torch.full(
                (1, 1, self.height + self.padding[3], self.width + self.padding[1]),
                t,
                dtype=self.dType,
                device=checker.device,
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
        """
        import tensorrt as trt
        from .utils.trtHandler import (
            tensorRTEngineCreator,
            tensorRTEngineLoader,
            tensorRTEngineNameHandler,
        )

        self.tensorRTEngineCreator = tensorRTEngineCreator
        self.tensorRTEngineLoader = tensorRTEngineLoader
        self.tensorRTEngineNameHandler = tensorRTEngineNameHandler
        self.trt = trt

        self.interpolateMethod = interpolateMethod
        self.interpolateFactor = interpolateFactor
        self.width = width
        self.height = height
        self.half = half
        self.ensemble = ensemble
        self.model = None
        if self.width > 1920 and self.height > 1080:
            if self.half:
                logAndPrint(
                    "UHD and fp16 are not compatible with RIFE, defaulting to fp32",
                    "yellow",
                )
                logging.info(
                    "UHD and fp16 for rife are not compatible due to flickering issues, defaulting to fp32"
                )
                self.half = False
                # self.scale = 1.0
        else:
            pass
            # self.scale = 1.0
        self.scale = 1.0
        self.handleModel()

    def handleModel(self):
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

        if self.interpolateMethod in [
            "rife_elexor-tensorrt",
            "rife4.25-tensorrt",
            "rife4.25-heavy-tensorrt",
        ]:
            channels = 4
            mul = 64
        elif self.interpolateMethod in [
            "rife4.22-lite-tensorrt",
        ]:
            channels = 4
            mul = 32
        elif self.interpolateMethod in [
            "rife4.25-lite-tensorrt",
        ]:
            channels = 4
            mul = 128
        else:
            channels = 8
            mul = 32

        self.dtype = torch.float16 if self.half else torch.float32
        tmp = max(mul, int(mul / 1.0))
        self.pw = math.ceil(self.width / tmp) * tmp
        self.ph = math.ceil(self.height / tmp) * tmp
        self.padding = (0, self.pw - self.width, 0, self.ph - self.height)

        IFNet, Head = importRifeArch(self.interpolateMethod, "v3")

        enginePath = self.tensorRTEngineNameHandler(
            modelPath=self.modelPath,
            fp16=self.half,
            optInputShape=[1, 3, self.height, self.width],
            ensemble=self.ensemble,
            isRife=True,
        )

        self.model = IFNet(
            scale=self.scale,
            ensemble=self.ensemble,
            dtype=self.dtype,
            device=checker.device,
            width=self.width,
            height=self.height,
        )

        self.model.to(checker.device)
        if self.half:
            self.model.half()
        else:
            self.model.float()
        self.model.load_state_dict(torch.load(self.modelPath, map_location="cpu"))

        if Head is True:
            self.norm = self.model.encode
        else:
            self.norm = None

        self.engine, self.context = self.tensorRTEngineLoader(enginePath)
        if (
            self.engine is None
            or self.context is None
            or not os.path.exists(enginePath)
        ):
            dummyInput1 = torch.zeros(
                1, 3, self.ph, self.pw, dtype=self.dtype, device=checker.device
            )
            dummyInput2 = torch.zeros(
                1, 3, self.ph, self.pw, dtype=self.dtype, device=checker.device
            )
            dummyInput3 = torch.full(
                (1, 1, self.ph, self.pw),
                0.5,
                dtype=self.dtype,
                device=checker.device,
            )

            if self.norm is not None:
                dummyInput4 = torch.zeros(
                    1,
                    channels,
                    self.ph,
                    self.pw,
                    dtype=self.dtype,
                    device=checker.device,
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
                opset_version=20,
                optimize=True,
            )

            inputs = [
                [1, 3, self.ph, self.pw],
                [1, 3, self.ph, self.pw],
                [1, 1, self.ph, self.pw],
            ]

            if self.norm is not None:
                inputs.append([1, channels, self.ph, self.pw])

            if hasattr(self, "model") and self.model is not None:
                del self.model
                import gc

                del (
                    dummyInput1,
                    dummyInput2,
                    dummyInput3,
                )  # No need to keep these in memory

                if self.norm is not None:
                    del dummyInput4
                gc.collect()
                torch.cuda.empty_cache()

            inputsMin = inputsOpt = inputsMax = inputs

            logging.info("Loading engine failed, creating a new one")
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
            device=checker.device,
        )

        self.I1 = torch.zeros(
            1,
            3,
            self.ph,
            self.pw,
            dtype=self.dtype,
            device=checker.device,
        )

        if self.norm is not None:
            self.f0 = torch.zeros(
                1,
                channels,
                self.ph,
                self.pw,
                dtype=self.dtype,
                device=checker.device,
            )

            self.f1 = torch.zeros(
                1,
                channels,
                self.ph,
                self.pw,
                dtype=self.dtype,
                device=checker.device,
            )

        self.dummyTimeStep = torch.full(
            (1, 1, self.ph, self.pw),
            0.5,
            dtype=self.dtype,
            device=checker.device,
        )

        self.dummyOutput = torch.zeros(
            (1, 3, self.height, self.width),
            device=checker.device,
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
    def processFrame(self, frame, name=None):
        with torch.cuda.stream(self.normStream):
            match name:
                case "I0":
                    self.I0.copy_(
                        F.pad(
                            frame.to(dtype=self.dtype),
                            self.padding,
                        ),
                        non_blocking=True,
                    )

                case "I1":
                    self.I1.copy_(
                        F.pad(
                            frame.to(dtype=self.dtype),
                            self.padding,
                        ),
                        non_blocking=True,
                    )

                case "f0":
                    self.f0.copy_(
                        self.norm(
                            F.pad(
                                frame.to(dtype=self.dtype),
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

        self.normStream.synchronize()

    @torch.inference_mode()
    def cacheFrameReset(self, frame):
        self.processFrame(frame, "I0")
        if self.norm is not None:
            self.processFrame(frame, "f0")

    @torch.inference_mode()
    def __call__(self, frame, interpQueue, framesToInsert=1, timesteps=None):
        if self.firstRun:
            if self.norm is not None:
                self.processFrame(frame, "f0")

            self.processFrame(frame, "I0")

            self.firstRun = False
            return

        self.processFrame(frame, "I1")
        for i in range(framesToInsert):
            if timesteps is not None and i < len(timesteps):
                t = timesteps[i]
            else:
                t = (i + 1) * 1 / (framesToInsert + 1)

            timestep = torch.full(
                (1, 1, self.ph, self.pw),
                t,
                dtype=self.dtype,
                device=checker.device,
            )

            self.processFrame(timestep, "timestep")
            with torch.cuda.stream(self.stream):
                self.cudaGraph.replay()
            self.stream.synchronize()
            with torch.cuda.stream(self.outputStream):
                output = self.dummyOutput.clone().detach()
            self.outputStream.synchronize()
            interpQueue.put(output)

        self.processFrame(None, "cache")
        if self.norm is not None:
            self.processFrame(None, "f0-copy")


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
        # scale = 2 if UHD else 1
        from rife_ncnn_vulkan_python import wrapped

        self.wrapped = wrapped

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

        padding = 32  # ADD 64 once 4.25-ncnn is added

        self.Rife = self.wrapped.RifeWrapped(
            0,
            self.ensemble,
            False,
            UHD,
            2,
            False,
            True,
            padding,
        )

        modelDir = self.wrapped.StringType()
        modelDir.wstr = self.wrapped.new_wstr_p()
        self.wrapped.wstr_p_assign(modelDir.wstr, str(modelPath))
        self.Rife.load(modelDir)

        self.outputBytes = bytearray(width * height * 3)
        self.output = self.wrapped.Image(self.outputBytes, self.width, self.height, 3)
        self.frame0 = None
        self.shape = (self.height, self.width)

    def cacheFrame(self):
        self.frame0Bytes = self.frame1Bytes
        self.frame0 = self.frame1

    def cacheFrameReset(self, frame):
        self.frame0 = frame.cpu().numpy().astype("uint8")
        self.frame0Bytes = bytearray(self.frame0.tobytes())
        self.frame0 = self.wrapped.Image(self.frame0Bytes, self.width, self.height, 3)

    def __call__(self, frame, interpQueue, framesToInsert=1, timesteps=None):
        if self.frame0 is None:
            self.frame0 = (
                frame.mul(255).squeeze(0).permute(1, 2, 0).cpu().numpy().astype("uint8")
            )
            self.frame0Bytes = bytearray(self.frame0.tobytes())
            self.frame0 = self.wrapped.Image(
                self.frame0Bytes, self.width, self.height, 3
            )

            return False

        self.frame1 = (
            frame.mul(255).squeeze(0).permute(1, 2, 0).cpu().numpy().astype("uint8")
        )
        self.frame1Bytes = bytearray(self.frame1.tobytes())
        self.frame1 = self.wrapped.Image(self.frame1Bytes, self.width, self.height, 3)

        for i in range(framesToInsert):
            if timesteps is not None and i < len(timesteps):
                t = timesteps[i]
            else:
                t = (i + 1) * 1 / (framesToInsert + 1)

            self.Rife.process(self.frame0, self.frame1, t, self.output)

            output = (
                torch.frombuffer(self.outputBytes, dtype=torch.uint8)
                .reshape(self.height, self.width, 3)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(dtype=torch.float32 if self.half else torch.float16)
                .mul(1 / 255.0)
            )
            interpQueue.put(output)

        self.cacheFrame()


class RifeDirectML:
    def __init__(
        self,
        interpolateMethod: str = "rife4.25-directml",
        interpolateFactor: int = 2,
        width: int = 0,
        height: int = 0,
        half: bool = True,
        ensemble: bool = False,
    ):
        """
        Interpolates frames using DirectML

        Arguments:
            - interpolateMethod (str, optional): Interpolation method. Defaults to "rife415".
            - interpolateFactor (int, optional): Interpolation factor. Defaults to 2.
            - width (int, optional): Width of the frame. Defaults to 0.
            - height (int, optional): Height of the frame. Defaults to 0.
            - half (bool, optional): Half resolution. Defaults to True.
            - ensemble (bool, optional): Ensemble. Defaults to False.
        """

        raise NotImplementedError(
            "DirectML is not supported yet, please use RIFE-NCNN instead."
        )

        import onnxruntime as ort

        self.ort = ort

        self.interpolateMethod = interpolateMethod
        self.interpolateFactor = interpolateFactor
        self.width = width
        self.height = height
        self.half = half
        self.ensemble = ensemble
        self.model = None

        if self.half:
            logging.info(
                "Half precision is not supported for DirectML, defaulting to fp32"
            )
            self.half = False

        if self.width > 1920 and self.height > 1080:
            if self.half:
                logAndPrint(
                    "UHD and fp16 are not compatible with RIFE, defaulting to fp32",
                    "yellow",
                )
                logging.info(
                    "UHD and fp16 for rife are not compatible due to flickering issues, defaulting to fp32"
                )
                self.half = False
                # self.scale = 1.0
        else:
            pass
            # self.scale = 1.0
        self.scale = 1.0

        self.handleModel()

    def handleModel(self):
        self.deviceType = "cpu"
        self.device = torch.device(self.deviceType)

        if self.half:
            self.numpyDType = np.float16
            self.torchDType = torch.float16
        else:
            self.numpyDType = np.float32
            self.torchDType = torch.float32

        if self.half:
            torch.set_default_dtype(torch.float16)
        self.filename = modelsMap(
            self.interpolateMethod.replace("-directml", ""),
            modelType="pth",
            half=self.half,
            ensemble=self.ensemble,
        )

        folderName = self.interpolateMethod.replace("-directml", "")
        if not os.path.exists(os.path.join(weightsDir, folderName, self.filename)):
            self.modelPath = downloadModels(
                model=self.interpolateMethod.replace("-directml", ""),
                modelType="pth",
                half=self.half,
                ensemble=self.ensemble,
            )
        else:
            self.modelPath = os.path.join(weightsDir, folderName, self.filename)

        if self.interpolateMethod in [
            "rife_elexor-directml",
            "rife4.25-directml",
        ]:
            channels = 4
            mul = 64
        elif self.interpolateMethod in [
            "rife4.22-lite-directml",
        ]:
            channels = 4
            mul = 32
        elif self.interpolateMethod in [
            "rife4.25-lite-directml",
        ]:
            channels = 4
            mul = 128
        else:
            channels = 8
            mul = 32

        self.dtype = torch.float16 if self.half else torch.float32
        tmp = max(mul, int(mul / 1.0))
        self.pw = math.ceil(self.width / tmp) * tmp
        self.ph = math.ceil(self.height / tmp) * tmp
        self.padding = (0, self.pw - self.width, 0, self.ph - self.height)

        IFNet, Head = importRifeArch(self.interpolateMethod, "v3")

        self.model = IFNet(
            scale=self.scale,
            ensemble=self.ensemble,
            dtype=self.dtype,
            device=self.device,
            width=self.width,
            height=self.height,
        )
        if self.half:
            self.model.half()
        else:
            self.model.float()
        self.model.load_state_dict(torch.load(self.modelPath, map_location="cpu"))

        if Head is True:
            self.norm = self.model.encode
        else:
            self.norm = None

        dummyInput1 = torch.zeros(
            1, 3, self.ph, self.pw, dtype=self.dtype, device=self.device
        )
        dummyInput2 = torch.zeros(
            1, 3, self.ph, self.pw, dtype=self.dtype, device=self.device
        )
        dummyInput3 = torch.full(
            (1, 1, self.ph, self.pw),
            0.5,
            dtype=self.dtype,
            device=self.device,
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
            opset_version=20,
        )
        inputs = [
            [1, 3, self.ph, self.pw],
            [1, 3, self.ph, self.pw],
            [1, 1, self.ph, self.pw],
        ]
        if self.norm is not None:
            inputs.append([1, channels, self.ph, self.pw])

        providers = self.ort.get_available_providers()

        if "DmlExecutionProvider" in providers:
            logging.info("DirectML provider available. Defaulting to DirectML")
            self.model = self.ort.InferenceSession(
                self.modelPath, providers=["DmlExecutionProvider"]
            )
        else:
            logging.info(
                "DirectML provider not available, falling back to CPU, expect significantly worse performance, ensure that your drivers are up to date and your GPU supports DirectX 12"
            )
            self.model = self.ort.InferenceSession(
                self.modelPath, providers=["CPUExecutionProvider"]
            )

        self.IoBinding = self.model.io_binding()
        self.I0 = torch.zeros(
            1,
            3,
            self.ph,
            self.pw,
            dtype=self.dtype,
            device=self.device,
        ).contiguous()

        self.I1 = torch.zeros(
            1,
            3,
            self.ph,
            self.pw,
            dtype=self.dtype,
            device=self.device,
        ).contiguous()

        if self.norm is not None:
            self.f0 = torch.zeros(
                1,
                channels,
                self.ph,
                self.pw,
                dtype=self.dtype,
                device=self.device,
            ).contiguous()

            self.f1 = torch.zeros(
                1,
                channels,
                self.ph,
                self.pw,
                dtype=self.dtype,
                device=self.device,
            ).contiguous()

        self.dummyTimeStep = torch.full(
            (1, 1, self.ph, self.pw),
            0.5,
            dtype=self.dtype,
            device=self.device,
        ).contiguous()

        self.dummyOutput = torch.zeros(
            (1, 3, self.height, self.width),
            device=self.device,
            dtype=self.dtype,
        ).contiguous()

        self.IoBinding.bind_output(
            name="output",
            device_type=self.deviceType,
            device_id=0,
            element_type=self.numpyDType,
            shape=self.dummyOutput.shape,
            buffer_ptr=self.dummyOutput.data_ptr(),
        )

        if self.norm is not None:
            self.IoBinding.bind_input(
                name="f0",
                device_type=self.deviceType,
                device_id=0,
                element_type=self.numpyDType,
                shape=self.f0.shape,
                buffer_ptr=self.f0.data_ptr(),
            )

        self.firstRun = True

    @torch.inference_mode()
    def cacheFrameReset(self, frame):
        self.processFrame(frame, "I0")
        if self.norm is not None:
            self.processFrame(frame, "f0")

    @torch.inference_mode()
    def processFrame(self, frame, name=None):
        match name:
            case "I0":
                self.I0.copy_(
                    F.pad(
                        frame.to(dtype=self.dtype),
                        self.padding,
                    ),
                    non_blocking=False,
                )

            case "I1":
                self.I1.copy_(
                    F.pad(
                        frame.to(dtype=self.dtype),
                        self.padding,
                    ),
                    non_blocking=False,
                )

            case "f0":
                self.f0.copy_(
                    self.norm(
                        F.pad(
                            frame.to(dtype=self.dtype),
                            self.padding,
                        )
                    ),
                    non_blocking=False,
                )

            case "f0-copy":
                self.f0.copy_(self.f1, non_blocking=False)

            case "cache":
                self.I0.copy_(self.I1, non_blocking=False)

            case "timestep":
                self.dummyTimeStep.copy_(frame, non_blocking=False)

    @torch.inference_mode()
    def __call__(
        self, frame: torch.Tensor, interpQueue, framesToInsert=1, timesteps=None
    ):
        if self.firstRun:
            if self.norm is not None:
                self.processFrame(frame, "f0")

            self.processFrame(frame, "I0")

            self.firstRun = False
            return

        self.processFrame(frame, "I1")
        for i in range(framesToInsert):
            if timesteps is not None and i < len(timesteps):
                t = timesteps[i]
            else:
                t = (i + 1) * 1 / (framesToInsert + 1)

            timestep = torch.full(
                (1, 1, self.ph, self.pw),
                t,
                dtype=self.dtype,
                device=self.device,
            )
            self.processFrame(timestep, "timestep")

            self.IoBinding.bind_input(
                name="img0",
                device_type=self.deviceType,
                device_id=0,
                element_type=self.numpyDType,
                shape=self.I0.shape,
                buffer_ptr=self.I0.data_ptr(),
            )

            self.IoBinding.bind_input(
                name="img1",
                device_type=self.deviceType,
                device_id=0,
                element_type=self.numpyDType,
                shape=self.I1.shape,
                buffer_ptr=self.I1.data_ptr(),
            )
            self.IoBinding.bind_input(
                name="timestep",
                device_type=self.deviceType,
                device_id=0,
                element_type=self.numpyDType,
                shape=self.dummyTimeStep.shape,
                buffer_ptr=timestep.data_ptr(),
            )

            self.model.run_with_iobinding(self.IoBinding)
            interpQueue.put(self.dummyOutput)

        return frame


class MultiPassDedup:
    def __init__(
        self,
        half,
        width,
        height,
        interpolateMethod,
        ensemble=False,
        interpolateFactor=2,
        dynamicScale=False,
        staticStep=False,
    ):
        """
        Initialize the RIFE model

        Args:
            half (bool): Half resolution
            width (int): Width of the frame
            height (int): Height of the frame
            interpolateMethod (str): Interpolation method
            ensemble (bool, optional): Ensemble. Defaults to False.
            interpolateFactor (int, optional): Interpolation factor. Defaults to 2.
            dynamicScale (bool, optional): Use Dynamic scale. Defaults to False.
            staticStep (bool, optional): Use static timestep. Defaults to False.
        """
        raise NotImplementedError(
            "MultiPassDedup is not implemented yet. Please use Rife."
        )
        self.half = half
        self.scale = 1.0
        self.width = width
        self.height = height
        self.interpolateMethod = interpolateMethod
        self.ensemble = ensemble
        self.interpolateFactor = interpolateFactor
        self.dynamicScale = dynamicScale
        self.staticStep = staticStep

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

        self.handleModel()

    def handleModel(self):
        """
        Load the desired model
        """
        from src.multipassdedup.utils.tools import TMapper

        self.TMAPPER = TMapper()

        self.filename = modelsMap(self.interpolateMethod)
        if not os.path.exists(os.path.join(weightsDir, "rife", self.filename)):
            modelPath = downloadModels(model=self.interpolateMethod)
        else:
            modelPath = os.path.join(weightsDir, "rife", self.filename)

        self.dType = torch.float16 if self.half else torch.float32

        IFNet = importRifeArch(self.interpolateMethod, "v1")
        if self.interpolateMethod in ["rife_elexor"] and self.staticStep:
            self.staticStep = False
            logAndPrint(
                "Static step is not supported for rife_elexor, automatically disabling it",
                "yellow",
            )
        if (
            self.interpolateMethod not in ["rife4.6", "rife4.15", "rife4.15-lite"]
            and self.staticStep
        ):
            self.staticStep = False
            logAndPrint(
                "Static step is not supported for this interpolation model yet, automatically disabling it",
                "yellow",
            )
        if self.interpolateMethod in ["rife_elexor"]:
            self.model = IFNet(
                self.scale,
                self.ensemble,
                self.dType,
                checker.device,
                self.width,
                self.height,
                self.interpolateFactor,
            )
        else:
            if self.interpolateMethod in ["rife4.6", "rife4.15", "rife4.15-lite"]:
                self.model = IFNet(
                    self.ensemble,
                    self.dynamicScale,
                    self.scale,
                    self.interpolateFactor,
                    self.staticStep,
                )
            else:
                self.model = IFNet(
                    self.ensemble,
                    self.dynamicScale,
                    self.scale,
                    self.interpolateFactor,
                )

        if checker.cudaAvailable and self.half:
            self.model.half()
        else:
            self.half = False
            self.model.float()

        self.model.load_state_dict(torch.load(modelPath, map_location=checker.device))
        self.model.eval().cuda() if checker.cudaAvailable else self.model.eval()
        self.model = self.model.to(checker.device)
        self.model = self.model.to(memory_format=torch.channels_last)

        if self.interpolateMethod in ["rife4.25", "rife4.25-heavy", "rife4.25-lite"]:
            ph = ((self.height - 1) // 128 + 1) * 128
            pw = ((self.width - 1) // 128 + 1) * 128
        else:
            ph = ((self.height - 1) // 64 + 1) * 64
            pw = ((self.width - 1) // 64 + 1) * 64
        self.padding = (0, pw - self.width, 0, ph - self.height)

        self.I0 = torch.zeros(
            1,
            3,
            self.height + self.padding[3],
            self.width + self.padding[1],
            dtype=self.dType,
            device=checker.device,
        ).to(memory_format=torch.channels_last)

        self.I1 = torch.zeros(
            1,
            3,
            self.height + self.padding[3],
            self.width + self.padding[1],
            dtype=self.dType,
            device=checker.device,
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
                self.I0.copy_(
                    self.padFrame(
                        frame.to(
                            device=checker.device,
                            dtype=self.dType,
                            non_blocking=True,
                        )
                    )
                ).to(memory_format=torch.channels_last)

            case "I1":
                self.I1.copy_(
                    self.padFrame(
                        frame.to(
                            device=checker.device,
                            dtype=self.dType,
                            non_blocking=True,
                        )
                    ),
                    non_blocking=True,
                ).to(memory_format=torch.channels_last)

            case "cache":
                self.I0.copy_(
                    self.I1,
                    non_blocking=True,
                )
                self.model.cache()

            case "pad":
                output = self.padFrame(frame)
                return output

            case "infer":
                with torch.cuda.stream(self.stream):
                    if self.staticStep:
                        output = self.model(self.I0, self.I1, frame)
                    else:
                        output = self.model(self.I0, self.I1, frame)[
                            :, :, : self.height, : self.width
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

    # A bit tired of understanding this,
    # TO:DO, postponed to a later date
    @torch.inference_mode()
    def __call__(self, frame, interpQueue, writeBuffer=None):
        if self.firstRun:
            with torch.cuda.stream(self.normStream):
                self.processFrame(frame, "I0")
            self.normStream.synchronize()
            self.firstRun = False
            return

        with torch.cuda.stream(self.normStream):
            self.processFrame(frame, "I1")
        self.normStream.synchronize()

        for i in range(self.interpolateFactor - 1):
            t = 0.5
            timestep = torch.full(
                (1, 1, self.height + self.padding[3], self.width + self.padding[1]),
                t,
                dtype=self.dType,
                device=checker.device,
            )
            output = self.processFrame(timestep, "infer")
            interpQueue.put(output)

        self.processFrame(None, "cache")
