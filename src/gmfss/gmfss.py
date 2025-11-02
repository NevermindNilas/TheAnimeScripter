import os
import torch
import logging
import math

from src.utils.downloadModels import downloadModels, weightsDir
from torch.nn import functional as F
from src.utils.isCudaInit import CudaChecker
from src.utils.logAndPrint import logAndPrint

checker = CudaChecker()

torch.set_float32_matmul_precision("medium")
# from: https://github.com/HolyWu/vs-gmfss_fortuna/blob/master/vsgmfss_fortuna/__init__.py


class GMFSS:
    def __init__(
        self,
        interpolation_factor,
        half,
        width,
        height,
        ensemble=False,
        compileMode: str = "default",
    ):
        self.width = width
        self.height = height
        self.half = half
        self.interpolation_factor = interpolation_factor
        self.ensemble = ensemble
        self.compileMode: str = compileMode

        self.ph = ((self.height - 1) // 64 + 1) * 64
        self.pw = ((self.width - 1) // 64 + 1) * 64

        if self.width > 1920 or self.height > 1080:
            self.scale = 0.5
        else:
            self.scale = 1

        self.handleModel()

    def handleModel(self):
        if not os.path.exists(os.path.join(weightsDir, "gmfss")):
            modelDir = os.path.dirname(downloadModels("gmfss"))
        else:
            modelDir = os.path.join(weightsDir, "gmfss")

        modelType = "union"

        self.device = torch.device("cuda" if checker.cudaAvailable else "cpu")

        torch.set_grad_enabled(False)
        if checker.cudaAvailable:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

        from .model.GMFSS import GMFSS as Model

        self.model = Model(modelDir, modelType, self.scale, ensemble=self.ensemble)
        self.model.eval().to(self.device, memory_format=torch.channels_last)

        self.dtype = torch.float
        if checker.cudaAvailable and self.half:
            self.model.half()
            self.dtype = torch.half

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

        self.I0 = torch.zeros(
            1,
            3,
            self.ph,
            self.pw,
            dtype=torch.float16 if self.half else torch.float32,
            device=self.device,
        )

        self.I1 = torch.zeros(
            1,
            3,
            self.ph,
            self.pw,
            dtype=torch.float16 if self.half else torch.float32,
            device=self.device,
        )

        self.stream = torch.cuda.Stream()
        self.firstRun = True

    @torch.inference_mode()
    def cacheFrame(self):
        self.I0.copy_(self.I1, non_blocking=True)
        # self.model.cacheFrame()

    @torch.inference_mode()
    def processFrame(self, frame):
        return frame.to(
            self.device,
            non_blocking=True,
            dtype=torch.float16 if self.half else torch.float32,
        ).to(memory_format=torch.channels_last)

    @torch.inference_mode()
    def padFrame(self, frame):
        return (
            F.pad(frame, [0, self.pw - self.width, 0, self.ph - self.height])
            if (self.pw != self.width or self.ph != self.height)
            else frame
        )

    @torch.inference_mode()
    def __call__(self, frame, interpQueue, framesToInsert: int = 2, timesteps=None):
        with torch.cuda.stream(self.stream):
            if self.firstRun is True:
                self.I0 = self.padFrame(self.processFrame(frame))
                self.firstRun = False
                return

            self.I1 = self.padFrame(self.processFrame(frame))

            for i in range(framesToInsert):
                if timesteps is not None and i < len(timesteps):
                    t = timesteps[i]
                else:
                    t = (i + 1) * 1 / (framesToInsert + 1)
                timestep = torch.tensor(
                    [t],
                    dtype=self.dtype,
                    device=checker.device,
                )
                output = self.model(self.I0, self.I1, timestep)[
                    :, :, : self.height, : self.width
                ]
                self.stream.synchronize()
                interpQueue.put(output)

            self.cacheFrame()


class GMFSSTensorRT:
    def __init__(
        self,
        interpolateFactor: int = 2,
        width: int = 0,
        height: int = 0,
        half: bool = True,
        ensemble: bool = False,
    ):
        raise NotImplementedError(
            "TensorRT is not supported for GMFSS, use the default GMFSS implementation"
        )
        """
        Interpolates frames using TensorRT

        Args:
            interpolateFactor (int, optional): Interpolation factor. Defaults to 2.
            width (int, optional): Width of the frame. Defaults to 0.
            height (int, optional): Height of the frame. Defaults to 0.
            half (bool, optional): Half resolution. Defaults to True.
            ensemble (bool, optional): Ensemble. Defaults to False.
        """
        import tensorrt as trt
        from src.utils.trtHandler import (
            tensorRTEngineCreator,
            tensorRTEngineLoader,
            tensorRTEngineNameHandler,
        )

        self.tensorRTEngineCreator = tensorRTEngineCreator
        self.tensorRTEngineLoader = tensorRTEngineLoader
        self.tensorRTEngineNameHandler = tensorRTEngineNameHandler
        self.trt = trt

        self.interpolateFactor = interpolateFactor
        self.width = width
        self.height = height
        self.half = half
        self.ensemble = ensemble
        self.model = None
        if self.width > 1920 and self.height > 1080:
            if self.half:
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
        if not os.path.exists(os.path.join(weightsDir, "gmfss")):
            modelDir = os.path.dirname(downloadModels("gmfss"))
        else:
            modelDir = os.path.join(weightsDir, "gmfss")

        modelType = "union"

        self.device = torch.device("cuda" if checker.cudaAvailable else "cpu")

        torch.set_grad_enabled(False)
        if checker.cudaAvailable:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

        from .model.GMFSS import GMFSS as Model

        self.model = Model(modelDir, modelType, self.scale, ensemble=self.ensemble)
        self.model.eval().to(checker.device)

        self.dtype = torch.float16 if self.half else torch.float32
        mul = 64
        tmp = max(mul, int(mul / 1.0))
        self.pw = math.ceil(self.width / tmp) * tmp
        self.ph = math.ceil(self.height / tmp) * tmp
        self.padding = (0, self.pw - self.width, 0, self.ph - self.height)

        dummyInput1 = torch.zeros(
            1, 3, self.ph, self.pw, dtype=self.dtype, device=checker.device
        )
        dummyInput2 = torch.zeros(
            1, 3, self.ph, self.pw, dtype=self.dtype, device=checker.device
        )
        dummyInput3 = torch.tensor(
            [(0 + 1) * 1.0 / (self.interpolateFactor + 1)],
            dtype=self.dtype,
            device=checker.device,
        )
        self.modelPath = os.path.join(modelDir, "gmfss.onnx")
        inputList = [dummyInput1, dummyInput2, dummyInput3]
        inputNames = ["img0", "img1", "timestep"]
        outputNames = ["output"]
        dynamicAxes = {
            "img0": {2: "height", 3: "width"},
            "img1": {2: "height", 3: "width"},
            "output": {2: "height", 3: "width"},
        }
        torch.onnx.export(
            self.model,
            tuple(inputList),
            self.modelPath,
            input_names=inputNames,
            output_names=outputNames,
            dynamic_axes=dynamicAxes,
            opset_version=20,
            dynamo=False,
        )

        inputs = [
            [1, 3, self.ph, self.pw],
            [1, 3, self.ph, self.pw],
            [1],
        ]
        inputsMin = inputsOpt = inputsMax = inputs
        enginePath = self.modelPath.replace(".onnx", ".engine")
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

        self.dummyTimeStep = torch.tensor(
            (1 + 1) * 1.0 / (self.interpolateFactor + 1),
            dtype=self.dtype,
            device=self.device,
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

        self.tensors.extend([self.dummyOutput])

        self.bindings = [tensor.data_ptr() for tensor in self.tensors]

        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            self.context.set_tensor_address(tensor_name, self.bindings[i])

            if self.engine.get_tensor_mode(tensor_name) == self.trt.TensorIOMode.INPUT:
                self.context.set_input_shape(tensor_name, self.tensors[i].shape)

        self.firstRun = True
        self.normStream = torch.cuda.Stream()
        self.outputStream = torch.cuda.Stream()

        if self.interpolateSkip is not None:
            self.skippedCounter = 0

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

                case "cache":
                    self.I0.copy_(self.I1, non_blocking=True)

                case "timestep":
                    self.dummyTimeStep.copy_(frame, non_blocking=True)

        self.normStream.synchronize()

    @torch.inference_mode()
    def cacheFrameReset(self, frame):
        self.processFrame(frame, "I0")

    @torch.inference_mode()
    def __call__(self, frame, interpQueue, framesToInsert: int = 2, timesteps=None):
        if self.firstRun:
            if self.norm is not None:
                self.processFrame(frame, "f0")

            self.processFrame(frame, "I0")

            self.firstRun = False
            return

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

            self.processFrame(timestep, "timestep")
            with torch.cuda.stream(self.stream):
                self.cudaGraph.replay()
            self.stream.synchronize()
            with torch.cuda.stream(self.outputStream):
                output = self.dummyOutput.clone()
            self.outputStream.synchronize()
            interpQueue.put(output)

        self.processFrame(None, "cache")

    def getSkippedCounter(self):
        return self.skippedCounter
