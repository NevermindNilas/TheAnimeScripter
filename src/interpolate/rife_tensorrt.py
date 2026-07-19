import logging
import math
import os

import torch
import torch.nn.functional as F

from src.constants import ADOBE
from src.infra.isCudaInit import CudaChecker
from src.infra.logAndPrint import logAndPrint
from src.interpolate._shared import importRifeArch
from src.model.download import resolveWeightPath
from src.model.registry import modelsMap

if ADOBE:
    from src.server.aeComms import progressState


checker = CudaChecker()

torch.set_float32_matmul_precision("medium")


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
        from src.model.trtHandler import (
            tensorRTEngineCreator,
            tensorRTEngineLoader,
            tensorRTEngineNameHandler,
        )
        from src.utils.tensorrt_import import trt

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
        self.filename = modelsMap(
            self.interpolateMethod.replace("-tensorrt", ""),
            modelType="pth",
            half=self.half,
            ensemble=self.ensemble,
        )

        folderName = self.interpolateMethod.replace("-tensorrt", "")
        self.modelPath = resolveWeightPath(
            folderName,
            self.filename,
            downloadModel=folderName,
            modelType="pth",
            half=self.half,
            ensemble=self.ensemble,
        )

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

        IFNet, _Head = importRifeArch(self.interpolateMethod, "v3")

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

        stateDict = torch.load(self.modelPath, map_location="cpu")
        self.model.load_state_dict(stateDict)
        del stateDict

        if self.half:
            self.model = self.model.half()
        else:
            self.model = self.model.float()
        self.model = self.model.to(checker.device)
        torch.cuda.empty_cache()

        if _Head is True:
            self.norm = self.model.encode
        else:
            self.norm = None

        self.engine, self.context = self.tensorRTEngineLoader(enginePath)
        if (
            self.engine is None
            or self.context is None
            or not os.path.exists(enginePath)
        ):
            if ADOBE:
                progressState.update(
                    {"status": f"Exporting {self.interpolateMethod} to TensorRT."}
                )
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
                opset_version=22,
                optimize=False,
                dynamo=False,
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
        self._cachedTimestepValue = 0.5

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
                                frame.to(device=checker.device, dtype=self.dtype),
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
        # Scene-cut reset: re-anchor I0 = frame and, for Head models, re-seed the
        # encoder embedding f0 = encode(frame) IN-PLACE. The engine reads f0 as a
        # fixed binding and otherwise carries it across frames via the "f0-copy"
        # step, so without re-seeding here the previous scene's embedding bleeds
        # into the first interpolation after the cut. Both "I0" and "f0"
        # processFrame cases copy into the fixed buffers (graph/binding-safe).
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

            with torch.cuda.stream(self.stream):
                if self._cachedTimestepValue != t:
                    self.dummyTimeStep.fill_(t)
                    self._cachedTimestepValue = t
                self.cudaGraph.replay()
            self.stream.synchronize()
            with torch.cuda.stream(self.outputStream):
                output = self.dummyOutput.clone()
            self.outputStream.synchronize()
            interpQueue.put(output)

        self.processFrame(None, "cache")
        if self.norm is not None:
            self.processFrame(None, "f0-copy")
