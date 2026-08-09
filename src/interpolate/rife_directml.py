import logging
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from src.constants import ADOBE
from src.infra.isCudaInit import CudaChecker
from src.infra.logAndPrint import logAndPrint, logWarning
from src.infra.providerCheck import warnIfProviderMissing
from src.interpolate._shared import importRifeArch
from src.interpolate._timesteps import interpolateTimestep
from src.model.download import resolveWeightPath
from src.model.registry import modelsMap

if ADOBE:
    from src.server.aeComms import progressState


checker = CudaChecker()

torch.set_float32_matmul_precision("medium")

# ONNX opset 20 renamed GridSample's mode from "bilinear" to "linear", and the
# DirectML EP only registers the opset 16-19 kernel, so an opset 20 export
# drops every warp to the CPU EP (rife4.6 @1080p: 508 ms/frame vs 15 ms). At
# opset 16 DirectML runs GridSample natively and the whole graph stays on the
# GPU, which beats the decomposed warp it used to need.
#
# OpenVINO cannot use either half of that: its ONNX frontend fails outright on
# GridSample-20, and an opset 16 export runs ~2x slower than the decomposition
# (1638 ms vs 814 ms @1080p). So it keeps the decomposed warp at opset 20.
#
# The opset is part of the cached ONNX filename so the two coexist.
DML_OPSET = 16
OPENVINO_OPSET = 20


class RifeDirectML:
    def __init__(
        self,
        interpolateMethod: str = "rife4.6-directml",
        interpolateFactor: int = 2,
        width: int = 0,
        height: int = 0,
        half: bool = True,
        ensemble: bool = False,
    ):
        """
        Interpolates frames using DirectML (or OpenVINO).

        The export opset is picked per backend so the GridSample warps land on
        the GPU; see DML_OPSET/OPENVINO_OPSET above.

        Arguments:
            - interpolateMethod (str, optional): Interpolation method. Defaults to "rife4.6-directml".
            - interpolateFactor (int, optional): Interpolation factor. Defaults to 2.
            - width (int, optional): Width of the frame. Defaults to 0.
            - height (int, optional): Height of the frame. Defaults to 0.
            - half (bool, optional): Half precision. Defaults to True.
            - ensemble (bool, optional): Ensemble mode. Defaults to False.
        """
        import onnxruntime as ort

        if "openvino" in interpolateMethod:
            logAndPrint(
                "OpenVINO backend is an experimental feature, please report any issues you encounter.",
                "yellow",
            )
            import openvino  # noqa: F401

        self.ort = ort

        self.interpolateMethod = interpolateMethod
        self.interpolateFactor = interpolateFactor
        self.width = width
        self.height = height
        self.half = half
        self.ensemble = ensemble
        self.model = None

        if self.width > 1920 and self.height > 1080:
            self.scale = 0.5
            if self.half:
                logAndPrint(
                    "UHD and fp16 are not compatible with RIFE, defaulting to fp32",
                    "yellow",
                )
                self.half = False
        else:
            self.scale = 1.0

        self.handleModel()

    def handleModel(self):
        self.deviceType = "cpu"
        self.device = torch.device(self.deviceType)

        self.isOpenVino = "openvino" in self.interpolateMethod
        self.opset = OPENVINO_OPSET if self.isOpenVino else DML_OPSET

        if self.half:
            self.numpyDType = np.float16
            self.torchDType = torch.float16
        else:
            self.numpyDType = np.float32
            self.torchDType = torch.float32

        # No torch.set_default_dtype here: it was set to float16 and never
        # restored, so in a batch run every stage built after the interpolator
        # -- including all stages of videos 2..N -- allocated its parameters in
        # fp16. The model already takes dtype=self.dtype at construction and is
        # cast explicitly below, so nothing depended on the global.
        self.filename = modelsMap(
            self.interpolateMethod.replace("-directml", ""),
            modelType="pth",
            half=self.half,
            ensemble=self.ensemble,
        )

        folderName = self.interpolateMethod.replace("-directml", "").replace(
            "-openvino", ""
        )
        self.modelPath = resolveWeightPath(
            folderName,
            self.filename,
            downloadModel=folderName,
            modelType="pth",
            half=self.half,
            ensemble=self.ensemble,
        )

        if self.interpolateMethod in [
            "rife4.25-directml",
            "rife4.25-heavy-directml",
            "rife4.25-openvino",
            "rife4.25-heavy-openvino",
        ]:
            mul = 64
        elif self.interpolateMethod in [
            "rife4.25-lite-directml",
            "rife4.25-lite-openvino",
        ]:
            mul = 128
        else:
            mul = 32

        self.dtype = torch.float16 if self.half else torch.float32
        tmp = max(mul, int(mul / self.scale))
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
            decomposedWarp=self.isOpenVino,
        )
        stateDict = torch.load(self.modelPath, map_location="cpu")
        self.model.load_state_dict(stateDict, strict=False)
        del stateDict

        if self.half:
            self.model = self.model.half()
        else:
            self.model = self.model.float()

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

        # The opset is part of the name so an existing cache exported at a
        # different opset is not reused.
        self.modelPath = self.modelPath.replace(
            ".pth",
            f"_{self.width}x{self.height}_{'fp16' if self.half else 'fp32'}"
            f"_directml_op{self.opset}_nocache.onnx",
        )

        if not os.path.exists(self.modelPath):
            if ADOBE:
                progressState.update(
                    {"status": f"Exporting {self.interpolateMethod} to ONNX."}
                )
            logAndPrint("Exporting model to ONNX", "green")
            inputList = [dummyInput1, dummyInput2, dummyInput3]
            inputNames = ["img0", "img1", "timestep"]
            outputNames = ["output"]
            dynamicAxes = {
                "img0": {2: "height", 3: "width"},
                "img1": {2: "height", 3: "width"},
                "timestep": {2: "height", 3: "width"},
                "output": {1: "height", 2: "width"},
            }

            logging.info(f"Exporting model to {self.modelPath}")

            torch.onnx.export(
                self.model,
                tuple(inputList),
                self.modelPath,
                input_names=inputNames,
                output_names=outputNames,
                dynamic_axes=dynamicAxes,
                opset_version=self.opset,
                optimize=False,
                dynamo=False,
            )
        providers = self.ort.get_available_providers()
        logging.info(f"Available providers: {providers}")
        if (
            "DmlExecutionProvider" in providers
            or "OpenVINOExecutionProvider" in providers
        ):
            if "directml" in self.interpolateMethod:
                logging.info("DirectML provider available. Defaulting to DirectML")
                self.model = self.ort.InferenceSession(
                    self.modelPath, providers=["DmlExecutionProvider"]
                )
                warnIfProviderMissing(
                    self.model, "DmlExecutionProvider", "DirectML interpolate"
                )
            elif "openvino" in self.interpolateMethod:
                logging.info("Using OpenVINO model")
                self.model = self.ort.InferenceSession(
                    self.modelPath,
                    providers=[
                        ("OpenVINOExecutionProvider", {"device_type": "AUTO:GPU,CPU"})
                    ],
                )
                warnIfProviderMissing(
                    self.model, "OpenVINOExecutionProvider", "OpenVINO interpolate"
                )
        else:
            logWarning(
                "DirectML/OpenVINO provider not available, falling back to CPU, expect significantly worse performance, ensure that your drivers are up to date and your GPU supports DirectX 12"
            )
            self.model = self.ort.InferenceSession(
                self.modelPath, providers=["CPUExecutionProvider"]
            )

        self.needsOutputSync = any(
            provider in self.model.get_providers()
            for provider in ("DmlExecutionProvider", "OpenVINOExecutionProvider")
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

        self.firstRun = True

    @torch.inference_mode()
    def cacheFrameReset(self, frame):
        self.processFrame(frame, "I0")

    @torch.inference_mode()
    def processFrame(self, frame, name=None):
        match name:
            case "I0":
                self.I0.copy_(
                    F.pad(
                        frame.to(device=self.device, dtype=self.dtype),
                        self.padding,
                    ),
                    non_blocking=False,
                )

            case "I1":
                self.I1.copy_(
                    F.pad(
                        frame.to(device=self.device, dtype=self.dtype),
                        self.padding,
                    ),
                    non_blocking=False,
                )

            case "cache":
                self.I0.copy_(self.I1, non_blocking=False)

            case "timestep":
                self.dummyTimeStep.copy_(frame, non_blocking=False)

    @torch.inference_mode()
    def __call__(
        self, frame: torch.Tensor, interpQueue, framesToInsert=1, timesteps=None
    ):
        if self.firstRun:
            self.processFrame(frame, "I0")

            self.firstRun = False
            return

        self.processFrame(frame, "I1")

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
        for i in range(framesToInsert):
            t = interpolateTimestep(i, framesToInsert, timesteps)

            self.dummyTimeStep.fill_(t)

            # ORT reads bound inputs at bind_input() time, not at run time, so the
            # timestep MUST be rebound after every fill_; otherwise all inserted
            # frames in a gap reuse the timestep captured before the loop (0.5),
            # collapsing factor>2 interpolation into duplicate frames.
            self.IoBinding.bind_input(
                name="timestep",
                device_type=self.deviceType,
                device_id=0,
                element_type=self.numpyDType,
                shape=self.dummyTimeStep.shape,
                buffer_ptr=self.dummyTimeStep.data_ptr(),
            )

            self.model.run_with_iobinding(self.IoBinding)
            if self.needsOutputSync:
                self.IoBinding.synchronize_outputs()

            interpQueue.put(self.dummyOutput.clone())

        self.processFrame(None, "cache")
        return frame
