import os
import torch
import logging
import torch.nn.functional as F
import math
import numpy as np

from src.model.download import resolveWeightPath
from src.model.registry import modelsMap
from src.infra.isCudaInit import CudaChecker
from src.infra.logAndPrint import logAndPrint

from src.constants import ADOBE

if ADOBE:
    from src.server.aeComms import progressState


checker = CudaChecker()

torch.set_float32_matmul_precision("medium")


_RIFE_V1 = {
    "rife": ("IFNet425", "IFNet_rife425"),
    "rife4.25": ("IFNet425", "IFNet_rife425"),
    "rife4.25-heavy": ("IFNet425Heavy", "IFNet_rife425heavy"),
    "rife4.25-lite": ("IFNet425Lite", "IFNet_rife425lite"),
    "rife4.22": ("IFNet422", "IFNet_rife422"),
    "rife4.22-lite": ("IFNet422Lite", "IFNet_rife422lite"),
    "rife4.21": ("IFNet421", "IFNet_rife421"),
    "rife4.20": ("IFNet420", "IFNet_rife420"),
    "rife4.18": ("IFNet418", "IFNet_rife418"),
    "rife4.17": ("IFNet417", "IFNet_rife417"),
    "rife4.15-lite": ("IFNet415Lite", "IFNet_rife415lite"),
    "rife4.16-lite": ("IFNet416Lite", "IFNet_rife416lite"),
    "rife4.6": ("IFNet46", "IFNet_rife46"),
    "rife_elexor": (None, "IFNet_elexor_cuda"),
}


def _loadV1(method, half):
    fastName, baseMod = _RIFE_V1[method]
    if half and fastName:
        from src.rifearches import rife_fast

        return getattr(rife_fast, fastName)
    mod = __import__(f"src.rifearches.{baseMod}", fromlist=["IFNet"])
    return mod.IFNet


def importRifeArch(interpolateMethod, version, half=True):
    match version:
        case "v1":
            return _loadV1(interpolateMethod, half)

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
                case "rife4.6-tensorrt":
                    from src.rifearches.Rife46_v3 import IFNet

                    Head = False
                case "rife4.6-directml" | "rife4.6-openvino":
                    from src.rifearches.Rife_directml import IFNet_46 as IFNet

                    Head = False
                case "rife4.22-directml" | "rife4.22-openvino":
                    from src.rifearches.Rife_directml import IFNet_422 as IFNet

                    Head = True
                case (
                    "rife4.15-directml"
                    | "rife4.17-directml"
                    | "rife4.18-directml"
                    | "rife4.15-openvino"
                    | "rife4.17-openvino"
                    | "rife4.18-openvino"
                ):
                    from src.rifearches.Rife_directml import IFNet_415 as IFNet

                    Head = True
                case (
                    "rife4.20-directml"
                    | "rife4.21-directml"
                    | "rife4.20-openvino"
                    | "rife4.21-openvino"
                ):
                    from src.rifearches.Rife_directml import IFNet_420 as IFNet

                    Head = True
                case "rife4.22-lite-directml" | "rife4.22-lite-openvino":
                    from src.rifearches.Rife_directml import IFNet_422_lite as IFNet

                    Head = True
                case "rife4.25-directml" | "rife4.25-openvino":
                    from src.rifearches.Rife_directml import IFNet_425 as IFNet

                    Head = True
                case "rife4.25-lite-directml" | "rife4.25-lite-openvino":
                    from src.rifearches.Rife_directml import IFNet_425_lite as IFNet

                    Head = True
                case "rife4.25-heavy-directml" | "rife4.25-heavy-openvino":
                    from src.rifearches.Rife_directml import IFNet_425_heavy as IFNet

                    Head = True
                case "rife_elexor-tensorrt":
                    from src.rifearches.IFNet_elexor_tensorrt import IFNet

                    Head = True
            return IFNet, Head


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
        Interpolates frames using DirectML with decomposed grid_sample.

        This implementation uses a custom grid_sample decomposition that breaks
        down the operation into primitive tensor ops (floor, gather, weighted sum)
        that DirectML can accelerate, avoiding CPU fallback.

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
        elif self.interpolateMethod in [
            "rife4.22-lite-directml",
            "rife4.22-lite-openvino",
        ]:
            mul = 32
        elif self.interpolateMethod in [
            "rife4.6-directml",
            "rife4.6-openvino",
        ]:
            mul = 32
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

        self.modelPath = self.modelPath.replace(
            ".pth",
            f"_{self.width}x{self.height}_{'fp16' if self.half else 'fp32'}_directml_nocache.onnx",
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
                opset_version=20,
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
            elif "openvino" in self.interpolateMethod:
                logging.info("Using OpenVINO model")
                self.model = self.ort.InferenceSession(
                    self.modelPath,
                    providers=[
                        ("OpenVINOExecutionProvider", {"device_type": "AUTO:GPU,CPU"})
                    ],
                )
        else:
            logging.info(
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
            if timesteps is not None and i < len(timesteps):
                t = timesteps[i]
            else:
                t = (i + 1) * 1 / (framesToInsert + 1)

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
