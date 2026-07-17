import logging
import os

import torch

from src.constants import ADOBE
from src.infra.isCudaInit import CudaChecker
from src.infra.logAndPrint import logAndPrint, logWarning
from src.infra.providerCheck import warnIfProviderMissing
from src.model.download import resolveWeightPath
from src.model.registry import modelsMap

if ADOBE:
    from src.server.aeComms import progressState

checker = CudaChecker()

torch.set_float32_matmul_precision("medium")


def calculatePadding(width, height, multiple=4):
    padW = (multiple - (width % multiple)) % multiple
    padH = (multiple - (height % multiple)) % multiple
    return (0, padW, 0, padH)


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

        import numpy as np
        import onnxruntime as ort

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
        if ADOBE:
            progressState.update(
                {"status": f"Loading DirectML upscale model: {self.upscaleMethod}..."}
            )

        if not self.customModel:
            method = self.upscaleMethod
            if "openvino" in self.upscaleMethod:
                method = method.replace("openvino", "directml")

            self.filename = modelsMap(method, self.upscaleFactor, modelType="onnx")
            if "-directml" in self.upscaleMethod:
                folderName = self.upscaleMethod.replace("-directml", "-onnx")
            elif "-openvino" in self.upscaleMethod:
                folderName = self.upscaleMethod.replace("-openvino", "-onnx")

            modelPath = resolveWeightPath(
                folderName,
                self.filename,
                downloadModel=method,
                upscaleFactor=self.upscaleFactor,
                modelType="onnx",
                half=self.half,
            )
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

        if (
            "DmlExecutionProvider" in providers
            or "OpenVINOExecutionProvider" in providers
        ):
            if "directml" in self.upscaleMethod:
                logging.info("DirectML provider available. Defaulting to DirectML")
                self.model = self.ort.InferenceSession(
                    modelPath, providers=["DmlExecutionProvider"]
                )
                warnIfProviderMissing(
                    self.model, "DmlExecutionProvider", "DirectML upscale"
                )
            elif "openvino" in self.upscaleMethod:
                logging.info("Using OpenVINO model")
                self.model = self.ort.InferenceSession(
                    modelPath,
                    providers=[
                        ("OpenVINOExecutionProvider", {"device_type": "AUTO:GPU,CPU"})
                    ],
                )
                warnIfProviderMissing(
                    self.model, "OpenVINOExecutionProvider", "OpenVINO upscale"
                )
        else:
            logWarning(
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

        self.modelPath = modelPath

        # RealCUGAN-family archs (fallin_*, shufflecugan, aniscale2) need both
        # input dims divisible by 4; feeding an odd dim raises an ONNX broadcast
        # error on an internal skip Add. Detect the arch's requirement once and
        # reflect-pad the input to it, cropping the surplus back off the output.
        # Fully-convolutional archs report 1 here, so padding stays zero and the
        # output is bit-identical to the unpadded path.
        self.requiredMultiple = self._detectRequiredMultiple()
        self.padding = calculatePadding(self.width, self.height, self.requiredMultiple)
        self.paddedWidth = self.width + self.padding[1]
        self.paddedHeight = self.height + self.padding[3]

        self.IoBinding = self.model.io_binding()
        self.dummyInput = torch.zeros(
            (1, 3, self.paddedHeight, self.paddedWidth),
            device=self.deviceType,
            dtype=self.torchDType,
        ).contiguous()

        self.dummyOutput = torch.zeros(
            (
                1,
                3,
                self.paddedHeight * self.upscaleFactor,
                self.paddedWidth * self.upscaleFactor,
            ),
            device=self.deviceType,
            dtype=self.torchDType,
        ).contiguous()

        self.usingCpuFallback = False

    def _detectRequiredMultiple(self) -> int:
        """Probe a lightweight CPU session to learn the arch's input multiple."""
        from src.upscale._shared import smallestValidMultiple

        probe = self.ort.InferenceSession(
            self.modelPath, providers=["CPUExecutionProvider"]
        )
        inputName = probe.get_inputs()[0].name

        def runOK(h, w):
            try:
                probe.run(
                    None, {inputName: self.np.zeros((1, 3, h, w), self.numpyDType)}
                )
                return True
            except Exception:
                return False

        return smallestValidMultiple(runOK)

    def _fallbackToCpu(self):
        """Reinitialize model with CPU provider after DirectML/OpenVINO failure."""
        logAndPrint(
            "DirectML/OpenVINO encountered an error, falling back to CPU. Performance will be slower.",
            "yellow",
        )

        self.model = self.ort.InferenceSession(
            self.modelPath, providers=["CPUExecutionProvider"]
        )

        self.IoBinding = self.model.io_binding()
        self.IoBinding.bind_output(
            name="output",
            device_type=self.deviceType,
            device_id=0,
            element_type=self.numpyDType,
            shape=self.dummyOutput.shape,
            buffer_ptr=self.dummyOutput.data_ptr(),
        )
        self.IoBinding.bind_input(
            name="input",
            device_type=self.deviceType,
            device_id=0,
            element_type=self.numpyDType,
            shape=self.dummyInput.shape,
            buffer_ptr=self.dummyInput.data_ptr(),
        )

        self.usingCpuFallback = True

    @torch.inference_mode()
    def __call__(self, frame: torch.tensor, nextFrame: None) -> torch.tensor:
        """
        Run the model on the input frame
        """
        try:
            if self.half:
                frame = frame.half()
            else:
                frame = frame.float()

            paddedFrame = frame
            if self.padding[1] or self.padding[3]:
                paddedFrame = torch.nn.functional.pad(
                    frame, self.padding, mode="reflect"
                )

            self.dummyInput.copy_(paddedFrame.contiguous(), non_blocking=False)

            self.IoBinding.bind_input(
                name="input",
                device_type=self.deviceType,
                device_id=0,
                element_type=self.numpyDType,
                shape=self.dummyInput.shape,
                buffer_ptr=self.dummyInput.data_ptr(),
            )

            self.IoBinding.bind_output(
                name="output",
                device_type=self.deviceType,
                device_id=0,
                element_type=self.numpyDType,
                shape=self.dummyOutput.shape,
                buffer_ptr=self.dummyOutput.data_ptr(),
            )

            self.model.run_with_iobinding(self.IoBinding)

            output = self.dummyOutput
            if self.padding[1] or self.padding[3]:
                output = output[
                    :,
                    :,
                    : self.height * self.upscaleFactor,
                    : self.width * self.upscaleFactor,
                ]

            return output.contiguous()

        except Exception as e:
            if not self.usingCpuFallback:
                logWarning(
                    f"DirectML/OpenVINO inference failed ({e}); falling back to CPU provider."
                )
                self._fallbackToCpu()
                return self.__call__(frame, nextFrame)
            else:
                logging.exception(
                    f"Something went wrong while processing the frame, {e}"
                )
                raise


class AnimeSRDirectML:
    """
    AnimeSR DirectML/OpenVINO implementation with 5-input architecture.
    This handles the multi-input/multi-output nature of AnimeSR.
    """

    # Recurrent arch: needs the next source frame handed to it. The previous
    # frame is cached internally (padded), so it is not requested here.
    temporalWindow = (0, 1)

    def __init__(
        self,
        upscaleMethod: str,
        half: bool,
        width: int,
        height: int,
    ):
        import numpy as np
        import onnxruntime as ort

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
            warnIfProviderMissing(
                self.model, "DmlExecutionProvider", "DirectML AnimeSR upscale"
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
            warnIfProviderMissing(
                self.model, "OpenVINOExecutionProvider", "OpenVINO AnimeSR upscale"
            )
        else:
            logWarning("DirectML/OpenVINO provider not available, falling back to CPU")
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
