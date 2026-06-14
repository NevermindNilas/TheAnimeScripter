import os
import torch
import logging

from src.model.download import resolveWeightPath
from src.model.registry import modelsMap
from src.infra.logAndPrint import logAndPrint
from src.constants import ADOBE

if ADOBE:
    from src.server.aeComms import progressState


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
            elif "openvino" in self.upscaleMethod:
                logging.info("Using OpenVINO model")
                self.model = self.ort.InferenceSession(
                    modelPath,
                    providers=[
                        ("OpenVINOExecutionProvider", {"device_type": "AUTO:GPU,CPU"})
                    ],
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

        self.usingCpuFallback = False
        self.modelPath = modelPath

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

            self.dummyInput.copy_(frame.contiguous(), non_blocking=False)

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
            frame = self.dummyOutput.contiguous()

            return frame

        except Exception as e:
            if not self.usingCpuFallback:
                logging.warning(
                    f"DirectML/OpenVINO inference failed ({e}); falling back to CPU provider."
                )
                self._fallbackToCpu()
                return self.__call__(frame, nextFrame)
            else:
                logging.exception(
                    f"Something went wrong while processing the frame, {e}"
                )
                raise
