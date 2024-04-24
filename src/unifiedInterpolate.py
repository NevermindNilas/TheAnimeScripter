import os
import torch
import numpy as np
import logging
import onnxruntime as ort
from torch.nn import functional as F

from .downloadModels import downloadModels, weightsDir, modelsMap

# Apparently this can improve performance slightly
torch.set_float32_matmul_precision("medium")
ort.set_default_logger_severity(3)

class RifeCuda:
    def __init__(
        self,
        interpolation_factor,
        half,
        width,
        height,
        UHD,
        interpolate_method,
        ensemble=False,
        nt=1,
    ):
        """
        Initialize the RIFE model

        Args:
            interpolation_factor (int): The factor to interpolate by.
            half (bool): Whether to use half precision.
            width (int): The width of the input frame.
            height (int): The height of the input frame.
            UHD (bool): Whether to use UHD mode.
            interpolate_method (str): The method to use for interpolation.
            ensemble (bool): Whether to use ensemble mode.
            nt (int): The number of streams to use, not available for now.
        """
        self.interpolation_factor = interpolation_factor
        self.half = half
        self.UHD = UHD
        self.scale = 1.0
        self.width = width
        self.height = height
        self.interpolate_method = interpolate_method
        self.ensemble = ensemble
        self.nt = nt

        if self.UHD:
            self.scale = 0.5

        self.handle_model()

    def handle_model(self):
        """
        Load the desired model
        """

        self.filename = modelsMap(self.interpolate_method)
        if not os.path.exists(os.path.join(weightsDir, "rife", self.filename)):
            modelPath = downloadModels(model=self.interpolate_method)
        else:
            modelPath = os.path.join(weightsDir, "rife", self.filename)

        match self.interpolate_method:
            case "rife" | "rife4.15":
                from .rifearches.IFNet_rife415 import IFNet
            case "rife4.15-lite":
                from .rifearches.IFNet_rife415lite import IFNet
            case "rife4.16-lite":
                from .rifearches.IFNet_rife416lite import IFNet
            case "rife4.14":
                from .rifearches.IFNet_rife414 import IFNet
            case "rife4.6":
                from .rifearches.IFNet_rife46 import IFNet

        self.model = IFNet()
        self.isCudaAvailable = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.isCudaAvailable else "cpu")

        torch.set_grad_enabled(False)
        if self.isCudaAvailable:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            if self.half:
                torch.set_default_dtype(torch.float16)

        if self.isCudaAvailable and self.half:
            self.model.half()

        self.model.load_state_dict(torch.load(modelPath))
        self.model.eval().cuda() if self.isCudaAvailable else self.model.eval()
        self.model.to(self.device)

        ph = ((self.height - 1) // 32 + 1) * 32
        pw = ((self.width - 1) // 32 + 1) * 32
        self.padding = (0, pw - self.width, 0, ph - self.height)

        self.I0 = None
        self.scaleList = [
            8 / self.scale,
            4 / self.scale,
            2 / self.scale,
            1 / self.scale,
        ]

    @torch.inference_mode()
    def make_inference(self, timestep):
        output = self.model(self.I0, self.I1, timestep, self.scaleList, self.ensemble)
        output = output[:, :, : self.height, : self.width]
        output = (output[0] * 255.0).byte().cpu().numpy().transpose(1, 2, 0)

        return output

    def cacheFrame(self):
        self.I0 = self.I1.clone()

    @torch.inference_mode()
    def processFrame(self, frame):
        frame = (
            torch.from_numpy(frame)
            .to(self.device, non_blocking=True)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .mul_(1 / 255)
        )

        frame = frame.half() if self.half and self.isCudaAvailable else frame

        if self.padding != (0, 0, 0, 0):
            frame = F.pad(frame, [0, self.padding[1], 0, self.padding[3]])

        return frame.contiguous(memory_format=torch.channels_last)

    @torch.inference_mode()
    def run(self, I1):
        if self.I0 is None:
            self.I0 = self.processFrame(I1)
            return False

        self.I1 = self.processFrame(I1)
        return True


class RifeDirectML:
    def __init__(
        self,
        interpolateMethod: str = "rife4.15",
        half=True,
        ensemble: bool = False,
        nt: int = 1,
        width: int = 0,
        height: int = 0,
    ):
        """
        Interpolates frames using DirectML

        Args:
            interpolateMethod (str, optional): Interpolation method. Defaults to "rife415".
            half (bool, optional): Half resolution. Defaults to True.
            ensemble (bool, optional): Ensemble. Defaults to False.
            nt (int, optional): Number of threads. Defaults to 1.
        """
        self.interpolateMethod = interpolateMethod
        self.half = half
        self.ensemble = ensemble
        self.nt = nt
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.width = width
        self.height = height
        #self.modelPath = os.path.join(weightsDir, modelsMap[self.interpolateMethod])

        self.handleModel()

    def handleModel(self):
        """
        Load the model
        """

        """
        self.filename = modelsMap(model=self.interpolateMethod, modelType="onnx")
        if not os.path.exists(self.modelPath):
            os.path.join(weightsDir, self.interpolateMethod, self.filename)

            modelPath = downloadModels(
                model=self.interpolateMethod,
                modelType="onnx",
                half=self.half,
            )
        else:
            modelPath = os.path.join(weightsDir, self.interpolateMethod, self.filename)
        """
        modelPath = r"G:\TheAnimeScripter\rife_4.15_fp32 (3).onnx"
        providers = ort.get_available_providers()

        if "DmlExecutionProvider" in providers:
            logging.info("DirectML provider available. Defaulting to DirectML")
            self.model = ort.InferenceSession(
                modelPath, providers=["DmlExecutionProvider"]
            )
        else:
            logging.info(
                "DirectML provider not available, falling back to CPU, expect significantly worse performance, ensure that your drivers are up to date and your GPU supports DirectX 12"
            )
            self.model = ort.InferenceSession(
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
        self.dummyInput1 = torch.zeros(1, 3, self.height, self.width, dtype=self.torchDType).to(self.device)
        self.dummyInput2 = torch.zeros(1, 3, self.height, self.width, dtype=self.torchDType).to(self.device)
        self.dummyTimeStep = torch.tensor([0.5], dtype=self.torchDType).to(self.device)
        self.dummyOutput = torch.zeros(1, 3, self.height, self.width, dtype=self.torchDType).to(self.device)

        print(self.dummyInput1.shape, self.dummyInput2.shape, self.dummyTimeStep.shape, self.dummyOutput.shape)
        self.dummyInput1 = self.dummyInput1.contiguous()
        self.dummyInput2 = self.dummyInput2.contiguous()
        self.dummyTimeStep = self.dummyTimeStep.contiguous()
        self.dummyOutput = self.dummyOutput.contiguous()

        self.IoBinding.bind_output(
            name="out0",
            device_type=self.deviceType,
            device_id=0,
            element_type=self.numpyDType,
            shape=self.dummyOutput.shape,
            buffer_ptr=self.dummyOutput.data_ptr(),
        )

        self.I0 = None

    @torch.inference_mode()
    def make_inference(self, timestep):
        timestep = torch.tensor([timestep], dtype=self.torchDType).to(self.device)
        self.dummyTimeStep.copy_(timestep)

        self.IoBinding.bind_input(
            name="in0",
            device_type=self.deviceType,
            device_id=0,
            element_type=self.numpyDType,
            shape=self.dummyInput1.shape,
            buffer_ptr=self.dummyInput1.data_ptr(),
        )

        self.IoBinding.bind_input(
            name="in1",
            device_type=self.deviceType,
            device_id=0,
            element_type=self.numpyDType,
            shape=self.dummyInput2.shape,
            buffer_ptr=self.dummyInput2.data_ptr(),
        )

        self.IoBinding.bind_input(
            name="in2",
            device_type=self.deviceType,
            device_id=0,
            element_type=self.numpyDType,
            shape=self.dummyTimeStep.shape,
            buffer_ptr=self.dummyTimeStep.data_ptr(),
        )

        self.model.run_with_iobinding(self.IoBinding)

        frame = (
            self.dummyOutput.squeeze(0)
            .permute(1, 2, 0)
            .mul_(255)
            .clamp_(1, 255)
            .byte()
            .cpu()
            .numpy()
        )

        return frame
    
    def cacheFrame(self):
        self.I0 = self.I1.clone()
        self.dummyInput1.copy_(self.I0)
    
    @torch.inference_mode()
    def processFrame(self, frame):
        frame = (
            torch.from_numpy(frame)
            .to(self.device, non_blocking=True)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .mul_(1 / 255)
        )

        frame = frame.half() if self.half else frame

        return frame.contiguous()

    @torch.inference_mode()
    def run(self, I1):
        if self.I0 is None:
            self.I0 = self.processFrame(I1)
            self.dummyInput1.copy_(self.I0)
            return False

        self.I1 = self.processFrame(I1)
        self.dummyInput2.copy_(self.I1)
        return True



        
