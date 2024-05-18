import os
import torch
import logging
from torch.nn import functional as F
from .downloadModels import downloadModels, weightsDir, modelsMap
from .coloredPrints import yellow

# Apparently this can improve performance slightly
torch.set_float32_matmul_precision("medium")


class RifeCuda:
    def __init__(
        self,
        half,
        width,
        height,
        UHD,
        interpolateMethod,
        ensemble=False,
        nt=1,
        interpolateFactor=2,
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
        """
        self.half = half
        self.UHD = UHD
        self.scale = 1.0
        self.width = width
        self.height = height
        self.interpolateMethod = interpolateMethod
        self.ensemble = ensemble
        self.nt = nt
        self.interpolateFactor = interpolateFactor

        if self.UHD:
            self.scale = 0.5

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

        match self.interpolateMethod:
            case "rife" | "rife4.15":
                from .rifearches.IFNet_rife415 import IFNet
            case "rife4.15-lite":
                from .rifearches.IFNet_rife415lite import IFNet
            case "rife4.16-lite":
                from .rifearches.IFNet_rife416lite import IFNet
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

        self.scaleList = [
            8 / self.scale,
            4 / self.scale,
            2 / self.scale,
            1 / self.scale,
        ]

        self.I0 = torch.zeros(
            1,
            3,
            self.height + self.padding[3],
            self.width + self.padding[1],
            dtype=torch.float16
            if self.half and self.isCudaAvailable
            else torch.float32,
        ).to(self.device)

        self.I1 = torch.zeros(
            1,
            3,
            self.height + self.padding[3],
            self.width + self.padding[1],
            dtype=torch.float16
            if self.half and self.isCudaAvailable
            else torch.float32,
        ).to(self.device)

        self.firstRun = True

        if self.interpolateFactor == 2:
            self.timestep = (
                torch.zeros(
                    1,
                    1,
                    self.I0.shape[2],
                    self.I0.shape[3],
                    dtype=torch.float16 if self.half else torch.float32,
                ).to(self.device)
                * 0.5
            )

    @torch.inference_mode()
    def make_inference(self, timestep):
        if self.interpolateFactor != 2:
            self.timestep = (
                torch.zeros(
                    1,
                    1,
                    self.I0.shape[2],
                    self.I0.shape[3],
                    dtype=torch.float16 if self.half else torch.float32,
                ).to(self.device)
                * timestep
            )

        output = self.model(
            self.I0, self.I1, self.timestep, self.scaleList, self.ensemble
        )
        output = output[:, :, : self.height, : self.width]

        return output.mul_(255.0).squeeze(0).permute(1,2,0)
    
    def cacheFrame(self):
        self.I0.copy_(self.I1, non_blocking=True)

    @torch.inference_mode()
    def processFrame(self, frame):
        if self.half:
            frame = (
                frame
                .to(self.device, non_blocking=True)
                .permute(2, 0, 1)
                .unsqueeze_(0)
                .half()
                .mul_(1 / 255)
            )
        else:
            frame = (
                frame
                .to(self.device, non_blocking=True)
                .permute(2, 0, 1)
                .unsqueeze_(0)
                .float()
                .mul_(1 / 255)
            )

        if self.padding != (0, 0, 0, 0):
            frame = F.pad(frame, [0, self.padding[1], 0, self.padding[3]])

        return frame

    @torch.inference_mode()
    def run(self, I1):
        if self.firstRun is True:
            self.I0.copy_(self.processFrame(I1), non_blocking=True)
            self.firstRun = False
            return False

        self.I1.copy_(self.processFrame(I1), non_blocking=True)
        return True

class RifeTensorRT:
    def __init__(
        self,
        interpolateMethod: str = "rife4.15",
        interpolateFactor: int = 2,
        width: int = 0,
        height: int = 0,
        half: bool = True,
        ensemble: bool = False,
        nt: int = 1,
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
        from polygraphy.backend.trt import (
            TrtRunner,
            engine_from_network,
            network_from_onnx_path,
            CreateConfig,
            Profile,
            EngineFromBytes,
            SaveEngine,
        )
        from polygraphy.backend.common import BytesFromPath

        self.TrtRunner = TrtRunner
        self.engine_from_network = engine_from_network
        self.network_from_onnx_path = network_from_onnx_path
        self.CreateConfig = CreateConfig
        self.Profile = Profile
        self.EngineFromBytes = EngineFromBytes
        self.SaveEngine = SaveEngine
        self.BytesFromPath = BytesFromPath

        self.interpolateMethod = interpolateMethod
        self.interpolateFactor = interpolateFactor
        self.width = width
        self.height = height
        self.half = half
        self.ensemble = ensemble
        self.nt = nt
        self.model = None

        self.handleModel()

    def handleModel(self):
        self.filename = modelsMap(
            self.interpolateMethod,
            modelType="onnx",
            half=self.half,
            ensemble=self.ensemble,
        )

        if not os.path.exists(
            os.path.join(weightsDir, self.interpolateMethod, self.filename)
        ):
            modelPath = downloadModels(
                model=self.interpolateMethod,
                modelType="onnx",
                half=self.half,
                ensemble=self.ensemble,
            )
        else:
            modelPath = os.path.join(weightsDir, self.interpolateMethod, self.filename)

        self.isCudaAvailable = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.isCudaAvailable else "cpu")
        if self.isCudaAvailable:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            if self.half:
                torch.set_default_dtype(torch.float16)

        if "fp16" in modelPath:
            trtEngineModelPath = modelPath.replace(".onnx", "_fp16.engine")
        elif "fp32" in modelPath:
            trtEngineModelPath = modelPath.replace(".onnx", "_fp32.engine")
        else:
            trtEngineModelPath = modelPath.replace(".onnx", ".engine")

        if os.path.exists(trtEngineModelPath):
            self.engine = self.EngineFromBytes(self.BytesFromPath(trtEngineModelPath))
        else:
            toPrint = f"Engine not found, creating dynamic engine for model: {modelPath}, this may take a while, but it is worth the wait..."
            print(yellow(toPrint))
            logging.info(toPrint)

            profile = [
                self.Profile().add(
                    "input",
                    min=(1, 7, 32, 32),
                    opt=(1, 7, self.height, self.width),
                    max=(1, 7, 2160, 3840),
                )
            ]
            self.config = self.CreateConfig(
                fp16=self.half, bf16=self.half, profiles=profile, preview_features=[],
            )

            self.engine = self.engine_from_network(
                self.network_from_onnx_path(modelPath),
                config=self.config,
            )
            self.engine = self.SaveEngine(
                self.engine, modelPath.replace(".onnx", ".engine")
            )

        self.runner = self.TrtRunner(self.engine)
        self.runner.activate()

        self.dType = torch.float16 if self.half else torch.float32

        self.I0 = torch.zeros(
            1,
            3,
            self.height,
            self.width,
            dtype=self.dType,
            device=self.device,
        )
        self.timestep = (
            torch.zeros(
                1,
                1,
                self.height,
                self.width,
                dtype=self.dType,
                device=self.device,
            )
            * 0.5
        )

        scaleInt = 1 if self.width < 3840 and self.height < 2160 else 0.5
        self.scale = torch.zeros(
            1,
            1,
            self.height,
            self.width,
            dtype=self.dType,
            device=self.device,
        ) * scaleInt
        
        self.I0 = None

    @torch.inference_mode()
    def make_inference(self, n):
        if self.interpolateFactor != 2:
            self.timestep = (
                torch.zeros(
                    1,
                    1,
                    self.height,
                    self.width,
                    dtype=self.dType,
                    device=self.device,
                )
                * n
            )
        
        return (
            self.runner.infer(
                {
                    "input": torch.cat(
                        [
                            self.I0,
                            self.I1,
                            self.timestep,
                        ],
                        dim=1,
                    )
                },
                check_inputs=False,
            )["output"]
            .squeeze(0)
            .permute(1, 2, 0)
            .mul_(255)
        )

    @torch.inference_mode()
    def cacheFrame(self):
        self.I0 = self.I1.clone()

    @torch.inference_mode()
    def processFrame(self, frame):
        # Is this what they mean with Pythonic Code?
        return (
            (
                frame
                .to(self.device, non_blocking=True)
                .permute(2, 0, 1)
                .unsqueeze_(0)
                .float()
                if not self.half
                else frame
                .to(self.device, non_blocking=True)
                .permute(2, 0, 1)
                .unsqueeze_(0)
                .half()
            )
            .mul_(1 / 255)
            .contiguous()
        )

    @torch.inference_mode()
    def run(self, I1):
        if self.I0 is None:
            self.I0 = self.processFrame(I1)
            return False

        self.I1 = self.processFrame(I1)
        return True
