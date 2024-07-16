import os
import torch
import logging
import cupy

from src.downloadModels import downloadModels, weightsDir
from torch.nn import functional as F
from src.coloredPrints import yellow, red

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
        nt=1,
        sceneChange=False,
        sceneChangeThreshold=0.85,
    ):
        self.width = width
        self.height = height
        self.half = half
        self.interpolation_factor = interpolation_factor
        self.ensemble = ensemble
        self.nt = nt
        self.sceneChange = sceneChange
        self.sceneChangeThreshold = sceneChangeThreshold

        ph = ((self.height - 1) // 32 + 1) * 32
        pw = ((self.width - 1) // 32 + 1) * 32
        self.padding = (0, pw - self.width, 0, ph - self.height)

        if self.width > 1920 or self.height > 1080:
            print(
                yellow(
                    "Warning: Output Resolution is higher than 1080p. Expect significant slowdowns or no functionality at all due to VRAM Constraints when using GMFSS, in case of issues consider switching to RIFE."
                )
            )
            self.scale = 0.5
        else:
            self.scale = 1

        self.handle_model()

    def handle_model(self):
        if not os.path.exists(os.path.join(weightsDir, "gmfss")):
            modelDir = os.path.dirname(downloadModels("gmfss"))
        else:
            modelDir = os.path.join(weightsDir, "gmfss")

        modelType = "union"

        self.isCudaAvailable = torch.cuda.is_available()

        if not self.isCudaAvailable:
            toPrint = "CUDA is not available, using CPU. Expect significant slowdows or no functionality at all. If you have a NVIDIA GPU, please install CUDA and make sure that CUDA_Path is in the environment variables. CUDA Installation link: https://developer.nvidia.com/cuda-downloads"
            print(toPrint)
            logging.warning(toPrint)

        if cupy.cuda.get_cuda_path() is None:
            toPrint = "Couldn't find relevant CUDA installation. Please install CUDA TOOLKIT from: https://developer.nvidia.com/cuda-downloads and try again."
            print(red(toPrint))
            logging.error(toPrint)

        self.device = torch.device("cuda" if self.isCudaAvailable else "cpu")

        torch.set_grad_enabled(False)
        if self.isCudaAvailable:
            # self.stream = [torch.cuda.Stream() for _ in range(self.nt)]
            # self.current_stream = 0
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            if self.half:
                torch.set_default_tensor_type(torch.cuda.HalfTensor)

        from .model.GMFSS import GMFSS as Model

        self.model = Model(modelDir, modelType, self.scale, ensemble=self.ensemble)
        self.model.eval().to(self.device, memory_format=torch.channels_last)

        self.dtype = torch.float
        if self.isCudaAvailable and self.half:
            self.model.half()
            self.dtype = torch.half

        self.I0 = torch.zeros(
            1,
            3,
            self.height + self.padding[3],
            self.width + self.padding[1],
            dtype=torch.float16 if self.half else torch.float32,
            device=self.device,
        )

        self.I1 = torch.zeros(
            1,
            3,
            self.height + self.padding[3],
            self.width + self.padding[1],
            dtype=torch.float16 if self.half else torch.float32,
            device=self.device,
        )

        self.stream = torch.cuda.Stream()
        self.firstRun = True

        if self.sceneChange:
            from src.unifiedInterpolate import SceneChange
            self.sceneChangeProcess = SceneChange(self.half, self.sceneChangeThreshold)

    @torch.inference_mode()
    def make_inference(self, n):
        """
        if self.isCudaAvailable:
            torch.cuda.set_stream(self.stream[self.current_stream])
        """

        timestep = torch.tensor(
            (n + 1) * 1.0 / (self.interpolation_factor + 1),
            dtype=self.dtype,
            device=self.device,
        )
        output = self.model(self.I0, self.I1, timestep)

        # if self.isCudaAvailable:
        # torch.cuda.synchronize(self.stream[self.current_stream])
        # self.current_stream = (self.current_stream + 1) % len(self.stream)

        if self.padding != (0, 0, 0, 0):
            output = output[..., : self.height, : self.width]

        return output.squeeze(0).permute(1, 2, 0).mul_(255)
    
    @torch.inference_mode()
    def cacheFrame(self):
        self.I0.copy_(self.I1, non_blocking=True)

    @torch.inference_mode()
    def processFrame(self, frame):
        return (
            (
                frame.to(self.device).permute(2, 0, 1).unsqueeze(0).float()
                if not self.half
                else frame.to(self.device).permute(2, 0, 1).unsqueeze(0).half()
            )
            .mul(1 / 255)
            .contiguous()
        )

    @torch.inference_mode()
    def padFrame(self, frame):
        return (
            F.pad(frame, [0, self.padding[1], 0, self.padding[3]])
            if self.padding != (0, 0, 0, 0)
            else frame
        )

    @torch.inference_mode()
    def run(self, frame, interpolateFactor, writeBuffer):
        with torch.cuda.stream(self.stream):
            if self.firstRun is True:
                self.I0 = self.processFrame(frame)
                self.I0 = self.padFrame(self.I0)
                self.firstRun = False
                return
            self.I1 = self.processFrame(frame)
            self.I1 = self.padFrame(self.I1)

            if self.sceneChange:
                if self.sceneChangeProcess.run(self.I0, self.I1):
                    for _ in range(interpolateFactor - 1):
                        writeBuffer.write(frame)
                    self.cacheFrame()
                    self.stream.synchronize()
                    return

            for i in range(interpolateFactor - 1):
                timestep = torch.tensor(
                    (i + 1) * 1.0 / self.interpolation_factor,
                    dtype=self.dtype,
                    device=self.device,
                )
                output = self.model(self.I0, self.I1, timestep)
                output = output[:, :, : self.height, : self.width]
                output = output.mul(255.0).squeeze(0).permute(1, 2, 0)
                self.stream.synchronize()
                writeBuffer.write(output)
            
            self.cacheFrame()

