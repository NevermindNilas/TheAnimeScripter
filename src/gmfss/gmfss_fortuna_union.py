import os
import torch
import logging
import cupy

from src.downloadModels import downloadModels, weightsDir
from torch.nn import functional as F

torch.set_float32_matmul_precision("medium")
# from: https://github.com/HolyWu/vs-gmfss_fortuna/blob/master/vsgmfss_fortuna/__init__.py


class GMFSS:
    def __init__(
        self, interpolation_factor, half, width, height, UHD, ensemble=False, nt=1
    ):
        self.width = width
        self.height = height
        self.half = half
        self.interpolation_factor = interpolation_factor
        self.UHD = UHD
        self.ensemble = ensemble
        self.nt = nt

        ph = ((self.height - 1) // 32 + 1) * 32
        pw = ((self.width - 1) // 32 + 1) * 32
        self.padding = (0, pw - self.width, 0, ph - self.height)

        if self.UHD:
            self.scale = 0.5
        else:
            self.scale = 1.0

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
        try:
            cupy.cuda.get_cuda_path()
        except Exception:
            toPrint = "Couldn't find relevant CUDA installation. Please make sure that CUDA_Path is in the environment variables. CUDA Installation link: https://developer.nvidia.com/cuda-downloads"
            print(toPrint)
            logging.error(toPrint)

        self.device = torch.device("cuda" if self.isCudaAvailable else "cpu")

        torch.set_grad_enabled(False)
        if self.isCudaAvailable:
            #self.stream = [torch.cuda.Stream() for _ in range(self.nt)]
            #self.current_stream = 0
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
            dtype=torch.float16
            if self.half
            else torch.float32,
            device=self.device,
        )

        self.I1 = torch.zeros(
            1,
            3,
            self.height + self.padding[3],
            self.width + self.padding[1],
            dtype=torch.float16
            if self.half
            else torch.float32,
            device=self.device,
        )

        self.stream = torch.cuda.Stream()
        self.firstRun = True

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

        #if self.isCudaAvailable:
            #torch.cuda.synchronize(self.stream[self.current_stream])
            #self.current_stream = (self.current_stream + 1) % len(self.stream)

        if self.padding != (0, 0, 0, 0):
            output = output[..., : self.height, : self.width]
        
        return output.squeeze(0).permute(1, 2, 0).mul_(255)

    @torch.inference_mode()
    def processFrame(self, frame):
        return (
            (
                frame.to(self.device)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                if not self.half
                else frame.to(self.device)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .half()
            )
            .mul(1 / 255)
            .contiguous()
        )

    @torch.inference_mode()
    def padFrame(self, frame):
        return F.pad(frame, [0, self.padding[1], 0, self.padding[3]]) if self.padding != (0, 0, 0, 0) else frame

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

            for i in range(interpolateFactor - 1):
                timestep = torch.tensor(
                    (i + 1) * 1.0 / self.interpolation_factor,
                    dtype=self.dtype,
                    device=self.device,
                )
                output = self.model(
                        self.I0, self.I1, timestep
                )
                output = output[:, :, : self.height, : self.width]
                output = output.mul(255.0).squeeze(0).permute(1, 2, 0)
                self.stream.synchronize()
                writeBuffer.write(output)

            self.I0.copy_(self.I1, non_blocking=True)
