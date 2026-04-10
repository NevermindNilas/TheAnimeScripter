import os
import torch
import logging
from contextlib import nullcontext

from src.utils.downloadModels import downloadModels, weightsDir
from src.utils.isCudaInit import CudaChecker
from src.utils.logAndPrint import logAndPrint

checker = CudaChecker()

torch.set_float32_matmul_precision("medium")
# from: https://github.com/HolyWu/vs-gmfss_fortuna/blob/master/vsgmfss_fortuna/__init__.py  


def get_gmfss_model_dir():
    if not os.path.exists(os.path.join(weightsDir, "gmfss")):
        return os.path.dirname(downloadModels("gmfss"))
    return os.path.join(weightsDir, "gmfss")


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
        modelDir = get_gmfss_model_dir()
        self.device = torch.device("cuda" if checker.cudaAvailable else "cpu")
        self.use_cuda_stream = self.device.type == "cuda"

        torch.set_grad_enabled(False)
        if self.use_cuda_stream:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

        from .model.GMFSS import GMFSS as Model

        self.model = Model(modelDir, self.scale, ensemble=self.ensemble)
        self.model.eval().to(self.device, memory_format=torch.channels_last)

        self.dtype = torch.float32
        if self.use_cuda_stream and self.half:
            self.model.half()
            self.dtype = torch.float16

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
                    f"Error compiling model gmfss with mode {self.compileMode}: {e}"
                )
                logAndPrint(
                    f"Error compiling model gmfss with mode {self.compileMode}: {e}",
                    "red",
                )
                self.compileMode = "default"

        self.I0 = torch.zeros(
            1,
            3,
            self.ph,
            self.pw,
            dtype=self.dtype,
            device=self.device,
        ).to(memory_format=torch.channels_last)

        self.I1 = torch.zeros(
            1,
            3,
            self.ph,
            self.pw,
            dtype=self.dtype,
            device=self.device,
        ).to(memory_format=torch.channels_last)

        self._timestep_buffer = torch.empty(1, dtype=self.dtype, device=self.device)
        self.stream = torch.cuda.Stream() if self.use_cuda_stream else None
        self.firstRun = True

    def stream_context(self):
        if self.stream is None:
            return nullcontext()
        return torch.cuda.stream(self.stream)

    @torch.inference_mode()
    def validateFrame(self, frame):
        if frame.ndim != 4:
            raise ValueError(
                f"GMFSS expects a 4D frame tensor shaped [N, C, H, W], got {tuple(frame.shape)}"
            )
        if frame.shape[0] != 1 or frame.shape[1] != 3:
            raise ValueError(
                f"GMFSS expects a frame tensor shaped [1, 3, H, W], got {tuple(frame.shape)}"
            )
        if frame.shape[2] != self.height or frame.shape[3] != self.width:
            raise ValueError(
                f"GMFSS was initialized for {self.width}x{self.height} frames but received {frame.shape[3]}x{frame.shape[2]}"
            )

    @torch.inference_mode()
    def cacheFrame(self):
        self.I0.copy_(self.I1, non_blocking=self.use_cuda_stream)

    @torch.inference_mode()
    def processFrame(self, frame):
        self.validateFrame(frame)
        return frame.to(
            self.device,
            non_blocking=self.use_cuda_stream,
            dtype=self.dtype,
        ).to(memory_format=torch.channels_last)

    @torch.inference_mode()
    def stageFrame(self, frame, target):
        frame = self.processFrame(frame)

        if self.pw == self.width and self.ph == self.height:
            target.copy_(frame, non_blocking=self.use_cuda_stream)
            return

        target.zero_()
        target[:, :, : self.height, : self.width].copy_(
            frame, non_blocking=self.use_cuda_stream
        )

    @torch.inference_mode()
    def __call__(self, frame, interpQueue, framesToInsert: int = 2, timesteps=None):
        with self.stream_context():
            if self.firstRun is True:
                self.stageFrame(frame, self.I0)
                self.firstRun = False
                return

            self.stageFrame(frame, self.I1)
            reuse_cache = self.model.reuse(self.I0, self.I1)
            outputs = []

            for i in range(framesToInsert):
                if timesteps is not None and i < len(timesteps):
                    t = timesteps[i]
                else:
                    t = (i + 1) * 1 / (framesToInsert + 1)

                timestep = float(t)
                if not 0.0 <= timestep <= 1.0:
                    raise ValueError(f"GMFSS timestep must be within [0, 1], got {timestep}")
                self._timestep_buffer.fill_(timestep)
                outputs.append(
                    self.model.forward_from_reuse(reuse_cache, self._timestep_buffer)[
                        :, :, : self.height, : self.width
                    ]
                )

            if self.stream is not None:
                self.stream.synchronize()

            for output in outputs:
                interpQueue.put(output)

            self.cacheFrame()

