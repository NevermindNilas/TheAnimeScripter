import os
import torch

from src.model.download import resolveWeightPath
from src.model.registry import modelsMap
from src.infra.isCudaInit import CudaChecker
from src.constants import ADOBE


if ADOBE:
    from src.server.aeComms import progressState

checker = CudaChecker()

torch.set_float32_matmul_precision("medium")


def calculatePadding(width, height, multiple=4):
    padW = (multiple - (width % multiple)) % multiple
    padH = (multiple - (height % multiple)) % multiple
    return (0, padW, 0, padH)


class UniversalNCNN:
    def __init__(self, upscaleMethod, upscaleFactor):
        self.upscaleMethod = upscaleMethod
        self.upscaleFactor = upscaleFactor

        from upscale_ncnn_py import UPSCALE

        self.filename = modelsMap(
            self.upscaleMethod,
            modelType="ncnn",
        )

        if self.filename.endswith("-ncnn.zip"):
            self.filename = self.filename[:-9]
        elif self.filename.endswith("-ncnn"):
            self.filename = self.filename[:-5]

        modelPath = resolveWeightPath(
            self.upscaleMethod,
            self.filename,
            modelType="ncnn",
        )

        if modelPath.endswith("-ncnn.zip"):
            modelPath = modelPath[:-9]
        elif modelPath.endswith("-ncnn"):
            modelPath = modelPath[:-5]

        lastSlash = os.path.basename(modelPath)
        modelPath = os.path.join(modelPath, lastSlash)

        self.model = UPSCALE(
            gpuid=0,
            tta_mode=False,
            tilesize=0,
            model_str=modelPath,
            num_threads=2,
        )

    def __call__(self, frame, nextFrame: None) -> torch.tensor:
        iniFrameDtype = frame.dtype
        frame = self.model.process_torch(
            frame.mul(255).clamp(0, 255).to(torch.uint8).squeeze(0).permute(1, 2, 0).cpu()
        )

        frame = frame.to(iniFrameDtype).mul(1 / 255).permute(2, 0, 1).unsqueeze(0)
        return frame

