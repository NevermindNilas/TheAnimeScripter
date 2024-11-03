import numpy as np
import torch

from torch.functional import F


class DedupSSIMCuda:
    def __init__(
        self,
        ssimThreshold=0.9,
        half=True,
        sampleSize=224,
    ):
        self.ssimThreshold = ssimThreshold
        self.sampleSize = sampleSize
        self.half = half
        self.prevFrame = None

        from .ssim import SSIM

        self.interpolate = F.interpolate
        self.ssim = SSIM(data_range=1.0, channel=3).cuda()
        if half:
            self.ssim.half()
        else:
            self.ssim.float()

    def __call__(self, frame):
        """
        Returns True if the frames are duplicates
        """
        if self.prevFrame is None:
            self.prevFrame = self.processFrame(frame)
            return False

        frame = self.processFrame(frame)

        score = self.ssim(self.prevFrame, frame).mean()
        self.prevFrame.copy_(frame, non_blocking=False)
        return score > self.ssimThreshold

    def processFrame(self, frame):
        return (
            F.interpolate(
                frame.half(),
                (self.sampleSize, self.sampleSize),
                mode="nearest",
            )
            if self.half
            else F.interpolate(
                frame.float(),
                (self.sampleSize, self.sampleSize),
                mode="nearest",
            )
        )


class DedupSSIM:
    def __init__(
        self,
        ssimThreshold=0.9,
        sampleSize=224,
    ):
        from torchmetrics import StructuralSimilarityIndexMeasure

        self.device = torch.device("cpu")
        self.ssimThreshold = ssimThreshold
        self.sampleSize = sampleSize
        self.prevFrame = None
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)

    def __call__(self, frame):
        """
        Returns True if the frames are duplicates
        """
        if self.prevFrame is None:
            self.prevFrame = self.processFrame(frame)
            return False

        frame = self.processFrame(frame)

        score = self.ssim(self.prevFrame, frame).item()
        self.prevFrame = frame

        return score > self.ssimThreshold

    def processFrame(self, frame):
        return torch.nn.functional.interpolate(
            frame.float(),
            size=(self.sampleSize, self.sampleSize),
            mode="bilinear",
            align_corners=False,
        ).to(self.device)

    def reset(self):
        self.prevFrame = None


class DedupMSE:
    def __init__(
        self,
        mseThreshold=1000,
        sampleSize=224,
    ):
        self.mseThreshold = mseThreshold
        self.sampleSize = sampleSize
        self.prevFrame = None

    def __call__(self, frame):
        """
        Returns True if the frames are duplicates
        """
        if self.prevFrame is None:
            self.prevFrame = self.processFrame(frame)
            return False

        frame = self.processFrame(frame)
        score = ((self.prevFrame - frame) ** 2).mean()

        self.prevFrame = frame.copy()

        return score < self.mseThreshold

    def processFrame(self, frame):
        return np.resize(
            frame.mul(255).cpu().numpy(), (self.sampleSize, self.sampleSize, 3)
        )


class DedupMSSSIMCuda:
    def __init__(
        self,
        ssimThreshold=0.9,
        sampleSize=224,
        half=True,
    ):
        self.ssimThreshold = ssimThreshold
        self.sampleSize = sampleSize
        self.half = half
        self.prevFrame = None

        from .ssim import MS_SSIM
        from torch.functional import F

        self.interpolate = F.interpolate
        self.ssim = MS_SSIM(data_range=1.0, channel=3).cuda()
        if half:
            self.ssim.half()
        else:
            self.ssim.float()

    def __call__(
        self,
        frame,
    ):
        """
        Returns True if the frames are duplicates
        """
        if self.prevFrame is None:
            self.prevFrame = self.processFrame(frame)
            return False

        frame = self.processFrame(frame)

        score = self.ssim(self.prevFrame, frame).mean()
        self.prevFrame.copy_(frame, non_blocking=True)
        return score > self.ssimThreshold

    def processFrame(self, frame):
        return (
            self.interpolate(
                frame.half().permute(2, 0, 1).unsqueeze(0),
                (self.sampleSize, self.sampleSize),
                mode="nearest",
            )
            if self.half
            else self.interpolate(
                frame.float().permute(2, 0, 1).unsqueeze(0),
                (self.sampleSize, self.sampleSize),
                mode="nearest",
            )
        )
