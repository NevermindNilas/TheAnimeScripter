import numpy as np

class DedupSSIMCuda:
    def __init__(
        self,
        ssimThreshold=0.9,
        sampleSize=224,
        half=True,
    ):
        """
        A Cuda accelerated version of the SSIM deduplication method

        Args:
            ssimThreshold: float, SSIM threshold to consider two frames as duplicates
            sampleSize: int, size of the frame to be used for comparison
            half: bool, use half precision for the comparison
        """
        self.ssimThreshold = ssimThreshold
        self.sampleSize = sampleSize
        self.prevFrame = None
        self.half = half

        import torch
        import torch.nn.functional as F
        from torchmetrics.image import StructuralSimilarityIndexMeasure

        self.torch = torch
        self.F = F
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.DEVICE)

    def run(self, frame):
        """
        Returns True if the frames are duplicates
        """
        if self.prevFrame is None:
            self.prevFrame = self.processFrame(frame)
            return False

        frame = self.processFrame(frame)
        similarity = self.ssim(self.prevFrame, frame).item()
        self.prevFrame = frame.clone()

        return similarity > self.ssimThreshold

    def processFrame(self, frame):
        frame = (
            frame
            .to(self.DEVICE)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            if not self.half
            else frame
            .to(self.DEVICE)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .half()
        ).mul(1 / 255.0)
        frame = self.F.interpolate(
            frame, (self.sampleSize, self.sampleSize), mode="nearest"
        )
        return frame

class DedupSSIM:
    def __init__(
        self,
        ssimThreshold=0.9,
        sampleSize=224,
    ):
        self.ssimThreshold = ssimThreshold
        self.sampleSize = sampleSize
        self.prevFrame = None

        from skimage.metrics import structural_similarity as ssim
        from skimage import color

        self.ssim = ssim
        self.color = color

    def run(self, frame):
        """
        Returns True if the frames are duplicates
        """
        if self.prevFrame is None:
            self.prevFrame = self.processFrame(frame)
            return False

        frame = self.processFrame(frame)

        score = self.ssim(self.prevFrame, frame, data_range=frame.max() - frame.min())
        self.prevFrame = frame.copy()

        return score > self.ssimThreshold

    def processFrame(self, frame):
        frame = frame.cpu().numpy()
        frame = np.resize(frame, (self.sampleSize, self.sampleSize, 3))
        frame = self.color.rgb2gray(frame)

        return frame


class DedupMSE:
    def __init__(
        self,
        mseThreshold=1000,
        sampleSize=224,
    ):
        self.mseThreshold = mseThreshold
        self.sampleSize = sampleSize
        self.prevFrame = None

        from skimage.metrics import mean_squared_error as mse
        from skimage import color

        self.mse = mse
        self.color = color

    def run(self, frame):
        """
        Returns True if the frames are duplicates
        """
        if self.prevFrame is None:
            self.prevFrame = self.processFrame(frame)
            return False

        frame = self.processFrame(frame)

        score = self.mse(self.prevFrame, frame)
        self.prevFrame = frame.copy()

        return score < self.mseThreshold

    def processFrame(self, frame):
        frame = frame.cpu().numpy()
        frame = np.resize(frame, (self.sampleSize, self.sampleSize, 3))
        frame = self.color.rgb2gray(frame)

        return frame