import numpy as np
import torch
import os

from torch.functional import F
from src.utils.downloadModels import downloadModels, weightsDir, modelsMap
from src.utils.coloredPrints import yellow
from src.utils.isCudaInit import CudaChecker
from src.utils.modelOptimizer import ModelOptimizer

checker = CudaChecker()


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


class FlownetSDedup:
    def __init__(
        self,
        half: bool = True,
        dedupSens: float = 0.9,
        height: int = 224,
        width: int = 224,
    ):
        print(
            yellow(
                "Flownet deduplication is experimental and may not work as expected."
            )
        )
        import src.dedup.flownet as flownet

        self.dedupSens = dedupSens
        self.half = half
        self.height = height
        self.width = width

        self.filename = modelsMap("flownets", modelType="pth")

        if not os.path.exists(os.path.join(weightsDir, "flownets", self.filename)):
            modelPath = downloadModels(
                model="flownets",
            )
        else:
            modelPath = os.path.join(weightsDir, "flownets", self.filename)

        self.model = torch.load(modelPath)
        self.model = flownet.__dict__[self.model["arch"]](self.model).to(checker.device)
        self.model.eval()

        self.model = ModelOptimizer(
            model=self.model,
            dtype=torch.float16 if half else torch.float32,
            memoryFormat=torch.channels_last,
        ).optimizeModel()

        if half:
            self.model.half()
        else:
            self.model.float()

        self.prevFrame = None
        self.mean = (
            torch.tensor(
                [0.411, 0.432, 0.45],
                device=checker.device,
                dtype=torch.float16 if self.half else torch.float32,
            )
            .view(1, 3, 1, 1)
            .to(checker.device)
            .to(memory_format=torch.channels_last)
        )
        self.std = (
            torch.tensor(
                [1, 1, 1],
                device=checker.device,
                dtype=torch.float16 if self.half else torch.float32,
            )
            .view(1, 3, 1, 1)
            .to(checker.device)
            .to(memory_format=torch.channels_last)
        )

        self.dummyInput = torch.zeros(
            (1, 6, self.height, self.width),
            device=checker.device,
            dtype=torch.float16 if self.half else torch.float32,
        ).to(memory_format=torch.channels_last)

        self.stream = torch.cuda.Stream()
        with torch.cuda.stream(self.stream):
            for _ in range(3):
                output = self.model(self.dummyInput)
                self.stream.synchronize()

        self.dummyOutput = torch.zeros(
            (1, 2, output.size(2), output.size(3)),
            device=checker.device,
            dtype=torch.float16 if self.half else torch.float32,
        ).to(memory_format=torch.channels_last)

        self.normStream = torch.cuda.Stream()
        self.outputStream = torch.cuda.Stream()

        self.cudaGraph = torch.cuda.CUDAGraph()
        self.initTorchCudaGraph()

    @torch.inference_mode()
    def initTorchCudaGraph(self):
        with torch.cuda.graph(self.cudaGraph, stream=self.stream):
            self.dummyOutput = self.model(self.dummyInput)
        self.stream.synchronize()

    @torch.inference_mode()
    def prepareFrame(self, frame):
        return ((frame - self.mean) / self.std).to(memory_format=torch.channels_last)

    @torch.inference_mode()
    def __call__(self, frame):
        if self.prevFrame is None:
            with torch.cuda.stream(self.normStream):
                self.prevFrame = frame
                self.prevFrame = self.prepareFrame(self.prevFrame)
            self.normStream.synchronize()
            return False

        with torch.cuda.stream(self.normStream):
            frame = self.prepareFrame(frame)
            self.dummyInput.copy_(torch.cat((self.prevFrame, frame), dim=1))
        self.normStream.synchronize()

        with torch.cuda.stream(self.stream):
            self.cudaGraph.replay()
        self.stream.synchronize()

        with torch.cuda.stream(self.outputStream):
            flow = self.dummyOutput
            self.prevFrame.copy_(frame, non_blocking=True)
        self.outputStream.synchronize()

        return flow.mean() > self.dedupSens
