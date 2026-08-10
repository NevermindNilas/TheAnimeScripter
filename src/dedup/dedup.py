import torch
from torch.functional import F

from src.infra.isCudaInit import CudaChecker
from src.model.download import resolveWeightPath
from src.model.modelOptimizer import ModelOptimizer
from src.model.registry import modelsMap

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

        from frame_analytics import ssim

        self.interpolate = F.interpolate
        # frame_analytics accumulates in fp32/fp64 whatever the input dtype, so
        # `half` only picks the resize/compare dtype, never the score's precision.
        self.ssim = ssim

    def __call__(self, frame):
        """
        Returns True if the frames are duplicates
        """
        if self.prevFrame is None:
            self.prevFrame = self.processFrame(frame)
            return False

        frame = self.processFrame(frame)

        score = self.ssim(self.prevFrame, frame, data_range=1.0)

        if score < self.ssimThreshold:
            self.prevFrame.copy_(frame, non_blocking=False)
            return False
        else:
            return True

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
        from frame_analytics import ssim

        # Same fp32 SSIM math either way; on CUDA boxes the decoded frames are
        # already GPU-resident, and forcing CPU here shipped every frame D2H
        # and burned ~16 cores on the 224x224 convolutions while the GPU idled.
        self.device = checker.device if checker.cudaAvailable else torch.device("cpu")
        self.ssimThreshold = ssimThreshold
        self.sampleSize = sampleSize
        self.prevFrame = None
        self.ssim = ssim

    def __call__(self, frame):
        """
        Returns True if the frames are duplicates
        """
        if self.prevFrame is None:
            self.prevFrame = self.processFrame(frame)
            return False

        frame = self.processFrame(frame)

        score = self.ssim(self.prevFrame, frame, data_range=1.0).item()

        if score < self.ssimThreshold:
            self.prevFrame = frame
            return False
        else:
            return True

    def processFrame(self, frame):
        return torch.nn.functional.interpolate(
            frame.float(),
            size=(self.sampleSize, self.sampleSize),
            mode="bilinear",
            align_corners=False,
        ).to(self.device)


class DedupMSE:
    def __init__(
        self,
        mseThreshold=1000,
        sampleSize=224,
    ):
        from frame_analytics import mse

        self.mseThreshold = mseThreshold
        self.sampleSize = sampleSize
        self.prevFrame = None
        self.mse = mse

    def __call__(self, frame):
        """
        Returns True if the frames are duplicates
        """
        if self.prevFrame is None:
            self.prevFrame = self.processFrame(frame)
            return False

        frame = self.processFrame(frame)
        score = self.mse(self.prevFrame, frame)

        # Low MSE -> (near) identical to the previous kept frame -> duplicate.
        # NOTE: SSIM/VMAF treat a HIGH score as "similar"; MSE is the opposite
        # (0 == identical), so the comparison direction is inverted relative to
        # those backends. Keep the reference frame on a duplicate; advance it
        # only when we keep a distinct frame.
        if score < self.mseThreshold:
            return True
        else:
            self.prevFrame = frame
            return False

    def processFrame(self, frame):
        # np.resize flattens then tiles/truncates raw bytes (it does NOT scale
        # the image), which made the MSE comparison meaningless. Use a real
        # bilinear resize on the tensor and keep values on a 0-255 scale.
        return torch.nn.functional.interpolate(
            frame.float(),
            size=(self.sampleSize, self.sampleSize),
            mode="bilinear",
            align_corners=False,
        ).mul(255.0)


class DedupMSECuda:
    def __init__(
        self,
        mseThreshold=1000,
        half=True,
        sampleSize=224,
    ):
        from frame_analytics import mse

        self.mseThreshold = mseThreshold
        self.sampleSize = sampleSize
        self.half = half
        self.prevFrame = None
        self.interpolate = F.interpolate
        self.mse = mse

    def __call__(self, frame):
        """
        Returns True if the frames are duplicates
        """
        if self.prevFrame is None:
            self.prevFrame = self.processFrame(frame)
            return False

        frame = self.processFrame(frame)
        score = self.mse(self.prevFrame, frame)

        # Low MSE -> (near) identical -> duplicate (see DedupMSE for why the
        # direction is inverted vs SSIM/VMAF). Advance the reference only on a
        # distinct frame.
        if score < self.mseThreshold:
            return True
        else:
            self.prevFrame.copy_(frame, non_blocking=False)
            return False

    def processFrame(self, frame):
        return (
            F.interpolate(
                frame.half(),
                (self.sampleSize, self.sampleSize),
                mode="nearest",
            ).mul(255.0)
            if self.half
            else F.interpolate(
                frame.float(),
                (self.sampleSize, self.sampleSize),
                mode="nearest",
            ).mul(255.0)
        )


class DedupFlownetS:
    def __init__(
        self,
        half: bool = True,
        dedupSens: float = 0.9,
        height: int = 224,
        width: int = 224,
    ):
        import src.dedup.flownet as flownet

        self.dedupSens = dedupSens
        self.half = half
        self.height = height
        self.width = width

        self.filename = modelsMap("flownets", modelType="pth")

        modelPath = resolveWeightPath("flownets", self.filename)

        stateDict = torch.load(modelPath, map_location="cpu")
        self.model = flownet.__dict__[stateDict["arch"]](stateDict)
        del stateDict
        self.model = self.model.eval()

        self.model = ModelOptimizer(
            model=self.model,
            dtype=torch.float16 if half else torch.float32,
            memoryFormat=torch.channels_last,
        ).optimizeModel()

        if half:
            self.model = self.model.half()
        else:
            self.model = self.model.float()

        self.model = self.model.to(checker.device)
        if checker.cudaAvailable:
            torch.cuda.empty_cache()

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
                self.prevFrame = self.prepareFrame(frame)
            self.normStream.synchronize()
            return False

        with torch.cuda.stream(self.normStream):
            frame = self.prepareFrame(frame)
            self.dummyInput.copy_(
                torch.cat((self.prevFrame, frame), dim=1), non_blocking=True
            )
        self.normStream.synchronize()

        with torch.cuda.stream(self.stream):
            self.cudaGraph.replay()
        self.stream.synchronize()

        # FlowNetS outputs an optical-flow field (1, 2, H, W); duplicate frames
        # have little motion, so compare the mean flow magnitude (not the signed
        # mean) against the sensitivity threshold. `dummyOutput` is the CUDA
        # graph's output buffer, so read it before anything can replay.
        isDuplicate = bool(self.dummyOutput.abs().mean() < self.dedupSens)

        # Keep the reference frame on a duplicate; advance it only when we keep
        # a distinct frame -- the same contract DedupMSE/DedupSSIM state. This
        # used to advance unconditionally, so a slow pan or fade whose per-pair
        # flow stayed under the threshold was dropped frame after frame: the
        # comparison always reset to the immediately preceding frame instead of
        # the last kept one, and the whole move vanished from the output.
        if not isDuplicate:
            with torch.cuda.stream(self.outputStream):
                self.prevFrame.copy_(frame, non_blocking=True)
            self.outputStream.synchronize()

        return isDuplicate


class DedupVMAF:
    def __init__(
        self,
        dedupMethod="vmaf",
        treshold=90,
        sampleSize=224,
        half=True,
    ):
        self.treshold = treshold
        self.sampleSize = sampleSize
        self.half = half
        self.prevFrame = None
        self.isCuda = "cuda" in dedupMethod

        from torch.nn import functional as F
        from vmaf_torch import VMAF

        self.interpolate = F.interpolate

        if self.isCuda:
            self.vmaf = VMAF().cuda().float()
        else:
            self.vmaf = VMAF().float()

    def __call__(self, frame):
        """
        Returns True if the frames are duplicates
        """
        if self.prevFrame is None:
            self.prevFrame = self.processFrame(frame)
            return False

        frame = self.processFrame(frame)

        score = self.vmaf(self.prevFrame, frame).mean()

        if score < self.treshold:
            self.prevFrame.copy_(frame, non_blocking=True)
            return False
        else:
            return True

    def processFrame(self, frame):
        if not self.isCuda:
            frame = frame.cpu()
        resized = (
            self.interpolate(
                frame.half(),
                (self.sampleSize, self.sampleSize),
                mode="bilinear",
                align_corners=False,
            )
            if self.half
            else self.interpolate(
                frame.float(),
                (self.sampleSize, self.sampleSize),
                mode="bilinear",
                align_corners=False,
            )
        )
        return self.to_y(resized).float() * 255.0

    def to_y(self, tensor):
        if tensor.shape[1] == 3:
            return (
                0.299 * tensor[:, 0:1] + 0.587 * tensor[:, 1:2] + 0.114 * tensor[:, 2:3]
            )
        return tensor
