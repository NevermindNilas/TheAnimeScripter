import torch
from torch.functional import F

from src.infra.cudaStreams import waitForCaller
from src.infra.isCudaInit import CudaChecker
from src.model.download import resolveWeightPath
from src.model.modelOptimizer import ModelOptimizer
from src.model.registry import modelsMap

checker = CudaChecker()


class ReusableFrameScore:
    """Score and preprocessed current frame from a proven pair comparison."""

    __slots__ = (
        "family",
        "sampleSize",
        "dtype",
        "device",
        "resizeMode",
        "alignCorners",
        "scale",
        "score",
        "currentFrame",
    )

    def __init__(
        self,
        *,
        family: str,
        sampleSize: int,
        dtype,
        device,
        resizeMode: str,
        alignCorners,
        scale: float,
        score: float,
        currentFrame,
    ) -> None:
        self.family = family
        self.sampleSize = sampleSize
        self.dtype = dtype
        self.device = device
        self.resizeMode = resizeMode
        self.alignCorners = alignCorners
        self.scale = scale
        self.score = score
        self.currentFrame = currentFrame


def _scoreToFloat(score) -> float:
    return score.item() if hasattr(score, "item") else float(score)


def _reusableScore(
    *,
    family: str,
    sampleSize: int,
    resizeMode: str,
    alignCorners,
    scale: float,
    score,
    currentFrame,
) -> ReusableFrameScore:
    return ReusableFrameScore(
        family=family,
        sampleSize=sampleSize,
        dtype=currentFrame.dtype,
        device=currentFrame.device,
        resizeMode=resizeMode,
        alignCorners=alignCorners,
        scale=scale,
        score=_scoreToFloat(score),
        currentFrame=currentFrame,
    )


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
        self.lastReusableScore = None

        from frame_analytics import ssim

        self.interpolate = F.interpolate
        # frame_analytics accumulates in fp32/fp64 whatever the input dtype, so
        # `half` only picks the resize/compare dtype, never the score's precision.
        self.ssim = ssim

    def __call__(self, frame):
        """
        Returns True if the frames are duplicates
        """
        self.lastReusableScore = None
        if self.prevFrame is None:
            self.prevFrame = self.processFrame(frame)
            return False

        frame = self.processFrame(frame)

        score = self.ssim(self.prevFrame, frame, data_range=1.0)
        self.lastReusableScore = _reusableScore(
            family="ssim",
            sampleSize=self.sampleSize,
            resizeMode="nearest",
            alignCorners=None,
            scale=1.0,
            score=score,
            currentFrame=frame,
        )

        if self.lastReusableScore.score < self.ssimThreshold:
            self.prevFrame.copy_(frame, non_blocking=False)
            return False
        else:
            self.lastReusableScore = None
            return True

    def processFrame(self, frame):
        # Nearest sampling only selects source pixels, so it commutes with the
        # pointwise dtype cast: resize in the source dtype, then cast the small
        # 224px result instead of casting the full-res frame first. Bilinear
        # paths are excluded on purpose -- their interpolation arithmetic does
        # NOT commute across dtypes, so the same reorder would shift scores.
        resized = self.interpolate(
            frame,
            (self.sampleSize, self.sampleSize),
            mode="nearest",
        )
        return resized.half() if self.half else resized.float()


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
        self.lastReusableScore = None
        self.ssim = ssim

    def __call__(self, frame):
        """
        Returns True if the frames are duplicates
        """
        self.lastReusableScore = None
        if self.prevFrame is None:
            self.prevFrame = self.processFrame(frame)
            return False

        frame = self.processFrame(frame)

        score = self.ssim(self.prevFrame, frame, data_range=1.0)
        self.lastReusableScore = _reusableScore(
            family="ssim",
            sampleSize=self.sampleSize,
            resizeMode="bilinear",
            alignCorners=False,
            scale=1.0,
            score=score,
            currentFrame=frame,
        )

        if self.lastReusableScore.score < self.ssimThreshold:
            self.prevFrame = frame
            return False
        else:
            self.lastReusableScore = None
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
        self.lastReusableScore = None
        self.mse = mse

    def __call__(self, frame):
        """
        Returns True if the frames are duplicates
        """
        self.lastReusableScore = None
        if self.prevFrame is None:
            self.prevFrame = self.processFrame(frame)
            return False

        frame = self.processFrame(frame)
        score = self.mse(self.prevFrame, frame)
        self.lastReusableScore = _reusableScore(
            family="mse",
            sampleSize=self.sampleSize,
            resizeMode="bilinear",
            alignCorners=False,
            scale=255.0,
            score=score,
            currentFrame=frame,
        )

        # Low MSE -> (near) identical to the previous kept frame -> duplicate.
        # NOTE: SSIM/VMAF treat a HIGH score as "similar"; MSE is the opposite
        # (0 == identical), so the comparison direction is inverted relative to
        # those backends. Keep the reference frame on a duplicate; advance it
        # only when we keep a distinct frame.
        if self.lastReusableScore.score < self.mseThreshold:
            self.lastReusableScore = None
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
        self.lastReusableScore = None
        self.interpolate = F.interpolate
        self.mse = mse

    def __call__(self, frame):
        """
        Returns True if the frames are duplicates
        """
        self.lastReusableScore = None
        if self.prevFrame is None:
            self.prevFrame = self.processFrame(frame)
            return False

        frame = self.processFrame(frame)
        score = self.mse(self.prevFrame, frame)
        self.lastReusableScore = _reusableScore(
            family="mse",
            sampleSize=self.sampleSize,
            resizeMode="nearest",
            alignCorners=None,
            scale=255.0,
            score=score,
            currentFrame=frame,
        )

        # Low MSE -> (near) identical -> duplicate (see DedupMSE for why the
        # direction is inverted vs SSIM/VMAF). Advance the reference only on a
        # distinct frame.
        if self.lastReusableScore.score < self.mseThreshold:
            self.lastReusableScore = None
            return True
        else:
            self.prevFrame.copy_(frame, non_blocking=False)
            return False

    def processFrame(self, frame):
        # Same downsample-first reorder as DedupSSIMCuda: nearest sampling
        # commutes with the pointwise cast, so resize in the source dtype and
        # cast the small 224px result; the x255 scale stays last as before.
        resized = self.interpolate(
            frame,
            (self.sampleSize, self.sampleSize),
            mode="nearest",
        )
        return resized.half().mul(255.0) if self.half else resized.float().mul(255.0)


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
        waitForCaller(self.stream)
        # no_grad: a grad-mode warmup keeps every activation for a backward
        # that never runs (see UniversalPytorch's warmup).
        with torch.cuda.stream(self.stream), torch.no_grad():
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
            waitForCaller(self.normStream)
            with torch.cuda.stream(self.normStream):
                self.prevFrame = self.prepareFrame(frame)
            self.normStream.synchronize()
            return False

        waitForCaller(self.normStream)
        with torch.cuda.stream(self.normStream):
            frame = self.prepareFrame(frame)
            self.dummyInput.copy_(
                torch.cat((self.prevFrame, frame), dim=1), non_blocking=True
            )
        self.normStream.synchronize()

        waitForCaller(self.stream)
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
            waitForCaller(self.outputStream)
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
        self.isCuda = "cuda" in dedupMethod or "rocm" in dedupMethod

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


class DedupBlock:
    """Duplicate when no block of the frame changed, on average, by more than
    ``threshold`` (8-bit luma scale) from the last kept frame.

    Full resolution, so a mouth flap or a blink is not averaged away the way
    the 224px SSIM/MSE downsample does, and a per-block mean of the absolute
    change rather than a per-pixel max, so a few strong compression errors
    (ringing around line art) are diluted by their block instead of vetoing a
    real duplicate the way PixelDiffGuard's max does. Blocks
    are 32 pixels at 1080p and scale with the frame height, so the same
    threshold means the same fraction of the picture at any size.

    Runs on whatever device the frame is on. The comparison is against the
    last KEPT frame, as for every other detector, so a slow fade or drift
    accumulates until a block crosses the threshold instead of vanishing.
    """

    def __init__(self, threshold=5.0, blockSize=32):
        self.threshold = float(threshold) / 255.0
        self.blockSize = blockSize
        self.prevFrame = None
        self._weights = None
        # Not an SSIM/MSE score, so the scene-cut detector computes its own.
        self.lastReusableScore = None

    def _kernel(self, height: int) -> int:
        return max(2, round(self.blockSize * height / 1080))

    _LUMA = (0.299, 0.587, 0.114)

    def _luma(self, frame):
        """(N, 1, H, W) fp32 luma as a NEW tensor, never a view of ``frame``:
        it may become the reference, and the pipeline reuses frame buffers."""
        if frame.shape[1] < 3:
            return frame[:, :1].to(torch.float32, copy=True)
        key = (frame.device, frame.dtype)
        if self._weights is None or self._weights[0] != key:
            weights = torch.tensor(self._LUMA, device=frame.device, dtype=frame.dtype)
            self._weights = (key, weights.view(1, 3, 1, 1))
        # One kernel: multiply in the frame's dtype, accumulate in fp32. RGBA
        # ignores the alpha.
        return (frame[:, :3] * self._weights[1]).sum(
            1, keepdim=True, dtype=torch.float32
        )

    def score(self, frame):
        """Largest block-mean |luma change| against the reference (0-1 scale),
        and the frame's luma, so a kept frame does not convert it twice."""
        luma = self._luma(frame)
        h, w = luma.shape[-2:]
        k = self._kernel(h)
        # Adaptive pooling splits the frame into ceil(h/k) x ceil(w/k) nearly
        # equal blocks. A fixed k-stride grid either drops the leftover rows
        # and columns (1080 = 33*32 + 24: a subtitle line in the bottom strip
        # would sit outside every block) or keeps them as a strip a few pixels
        # wide, where noise on a handful of pixels counts as a whole block.
        grid = (-(-h // k), -(-w // k))
        change = (luma - self.prevFrame).abs_()
        if k >= 8:
            # Same block means at a sixteenth of the cost on CUDA (adaptive
            # pooling with ~32 px windows is the slow part): average 4x4 cells
            # first. A partial last cell (a height not divisible by 4) then
            # counts as a full one, which only reweights a 1-3 px edge strip.
            change = F.avg_pool2d(change, 4, 4, ceil_mode=True)
        change = F.adaptive_avg_pool2d(change, grid)
        return change.amax().item(), luma

    def __call__(self, frame):
        """
        Returns True if the frames are duplicates
        """
        if self.prevFrame is None:
            self.prevFrame = self._luma(frame)
            return False
        value, luma = self.score(frame)
        if value <= self.threshold:
            return True
        self.prevFrame = luma
        return False


class DedupSSIMROCm(DedupSSIMCuda):
    """ROCm (HIP) SSIM dedup. Same eager compare as CUDA; HIP device."""

    pass


class DedupMSEROCm(DedupMSECuda):
    """ROCm (HIP) MSE dedup. Same eager compare as CUDA; HIP device."""

    pass


class DedupFlownetSROCm(DedupFlownetS):
    """ROCm (HIP) FlowNetS dedup. Eager, no CUDA graphs or custom streams.

    The CUDA parent captures the flownet forward in a CUDAGraph; HIP graphs
    are skipped here for compatibility across RDNA/CDNA.
    """

    @torch.inference_mode()
    def initTorchCudaGraph(self):
        self.cudaGraph = None

    @torch.inference_mode()
    def __call__(self, frame):
        if self.prevFrame is None:
            self.prevFrame = self.prepareFrame(frame)
            return False

        frame = self.prepareFrame(frame)
        self.dummyInput.copy_(
            torch.cat((self.prevFrame, frame), dim=1), non_blocking=True
        )
        if self.dummyInput.device.type == "cuda":
            torch.cuda.current_stream().synchronize()

        output = self.model(self.dummyInput)
        if output.device.type == "cuda":
            torch.cuda.current_stream().synchronize()

        isDuplicate = bool(output.abs().mean() < self.dedupSens)

        if not isDuplicate:
            self.prevFrame.copy_(frame, non_blocking=True)

        return isDuplicate


def reseed(detector, frame) -> None:
    """Make ``frame`` the detector's reference, as if it had just been kept.

    Every detector above seeds its reference from the first frame it sees
    (``prevFrame is None``) and reports it as not a duplicate, so clearing the
    reference and calling it once re-seeds it in that detector's own format.
    """
    detector.prevFrame = None
    detector(frame)


class PixelDiffGuard:
    """Vetoes a duplicate verdict when any pixel changed more than ``maxDiff``.

    The detectors score a small downsample of the frame (224x224), so a local
    change -- a mouth flap, a blink, a HUD element appearing -- scores as a
    duplicate, and ``--smooth_dedup`` would replace that real drawing with an
    interpolated one. This compares the full-resolution frame with the last
    kept one and keeps the frame when its largest per-pixel difference, on an
    8-bit scale, is above ``maxDiff`` (DDFI's rule; 255 never vetoes). Only
    frames the detector already called duplicates are checked.

    The largest pixel, not an area average: in a dark scene a real new mouth
    shape can clear the limit at a single pixel, while the rest of it changes
    by 20-35 levels. A lone noise pixel above the limit therefore keeps a true
    duplicate now and then (about 1% of frames on a 1080p H.264 anime clip),
    which only costs that gap some smoothness; averaging would drop real
    drawings instead.

    On a veto the detector is re-seeded on the kept frame, so it and the guard
    keep comparing against the same frame.
    """

    def __init__(self, detector, maxDiff):
        self.detector = detector
        # Half a level of slack keeps the limit inclusive for float frames:
        # they are uint8 / 255, so a change of exactly maxDiff levels can land
        # a rounding error above maxDiff / 255.
        self.maxDiff = (float(maxDiff) + 0.5) / 255.0
        self.reference = None

    @property
    def lastReusableScore(self):
        # The scene-cut detector reuses this score when it directly follows the
        # duplicate detector; a vetoed frame leaves none (the re-seed clears it).
        return getattr(self.detector, "lastReusableScore", None)

    def __call__(self, frame):
        isDuplicate = bool(self.detector(frame))
        if isDuplicate and self.reference is not None:
            if self._largestChange(frame) <= self.maxDiff:
                return True
            reseed(self.detector, frame)
        self._keep(frame)
        return False

    def _largestChange(self, frame) -> float:
        reference = self.reference
        if frame.is_floating_point():
            return (frame - reference).abs().amax().item()
        white = torch.iinfo(frame.dtype).max
        return (frame.int() - reference.int()).abs().amax().item() / white

    def _keep(self, frame) -> None:
        reference = self.reference
        if (
            reference is None
            or reference.shape != frame.shape
            or reference.dtype != frame.dtype
            or reference.device != frame.device
        ):
            self.reference = frame.clone()
        else:
            reference.copy_(frame)
