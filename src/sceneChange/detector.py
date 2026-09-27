"""
Streaming scene-change detectors.

Each detector is a callable ``__call__(frame) -> bool`` returning True when
``frame`` is a hard cut relative to the previously seen frame. The detector
holds its own downsampled reference frame and advances it EVERY call (unlike
dedup, which only advances on a kept frame). The first frame always returns
False (no predecessor).

Cheap tier (ssim/mse) scores with ``frame_analytics``, the same kernels
``src/dedup`` uses. A cut is the inverse of the dedup duplicate test:
  - SSIM: high == similar, so cut when ``ssim < threshold``.
  - MSE:  low  == similar, so cut when ``mse > threshold``.
The maxxvit tier reuses the shared 6-channel classifier; cut when the softmax
cut-probability exceeds the threshold.
"""

import torch
import torch.nn.functional as F


def _scoreToFloat(score) -> float:
    return score.item() if hasattr(score, "item") else float(score)


class _ReusableScoreConsumer:
    supportsScoreReuse = True
    scoreFamily = None
    scoreScale = None
    scoreResizeMode = None
    scoreAlignCorners = None

    def _canReuseScore(self, reusableScore) -> bool:
        if reusableScore is None or self.prevFrame is None:
            return False
        currentFrame = getattr(reusableScore, "currentFrame", None)
        if currentFrame is None:
            return False
        return (
            getattr(reusableScore, "family", None) == self.scoreFamily
            and getattr(reusableScore, "sampleSize", None) == self.sampleSize
            and getattr(reusableScore, "dtype", None) == self.prevFrame.dtype
            and getattr(reusableScore, "device", None) == self.prevFrame.device
            and currentFrame.dtype == self.prevFrame.dtype
            and currentFrame.device == self.prevFrame.device
            and getattr(reusableScore, "resizeMode", None) == self.scoreResizeMode
            and getattr(reusableScore, "alignCorners", None) == self.scoreAlignCorners
            and getattr(reusableScore, "scale", None) == self.scoreScale
        )

    def _consumeReusableScore(self, reusableScore) -> float | None:
        if not self._canReuseScore(reusableScore):
            return None
        self.prevFrame = reusableScore.currentFrame
        return reusableScore.score


class _SSIMBase(_ReusableScoreConsumer):
    """Shared SSIM cut logic; subclasses set device/dtype and resize mode."""

    scoreFamily = "ssim"
    scoreScale = 1.0

    def __init__(self, threshold, sampleSize, device, half, mode):
        from frame_analytics import ssim

        self.threshold = threshold
        self.sampleSize = sampleSize
        self.device = device
        self.half = half
        self.mode = mode
        self.scoreResizeMode = mode
        self.scoreAlignCorners = None
        self.prevFrame = None
        # Accumulation is fp32/fp64 regardless of the input dtype, so `half`
        # only picks the resize/compare dtype, not the score's precision.
        self.ssim = ssim

    def _prep(self, frame):
        targetDtype = torch.float16 if self.half else torch.float32
        if frame.dtype == targetDtype:
            return F.interpolate(
                frame, (self.sampleSize, self.sampleSize), mode=self.mode
            ).to(self.device)
        if self.mode == "nearest" and frame.dtype in (
            torch.float16,
            torch.float32,
            torch.float64,
        ):
            # Nearest copies samples: cast only the small sampled frame.
            resized = F.interpolate(
                frame, (self.sampleSize, self.sampleSize), mode=self.mode
            )
            return (resized.half() if self.half else resized.float()).to(self.device)
        frame = frame.half() if self.half else frame.float()
        return F.interpolate(
            frame, (self.sampleSize, self.sampleSize), mode=self.mode
        ).to(self.device)

    @torch.inference_mode()
    def __call__(self, frame, reusableScore=None):
        score = self._consumeReusableScore(reusableScore)
        if score is not None:
            # SSIM high == similar; a scene cut is a large drop in similarity.
            return score < self.threshold

        cur = self._prep(frame)
        if self.prevFrame is None:
            self.prevFrame = cur
            return False
        score = _scoreToFloat(self.ssim(self.prevFrame, cur, data_range=1.0))
        self.prevFrame = cur
        # SSIM high == similar; a scene cut is a large drop in similarity.
        return score < self.threshold


class SceneChangeSSIMCuda(_SSIMBase):
    def __init__(self, threshold=0.5, half=True, sampleSize=224):
        super().__init__(
            threshold,
            sampleSize,
            device=torch.device("cuda"),
            half=half,
            mode="nearest",
        )


class SceneChangeSSIM(_SSIMBase):
    def __init__(self, threshold=0.5, sampleSize=224):
        # CPU SSIM: bilinear resize (matches DedupSSIM), fp32.
        super().__init__(
            threshold,
            sampleSize,
            device=torch.device("cpu"),
            half=False,
            mode="bilinear",
        )
        self.scoreAlignCorners = False

    def _prep(self, frame):
        return F.interpolate(
            frame.float(),
            size=(self.sampleSize, self.sampleSize),
            mode="bilinear",
            align_corners=False,
        ).to(self.device)


class _MSEBase(_ReusableScoreConsumer):
    """Shared MSE cut logic. MSE low == similar, so cut when mse > threshold."""

    scoreFamily = "mse"
    scoreScale = 255.0

    def __init__(self, threshold, sampleSize, half, cuda):
        from frame_analytics import mse

        self.threshold = threshold
        self.sampleSize = sampleSize
        self.half = half
        self.cuda = cuda
        self.scoreResizeMode = "nearest" if cuda else "bilinear"
        self.scoreAlignCorners = None if cuda else False
        self.prevFrame = None
        self.mse = mse

    def _prep(self, frame):
        if self.cuda:
            targetDtype = torch.float16 if self.half else torch.float32
            if frame.dtype == targetDtype:
                return F.interpolate(
                    frame, (self.sampleSize, self.sampleSize), mode="nearest"
                ).mul(255.0)
            if frame.dtype in (
                torch.float16,
                torch.float32,
                torch.float64,
            ):
                resized = F.interpolate(
                    frame, (self.sampleSize, self.sampleSize), mode="nearest"
                )
                return (resized.half() if self.half else resized.float()).mul(255.0)
            frame = frame.half() if self.half else frame.float()
            return F.interpolate(
                frame, (self.sampleSize, self.sampleSize), mode="nearest"
            ).mul(255.0)
        return F.interpolate(
            frame.float(),
            size=(self.sampleSize, self.sampleSize),
            mode="bilinear",
            align_corners=False,
        ).mul(255.0)

    @torch.inference_mode()
    def __call__(self, frame, reusableScore=None):
        score = self._consumeReusableScore(reusableScore)
        if score is not None:
            return score > self.threshold

        cur = self._prep(frame)
        if self.prevFrame is None:
            self.prevFrame = cur
            return False
        score = _scoreToFloat(self.mse(self.prevFrame, cur))
        self.prevFrame = cur
        return score > self.threshold


class SceneChangeMSECuda(_MSEBase):
    def __init__(self, threshold=1000.0, half=True, sampleSize=224):
        super().__init__(threshold, sampleSize, half=half, cuda=True)


class SceneChangeMSE(_MSEBase):
    def __init__(self, threshold=1000.0, sampleSize=224):
        super().__init__(threshold, sampleSize, half=False, cuda=False)


class SceneChangeScorer6chDetector:
    """Wrap the shared 6-channel ONNX classifier (maxxvit / differential /
    shift_lpips) as a streaming detector. Cut when cut-probability >
    threshold."""

    def __init__(self, method, threshold=0.5, half=True, size=224):
        from src.sceneChange.scorer6ch import SceneChangeScorer6ch

        self.threshold = threshold
        self.scorer = SceneChangeScorer6ch(method, half, size=size)
        self.prevFrame = None

    def __call__(self, frame):
        cur = self.scorer.preprocessCHW(frame)
        if self.prevFrame is None:
            self.prevFrame = cur
            return False
        prob = self.scorer.score(self.prevFrame, cur)
        self.prevFrame = cur
        return prob > self.threshold


class SceneChangeSSIMROCm(SceneChangeSSIMCuda):
    """ROCm (HIP) SSIM scene-cut detector. Same eager compare as CUDA."""

    pass


class SceneChangeMSEROCm(SceneChangeMSECuda):
    """ROCm (HIP) MSE scene-cut detector. Same eager compare as CUDA."""

    pass
