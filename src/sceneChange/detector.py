"""
Streaming scene-change detectors.

Each detector is a callable ``__call__(frame) -> bool`` returning True when
``frame`` is a hard cut relative to the previously seen frame. The detector
holds its own downsampled reference frame and advances it EVERY call (unlike
dedup, which only advances on a kept frame). The first frame always returns
False (no predecessor).

Cheap tier (ssim/mse) reuses the SSIM module and MSE math from ``src/dedup``.
A cut is the inverse of the dedup duplicate test:
  - SSIM: high == similar, so cut when ``ssim < threshold``.
  - MSE:  low  == similar, so cut when ``mse > threshold``.
The maxxvit tier reuses the shared 6-channel classifier; cut when the softmax
cut-probability exceeds the threshold.
"""

import torch
import torch.nn.functional as F

from src.infra.isCudaInit import CudaChecker

checker = CudaChecker()


class _SSIMBase:
    """Shared SSIM cut logic; subclasses set device/dtype and resize mode."""

    def __init__(self, threshold, sampleSize, device, half, mode):
        from src.dedup.ssim import SSIM

        self.threshold = threshold
        self.sampleSize = sampleSize
        self.device = device
        self.half = half
        self.mode = mode
        self.prevFrame = None
        self.ssim = SSIM(data_range=1.0, channel=3).to(device)
        if half:
            self.ssim.half()
        else:
            self.ssim.float()
        self.ssim.eval()

    def _prep(self, frame):
        frame = frame.half() if self.half else frame.float()
        return F.interpolate(
            frame, (self.sampleSize, self.sampleSize), mode=self.mode
        ).to(self.device)

    @torch.inference_mode()
    def __call__(self, frame):
        cur = self._prep(frame)
        if self.prevFrame is None:
            self.prevFrame = cur
            return False
        score = self.ssim(self.prevFrame, cur).mean().item()
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

    def _prep(self, frame):
        return F.interpolate(
            frame.float(),
            size=(self.sampleSize, self.sampleSize),
            mode="bilinear",
            align_corners=False,
        ).to(self.device)


class _MSEBase:
    """Shared MSE cut logic. MSE low == similar, so cut when mse > threshold."""

    def __init__(self, threshold, sampleSize, half, cuda):
        self.threshold = threshold
        self.sampleSize = sampleSize
        self.half = half
        self.cuda = cuda
        self.prevFrame = None

    def _prep(self, frame):
        if self.cuda:
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
    def __call__(self, frame):
        cur = self._prep(frame)
        if self.prevFrame is None:
            self.prevFrame = cur
            return False
        score = ((self.prevFrame - cur) ** 2).mean().item()
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
