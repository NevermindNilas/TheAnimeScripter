import os
os.environ.setdefault("DA3_LOG_LEVEL", "ERROR")

import torch
import logging
import numpy as np

from src.infra.isCudaInit import CudaChecker
from src.constants import ADOBE


if ADOBE:
    from src.server.aeComms import progressState

from collections import deque
import statistics

checker = CudaChecker()


class SlidingWindowNormalizer:
    """
    Causal window-median percentile deflicker for depth streams.

    Smooths only the normalization range (low/high percentiles) across a
    sliding window of past frames, then rescales each frame independently.
    No pixel-level temporal blending, so no motion ghosting. Resets the
    window when per-frame percentiles diverge sharply from the running
    median (scene cut / abrupt exposure change).
    """

    def __init__(
        self,
        low_pct: float = 2.0,
        high_pct: float = 98.0,
        window_size: int = 7,
        scene_cut_ratio: float = 2.5,
        scene_cut_shift: float = 1.0,
    ):
        self.low_q = low_pct / 100.0
        self.high_q = high_pct / 100.0
        self.window_size = window_size
        self.scene_cut_ratio = scene_cut_ratio
        self.scene_cut_shift = scene_cut_shift
        self.lows: deque = deque(maxlen=window_size)
        self.highs: deque = deque(maxlen=window_size)

    def normalize(self, depth, mask=None):
        is_numpy = isinstance(depth, np.ndarray)
        if is_numpy:
            t = torch.from_numpy(depth).float()
        else:
            t = depth if depth.dtype.is_floating_point else depth.float()

        if mask is not None:
            sample = t[mask]
            if sample.numel() == 0:
                sample = t.flatten()
        else:
            sample = t.flatten()

        if sample.dtype not in (torch.float32, torch.float64):
            sample = sample.float()

        cur_low = torch.quantile(sample, self.low_q).item()
        cur_high = torch.quantile(sample, self.high_q).item()

        if len(self.lows) >= 3:
            med_low = statistics.median(self.lows)
            med_high = statistics.median(self.highs)
            med_range = max(med_high - med_low, 1e-6)
            cur_range = max(cur_high - cur_low, 1e-6)
            ratio = cur_range / med_range
            shift = abs(cur_low - med_low) / med_range
            if (
                ratio > self.scene_cut_ratio
                or ratio < 1.0 / self.scene_cut_ratio
                or shift > self.scene_cut_shift
            ):
                self.lows.clear()
                self.highs.clear()

        self.lows.append(cur_low)
        self.highs.append(cur_high)

        target_low = statistics.median(self.lows)
        target_high = statistics.median(self.highs)
        denom = max(target_high - target_low, 1e-6)

        normalized = ((t - target_low) / denom).clamp(0.0, 1.0)

        if is_numpy:
            return normalized.cpu().numpy()
        return normalized


MEANTENSOR = (
    torch.tensor([0.485, 0.456, 0.406]).contiguous().view(3, 1, 1).to(checker.device)
)
STDTENSOR = (
    torch.tensor([0.229, 0.224, 0.225]).contiguous().view(3, 1, 1).to(checker.device)
)
MEANTENSOR_HALF = MEANTENSOR.half() if checker.cudaAvailable else MEANTENSOR
STDTENSOR_HALF = STDTENSOR.half() if checker.cudaAvailable else STDTENSOR


def calculateAspectRatio(width, height, depthQuality="high", isV3=False):
    if isV3:
        if depthQuality == "high":
            return ((max(width, height) + 13) // 14) * 14
        if depthQuality == "medium":
            return 700
        return 518

    if depthQuality == "high":
        # Whilst this doesn't necessarily allign with the model, it produces
        # sharper results at the cost of performance and some accuracy loss.
        newHeight = ((height + 13) // 14) * 14
        newWidth = ((width + 13) // 14) * 14
    else:
        # I'd suggest 700px as a good middle ground for resizing
        size = 700 if depthQuality == "medium" else 518
        newHeight = size
        newWidth = size

    logging.info(f"Depth Padding: {newWidth}x{newHeight}")
    return newHeight, newWidth

