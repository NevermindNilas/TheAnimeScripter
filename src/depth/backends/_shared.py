import os

os.environ.setdefault("DA3_LOG_LEVEL", "ERROR")

import logging

import cv2
import numpy as np
import torch

from src.infra.isCudaInit import CudaChecker

checker = CudaChecker()


class SlidingWindowNormalizer:
    """Ghost-free global affine stabilization for monocular depth streams.

    This class deliberately never composites historical depth pixels. It
    robustly normalizes the current prediction, then uses bidirectional flow
    correspondences only to estimate a small global scale/shift correction.
    A flow mistake can therefore change global contrast slightly, but cannot
    stamp an old object, texture, or edge into the current frame.
    """

    def __init__(
        self,
        low_pct: float = 2.0,
        high_pct: float = 98.0,
        flow_size: int = 320,
        strength: float = 0.75,
        max_scale_change: float = 0.10,
        max_shift: float = 0.04,
        min_coverage: float = 0.08,
        innovation_threshold: float = 0.12,
    ):
        self.low_q = low_pct / 100.0
        self.high_q = high_pct / 100.0
        self.flow_size = max(64, int(flow_size))
        self.strength = min(max(float(strength), 0.0), 1.0)
        self.max_scale_change = max(float(max_scale_change), 0.0)
        self.max_shift = max(float(max_shift), 0.0)
        self.min_coverage = min(max(float(min_coverage), 0.0), 1.0)
        self.innovation_threshold = max(float(innovation_threshold), 1e-6)
        self.prev_observation = None
        self.prev_stabilized = None
        self.prev_valid = None
        self.flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)

    def reset(self):
        self.prev_observation = None
        self.prev_stabilized = None
        self.prev_valid = None

    def _small_frame(self, frame):
        height, width = frame.shape
        scale = min(1.0, self.flow_size / max(height, width))
        if scale == 1.0:
            return np.ascontiguousarray(frame, dtype=np.float32)
        size = (max(8, round(width * scale)), max(8, round(height * scale)))
        return cv2.resize(frame, size, interpolation=cv2.INTER_AREA).astype(
            np.float32, copy=False
        )

    @staticmethod
    def _fit_affine(current, target):
        """Huber IRLS fit for target ~= scale * current + shift."""
        if current.size > 20_000:
            step = max(1, current.size // 20_000)
            current = current[::step]
            target = target[::step]

        design = np.column_stack((current, np.ones_like(current)))
        scale, shift = np.linalg.lstsq(design, target, rcond=None)[0]
        for _ in range(3):
            residual = target - (scale * current + shift)
            sigma = max(float(np.median(np.abs(residual))) * 1.4826, 0.01)
            weights = np.minimum(1.0, 1.5 * sigma / (np.abs(residual) + 1e-6))
            weighted = design * weights[:, None]
            scale, shift = np.linalg.lstsq(weighted, target * weights, rcond=None)[0]
        return float(scale), float(shift)

    def _temporal_affine(self, observation, valid):
        small = self._small_frame(observation)
        small_valid = self._small_frame(valid.astype(np.float32)) > 0.999

        if self.prev_observation is None or self.prev_observation.shape != small.shape:
            self.prev_observation = small
            self.prev_stabilized = small
            self.prev_valid = small_valid
            return 1.0, 0.0

        previous_u8 = np.clip(self.prev_observation * 255.0, 0, 255).astype(np.uint8)
        current_u8 = np.clip(small * 255.0, 0, 255).astype(np.uint8)
        backward = self.flow.calc(current_u8, previous_u8, None)
        forward = self.flow.calc(previous_u8, current_u8, None)

        height, width = small.shape
        grid_x, grid_y = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32),
        )
        map_x = grid_x + backward[..., 0]
        map_y = grid_y + backward[..., 1]
        remap_args = (map_x, map_y, cv2.INTER_LINEAR)
        previous_stabilized = cv2.remap(
            self.prev_stabilized,
            *remap_args,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=np.nan,
        )
        forward_at_previous = cv2.remap(
            forward,
            *remap_args,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=np.nan,
        )

        cycle_error = np.linalg.norm(backward + forward_at_previous, axis=2)
        previous_observation = cv2.remap(
            self.prev_observation,
            *remap_args,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=np.nan,
        )
        innovation = np.abs(previous_observation - small)
        correspondence = (
            small_valid
            & np.isfinite(previous_stabilized)
            & np.isfinite(forward_at_previous).all(axis=2)
            & (cycle_error < 1.0)
            & (innovation < self.innovation_threshold)
            & (small > 0.02)
            & (small < 0.98)
            & (previous_stabilized > 0.02)
            & (previous_stabilized < 0.98)
        )

        scale, shift = 1.0, 0.0
        if correspondence.mean() >= self.min_coverage:
            scale, shift = self._fit_affine(
                small[correspondence], previous_stabilized[correspondence]
            )
            scale = float(
                np.clip(
                    scale,
                    1.0 - self.max_scale_change,
                    1.0 + self.max_scale_change,
                )
            )
            shift = float(np.clip(shift, -self.max_shift, self.max_shift))
            scale = 1.0 + self.strength * (scale - 1.0)
            shift *= self.strength

        stabilized = np.clip(scale * small + shift, 0.0, 1.0)
        self.prev_observation = small
        self.prev_stabilized = stabilized.astype(np.float32, copy=False)
        self.prev_valid = small_valid
        return scale, shift

    def normalize(self, depth, mask=None):
        is_numpy = isinstance(depth, np.ndarray)
        if is_numpy:
            t = torch.from_numpy(np.ascontiguousarray(depth)).float()
        else:
            t = depth if depth.dtype.is_floating_point else depth.float()

        finite = torch.isfinite(t)
        if mask is None:
            valid = finite
        else:
            mask_tensor = torch.as_tensor(mask, device=t.device, dtype=torch.bool)
            valid = finite & mask_tensor
        sample = t[valid]
        if sample.numel() == 0:
            result = torch.zeros_like(t)
            return result.cpu().numpy() if is_numpy else result

        if sample.dtype not in (torch.float32, torch.float64):
            sample = sample.float()

        cur_low = torch.quantile(sample, self.low_q).item()
        cur_high = torch.quantile(sample, self.high_q).item()
        denom = max(cur_high - cur_low, 1e-6)
        normalized = ((torch.nan_to_num(t, nan=cur_low) - cur_low) / denom).clamp(
            0.0, 1.0
        )

        observation = normalized.detach().float().squeeze().cpu().numpy()
        valid_numpy = valid.detach().squeeze().cpu().numpy().astype(bool)
        if observation.ndim == 2:
            scale, shift = self._temporal_affine(observation, valid_numpy)
            normalized = (normalized * scale + shift).clamp(0.0, 1.0)
        else:
            self.reset()

        if is_numpy:
            return normalized.cpu().numpy()
        return normalized


class VideoRangeNormalizer:
    """Slow shared-range calibration for an already-temporal depth model.

    Video Depth Anything predicts frames in a shared affine system. Re-running
    min/max independently on every frame destroys that property. This keeps a
    scene-stable robust range and adapts it only very slowly. No depth history
    is warped, averaged, or composited.
    """

    def __init__(
        self,
        low_pct: float = 1.0,
        high_pct: float = 99.0,
        adapt_rate: float = 0.02,
        max_step_ratio: float = 0.05,
        cut_ratio: float = 3.0,
    ):
        self.low_q = low_pct / 100.0
        self.high_q = high_pct / 100.0
        self.adapt_rate = float(adapt_rate)
        self.max_step_ratio = float(max_step_ratio)
        self.cut_ratio = float(cut_ratio)
        self.low = None
        self.high = None

    def reset(self):
        self.low = None
        self.high = None

    def _update_range(self, current_low, current_high):
        current_range = max(current_high - current_low, 1e-6)
        if self.low is None:
            self.low, self.high = current_low, current_high
            return
        stable_range = max(self.high - self.low, 1e-6)
        ratio = current_range / stable_range
        shift = abs(current_low - self.low) / stable_range
        if ratio > self.cut_ratio or ratio < 1.0 / self.cut_ratio or shift > 2.0:
            self.low, self.high = current_low, current_high
            return
        max_step = self.max_step_ratio * stable_range
        low_delta = float(np.clip(current_low - self.low, -max_step, max_step))
        high_delta = float(np.clip(current_high - self.high, -max_step, max_step))
        self.low += self.adapt_rate * low_delta
        self.high += self.adapt_rate * high_delta

    def normalize(self, depth):
        if isinstance(depth, np.ndarray):
            array = np.ascontiguousarray(depth, dtype=np.float32)
            sample = array[np.isfinite(array)]
            if sample.size == 0:
                return np.zeros_like(array)
            current_low, current_high = np.quantile(sample, (self.low_q, self.high_q))
            self._update_range(float(current_low), float(current_high))
            denominator = max(self.high - self.low, 1e-6)
            return np.clip(
                (np.nan_to_num(array, nan=self.low) - self.low) / denominator,
                0.0,
                1.0,
            ).astype(np.float32, copy=False)
        else:
            tensor = depth if depth.dtype.is_floating_point else depth.float()

        finite = torch.isfinite(tensor)
        sample = tensor[finite]
        if sample.numel() == 0:
            return torch.zeros_like(tensor)
        if sample.dtype not in (torch.float32, torch.float64):
            sample = sample.float()

        current_low = torch.quantile(sample, self.low_q).item()
        current_high = torch.quantile(sample, self.high_q).item()
        self._update_range(current_low, current_high)

        denominator = max(self.high - self.low, 1e-6)
        normalized = (
            (torch.nan_to_num(tensor, nan=self.low) - self.low) / denominator
        ).clamp(0.0, 1.0)
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
