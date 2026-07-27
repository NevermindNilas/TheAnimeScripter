import math

import torch
import torch.nn.functional as F
from frame_analytics import ssim

# frame_analytics.ssim builds its gaussian window per call from the inputs, so
# there is nothing device- or dtype-bound to cache. The old torch.jit module
# held the window as a buffer, which is why it needed one instance per
# (device, dtype) or it raised "expected scalar type Half but found Float".

# The scales --dynamic_scale is allowed to pick. Powers of two only: scale_list
# entries are consumed as ``F.interpolate(scale_factor=1/scale_list[i])``, and a
# non-power-of-two pick (the old 1.5) gives fractional block resolutions --
# 16/1.5 = 10.67 -- whose rounded output shape depends on the input size. Three
# picks also keep the per-scale CUDA-graph set small, see
# ``RifeCuda._setupCudaGraph``.
DYNAMIC_SCALES = (0.5, 1.0, 2.0)

# SSIM is a whole-frame statistic; separating "these frames barely moved" from
# "these frames are unrelated" does not need full resolution. Scoring a
# downscaled copy is ~1/60 the work at 1080p and does not move the pick.
_ANALYSIS_MAX_SIDE = 256


def _forAnalysis(img: torch.Tensor) -> torch.Tensor:
    h, w = img.shape[-2], img.shape[-1]
    longSide = max(h, w)
    if longSide <= _ANALYSIS_MAX_SIDE:
        return img
    return F.interpolate(
        img,
        scale_factor=_ANALYSIS_MAX_SIDE / longSide,
        mode="bilinear",
        align_corners=False,
        recompute_scale_factor=False,
    )


def dynamicScale(
    img1: torch.Tensor,
    img2: torch.Tensor,
    minScale: float = 0.5,
    maxScale: float = 2.0,
) -> float:
    """
    Pick the RIFE pyramid scale for one frame pair from their SSIM.

    Direction matters and is easy to get backwards. ``scale_list`` is built as
    ``[base / scale, ...]`` and each entry is used as
    ``F.interpolate(scale_factor=1/scale_list[i])``, so a SMALL return value
    means MORE downsampling -- a coarser pyramid with a larger receptive field
    measured in source pixels -- and a LARGE one means a finer pyramid. That is
    the same convention Practical-RIFE documents: 0.5 is the 4K /
    large-displacement setting, 2.0 the 480p / small-displacement one.

    So low SSIM (the pair changed a lot, i.e. large displacement) maps to
    ``minScale`` and a near-duplicate pair maps to ``maxScale``. Until
    2026-07-26 this was inverted -- it handed the finest pyramid to exactly the
    pairs whose displacement did not fit the receptive field.

    The result is snapped to the nearest entry of ``DYNAMIC_SCALES``.
    """
    if img1.shape != img2.shape:
        raise ValueError(
            f"Input images must have the same shape, got {img1.shape} and {img2.shape}"
        )

    if img1.device != img2.device:
        raise ValueError(
            f"Both images must be on the same device, got {img1.device} and {img2.device}"
        )

    ssimValue = ssim(_forAnalysis(img1), _forAnalysis(img2), data_range=1.0).item()

    scale = minScale + (maxScale - minScale) * (ssimValue**2)
    scale = max(minScale, min(maxScale, scale))
    # Snap in log2, not linearly: the picks are octaves apart, so 1.46 sits
    # nearer to 1.0 than to 2.0 even though it is arithmetically closer to 2.0.
    target = math.log2(scale)
    return min(DYNAMIC_SCALES, key=lambda s: abs(math.log2(s) - target))


def pickScale(model, img0: torch.Tensor, img1: torch.Tensor) -> float:
    """
    The scale an arch's ``forward`` should use for this pair.

    Drivers that can do better than the arch -- ``RifeCuda`` / ``RifeMPS``, which
    hold the UNPADDED frame size and know the pair only changes once per
    ``__call__`` -- score the pair themselves and park the result on
    ``model.dsScale``. That is strictly better than scoring inside ``forward``:

      * ``forward`` is handed the zero-PADDED buffers, and the pad region is
        identical in both, so it scores as perfectly similar and drags the SSIM
        up. At 1080p / mod-128 that is 72 of 1152 rows pinned at 1.0.
      * ``forward`` runs once per inserted frame, so at ``--interpolate_factor
        4`` the same pair was scored (and synchronised on) three extra times.
      * a scale known before the forward lets the driver pick a pre-captured
        CUDA graph instead of giving graph capture up entirely.

    An arch used without such a driver keeps working: with no ``dsScale`` set it
    falls back to scoring what it was handed.
    """
    scale = getattr(model, "dsScale", None)
    return dynamicScale(img0, img1) if scale is None else scale
