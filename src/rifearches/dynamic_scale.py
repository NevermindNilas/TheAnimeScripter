import torch

from src.dedup.ssim import SSIM

# One SSIM module per (device, dtype). Its gaussian window is a buffer bound to
# both, so a single cached instance silently binds to whatever the first caller
# happened to pass and then raises "expected scalar type Half but found Float"
# for every caller afterwards.
_SSIMFUNCTIONS: dict[tuple[torch.device, torch.dtype], SSIM] = {}


def _ssimFor(device: torch.device, dtype: torch.dtype) -> SSIM:
    key = (device, dtype)
    fn = _SSIMFUNCTIONS.get(key)
    if fn is None:
        fn = SSIM(data_range=1.0, channel=3).to(device=device, dtype=dtype)
        _SSIMFUNCTIONS[key] = fn
    return fn


def dynamicScale(
    img1: torch.Tensor,
    img2: torch.Tensor,
    minScale: float = 0.5,
    maxScale: float = 2.0,
) -> float:
    """
    This function calculates the scale factor between two images using SSIM.
    The scale factor is inversely proportional to the Structural Similarity Index (SSIM) between the two images.
    The scale factor is calculated using linear interpolation between minScale and maxScale based on the SSIM value,
    then clamped between minScale (largest) and maxScale (smallest),
    and rounded to the nearest 0.5.
    """
    if img1.shape != img2.shape:
        raise ValueError(
            f"Input images must have the same shape, got {img1.shape} and {img2.shape}"
        )

    if img1.device != img2.device:
        raise ValueError(
            f"Both images must be on the same device, got {img1.device} and {img2.device}"
        )

    ssim_value = _ssimFor(img1.device, img1.dtype)(img1, img2).mean().item()

    scale = minScale + (maxScale - minScale) * (1 - (ssim_value**2))
    scale = max(minScale, min(maxScale, scale))
    scale = round(scale / 0.5) * 0.5
    return scale
