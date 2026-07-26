import torch
from frame_analytics import ssim

# frame_analytics.ssim builds its gaussian window per call from the inputs, so
# there is nothing device- or dtype-bound to cache. The old torch.jit module
# held the window as a buffer, which is why it needed one instance per
# (device, dtype) or it raised "expected scalar type Half but found Float".


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

    ssim_value = ssim(img1, img2, data_range=1.0).item()

    scale = minScale + (maxScale - minScale) * (1 - (ssim_value**2))
    scale = max(minScale, min(maxScale, scale))
    scale = round(scale / 0.5) * 0.5
    return scale
