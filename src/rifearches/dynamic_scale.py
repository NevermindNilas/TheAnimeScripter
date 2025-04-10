import torch
from src.dedup.ssim import SSIM

SSIMFUNCTION = None


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
    global SSIMFUNCTION
    if SSIMFUNCTION is None:
        SSIMFUNCTION = SSIM(data_range=1.0, channel=3)
        if img1.is_cuda:
            SSIMFUNCTION = SSIMFUNCTION.cuda()
        if img1.dtype == torch.float16:
            SSIMFUNCTION = SSIMFUNCTION.half()

    if img1.shape != img2.shape:
        raise ValueError(
            f"Input images must have the same shape, got {img1.shape} and {img2.shape}"
        )

    if img1.device != img2.device:
        raise ValueError(
            f"Both images must be on the same device, got {img1.device} and {img2.device}"
        )

    ssim_value = SSIMFUNCTION(img1, img2).mean().item()

    scale = minScale + (maxScale - minScale) * (1 - (ssim_value**2))
    scale = max(minScale, min(maxScale, scale))
    scale = round(scale / 0.5) * 0.5
    return scale


def dynamicScaleList(
    img1: torch.Tensor,
    img2: torch.Tensor,
    minScale: float = 0.5,
    maxScale: float = 2.0,
    scaleList: list = [],
) -> list:
    """
    This function is a wrapper around dynamicScale that applies the scale to a list of scales
    Should simplify the amount of extra repeated code in the main function
    """
    scale = dynamicScale(img1, img2, minScale, maxScale)
    for i in range(len(scaleList)):
        scaleList[i] = scaleList[i] / scale

    return scaleList
