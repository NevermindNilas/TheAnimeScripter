import torch
import torch.nn.functional as F

weights = torch.tensor([0.2989, 0.5870, 0.1140])
sobelX = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0)
sobelY = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).unsqueeze(0).unsqueeze(0)


def gaussianBlur(img, kernelSize=5, sigma=1.0):
    channels, height, width = img.shape
    x = torch.arange(-kernelSize // 2 + 1.0, kernelSize // 2 + 1.0)
    x = torch.exp(-(x**2) / (2 * sigma**2))
    kernel1d = x / x.sum()
    kernel2d = kernel1d[:, None] * kernel1d[None, :]
    kernel2d = kernel2d.to(img.device, dtype=img.dtype)
    kernel2d = kernel2d.expand(channels, 1, kernelSize, kernelSize)
    img = img.unsqueeze(0)
    blurredImg = F.conv2d(img, kernel2d, padding=kernelSize // 2, groups=channels)
    return blurredImg.squeeze(0)


def darkenLines(
    image: torch.Tensor,
    half: bool = True,
    thinEdges: bool = True,
    Gaussian: bool = True,
    darkenStrength: float = 0.6,
) -> torch.Tensor:
    """
    Darken the lines of an anime-style image to give it a slight perceived sharpness boost.

    Args:
        image (torch.Tensor): Input image tensor of shape (H, W, C) with values in the range [0, 255].
        half (bool): If True, process the image in half precision (FP16).
        thinEdges (bool): If True, apply line thinning.
        Gaussian (bool): If True, apply Gaussian blur to soften the edges.

    Returns:
        torch.Tensor: Image tensor with enhanced lines.
    """

    image = image.half() if half else image.float()
    image = image.permute(2, 0, 1).mul(1 / 255.0)

    weightsLocal = weights.to(device=image.device, dtype=image.dtype)
    grayscale = torch.tensordot(image, weightsLocal, dims=([0], [0]))

    sobelXLocal = sobelX.to(device=image.device, dtype=image.dtype)
    sobelYLocal = sobelY.to(device=image.device, dtype=image.dtype)

    def applyFilter(img, kernel):
        img = img.unsqueeze(0).unsqueeze(0)
        filteredImg = F.conv2d(img, kernel, padding=1)
        return filteredImg.squeeze(0).squeeze(0)

    edgesX = applyFilter(grayscale, sobelXLocal)
    edgesY = applyFilter(grayscale, sobelYLocal)
    edges = torch.sqrt(edgesX**2 + edgesY**2)

    edges = (edges - edges.min()) / (edges.max() - edges.min())

    if thinEdges:
        edges = edges.unsqueeze(0).unsqueeze(0)
        thinnedEdges = -F.max_pool2d(-edges, kernel_size=3, stride=1, padding=1)
        thinnedEdges = thinnedEdges.squeeze(0).squeeze(0)
    else:
        thinnedEdges = edges

    if Gaussian:
        softenedEdges = gaussianBlur(thinnedEdges.unsqueeze(0)).squeeze(0)
    else:
        softenedEdges = thinnedEdges

    enhancedImage = image.sub_(darkenStrength * softenedEdges.unsqueeze(0))
    enhancedImage = torch.clamp(enhancedImage, 0, 1)
    return enhancedImage.permute(1, 2, 0).mul(255)
