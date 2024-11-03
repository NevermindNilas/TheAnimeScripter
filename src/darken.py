import torch
import torch.nn.functional as F

"""
I can't find the original author anymore, but heavily inspired by the Vapoursynth implementation:

https://github.com/search?q=fastlinedarken&type=code
"""


class FastLineDarken:
    def __init__(self, half: bool = False):
        self.half = half

        # Initialize weights and Sobel kernels
        self.weights = torch.tensor([0.2989, 0.5870, 0.1140])
        self.sobelX = (
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0)
        )
        self.sobelY = (
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).unsqueeze(0).unsqueeze(0)
        )

        # Check if CUDA is available
        self.ISCUDA = torch.cuda.is_available()

        # Move tensors to appropriate device
        device = "cuda" if self.ISCUDA else "cpu"
        self.weightsLocal = self.weights.to(device)
        self.sobelXLocal = self.sobelX.to(device)
        self.sobelYLocal = self.sobelY.to(device)

        if self.half and self.ISCUDA:
            self.weightsLocal = self.weightsLocal.half()
            self.sobelXLocal = self.sobelXLocal.half()
            self.sobelYLocal = self.sobelYLocal.half()

        if self.ISCUDA:
            self.normStream = torch.cuda.Stream()

    def gaussianBlur(self, img, kernelSize=5, sigma=1.0):
        channels, _, _ = img.shape
        x = torch.arange(-kernelSize // 2 + 1.0, kernelSize // 2 + 1.0)
        x = torch.exp(-(x**2) / (2 * sigma**2))
        kernel1d = x / x.sum()
        kernel2d = kernel1d[:, None] * kernel1d[None, :]
        kernel2d = kernel2d.to(img.device, dtype=img.dtype)
        kernel2d = kernel2d.expand(channels, 1, kernelSize, kernelSize)
        img = img.unsqueeze(0)
        blurredImg = F.conv2d(img, kernel2d, padding=kernelSize // 2, groups=channels)
        return blurredImg.squeeze(0)

    def applyFilter(self, img, kernel):
        img = img.unsqueeze(0).unsqueeze(0)
        filteredImg = F.conv2d(img, kernel, padding=1)
        return filteredImg.squeeze(0).squeeze(0)

    def __call__(
        self,
        image: torch.Tensor,
        thinEdges: bool = True,
        Gaussian: bool = True,
        darkenStrength: float = 0.8,
    ) -> torch.Tensor:
        """
        Darken the lines of an anime-style image to give it a slight perceived sharpness boost.

        Args:
            image (torch.Tensor): Input image tensor of shape (H, W, C) with values in the range [0, 1].
            thinEdges (bool): If True, apply line thinning.
            Gaussian (bool): If True, apply Gaussian blur to soften the edges.
            darkenStrength (float): Strength of the darkening effect.

        Returns:
            torch.Tensor: Image tensor with enhanced lines.
        """
        if not isinstance(image, torch.Tensor):
            raise ValueError("Input image must be a torch.Tensor")
        if image.dim() != 3 or image.size(2) != 3:
            raise ValueError("Input image must have shape (H, W, C) with 3 channels")

        if self.ISCUDA:
            return self._CudaWorkflow(image, thinEdges, Gaussian, darkenStrength)
        else:
            return self._CPUWorkflow(image, thinEdges, Gaussian, darkenStrength)

    def _CudaWorkflow(self, image, thinEdges, Gaussian, darkenStrength):
        with torch.cuda.stream(self.normStream):
            image = image.half() if self.half else image.float()
            image = image.permute(2, 0, 1)

            grayscale = torch.tensordot(image, self.weightsLocal, dims=([0], [0]))

            edgesX = self.applyFilter(grayscale, self.sobelXLocal)
            edgesY = self.applyFilter(grayscale, self.sobelYLocal)
            edges = torch.sqrt(edgesX**2 + edgesY**2)

            edges = (edges - edges.min()) / (edges.max() - edges.min())

            if thinEdges:
                edges = edges.unsqueeze(0).unsqueeze(0)
                thinnedEdges = -F.max_pool2d(-edges, kernel_size=3, stride=1, padding=1)
                thinnedEdges = thinnedEdges.squeeze(0).squeeze(0)
            else:
                thinnedEdges = edges

            if Gaussian:
                softenedEdges = self.gaussianBlur(thinnedEdges.unsqueeze(0)).squeeze(0)
            else:
                softenedEdges = thinnedEdges

            enhancedImage = image.sub_(darkenStrength * softenedEdges.unsqueeze(0))
            enhancedImage = enhancedImage.permute(1, 2, 0).clamp(0, 1)
        self.normStream.synchronize()
        return enhancedImage

    def _CPUWorkflow(self, image, thinEdges, Gaussian, darkenStrength):
        image = image.half() if self.half else image.float()
        image = image.permute(2, 0, 1)

        grayscale = torch.tensordot(image, self.weightsLocal, dims=([0], [0]))

        edgesX = self.applyFilter(grayscale, self.sobelXLocal)
        edgesY = self.applyFilter(grayscale, self.sobelYLocal)
        edges = torch.sqrt(edgesX**2 + edgesY**2)

        edges = (edges - edges.min()) / (edges.max() - edges.min())

        if thinEdges:
            edges = edges.unsqueeze(0).unsqueeze(0)
            thinnedEdges = -F.max_pool2d(-edges, kernel_size=3, stride=1, padding=1)
            thinnedEdges = thinnedEdges.squeeze(0).squeeze(0)
        else:
            thinnedEdges = edges

        if Gaussian:
            softenedEdges = self.gaussianBlur(thinnedEdges.unsqueeze(0)).squeeze(0)
        else:
            softenedEdges = thinnedEdges

        enhancedImage = image.sub_(darkenStrength * softenedEdges.unsqueeze(0))
        return enhancedImage.permute(1, 2, 0).clamp(0, 1)
