import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal
import math


VARIANTCONFIG = {
        'lite': {
            'luma_scale': 0.5,
            'sobel_scale': 0.25,
            'spatial_sigma_factor': 0.5,
            'sobel_offset': 0.5,
            'kernel_offset': 0.25,
        },
        'medium': {
            'luma_scale': 1.0,
            'sobel_scale': 0.5,
            'spatial_sigma_factor': 1.0,
            'sobel_offset': 1.0,
            'kernel_offset': 0.5,
        },
        'heavy': {
            'luma_scale': 1.0,
            'sobel_scale': 1.0,
            'spatial_sigma_factor': 2.0,
            'sobel_offset': 1.0,
            'kernel_offset': 1.0,
        },
    }

class LineThinner(nn.Module):
    def __init__(
        self,
        variant: Literal['lite', 'medium', 'heavy'] = 'medium',
        strength: float = 0.1,
        iterations: int = 6,
    ):
        super().__init__()
        self.variant = variant
        self.config = VARIANTCONFIG[variant]
        self.strength = strength
        self.iterations = iterations

        self.register_buffer(
            'luma_weights',
            torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        )
    
    def _getLuma(self, rgb: torch.Tensor) -> torch.Tensor:
        """Convert RGB to luminance. Input: [B,3,H,W], Output: [B,1,H,W]"""
        weights = self.luma_weights.to(dtype=rgb.dtype, device=rgb.device)
        return (rgb * weights).sum(dim=1, keepdim=True)
    
    def _sobelX(self, luma: torch.Tensor, offset: float) -> torch.Tensor:
        """
        Horizontal Sobel gradient.
        Returns [xgrad, ygrad] stacked in channel dimension.
        """
        B, C, H, W = luma.shape

        device = luma.device

        yCords = torch.linspace(-1, 1, H, device=device, dtype=luma.dtype)
        xCords = torch.linspace(-1, 1, W, device=device, dtype=luma.dtype)
        gridY, gridX = torch.meshgrid(yCords, xCords, indexing='ij')

        dx = torch.tensor(2.0 * offset / W, device=device, dtype=luma.dtype)
        
        gridLeft = torch.stack([gridX - dx, gridY], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        left = F.grid_sample(luma, gridLeft, mode='bilinear', padding_mode='border', align_corners=True)
        
        center = luma

        gridRight = torch.stack([gridX + dx, gridY], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        right = F.grid_sample(luma, gridRight, mode='bilinear', padding_mode='border', align_corners=True)

        xgrad = -left + right
        ygrad = left + 2 * center + right
        
        return torch.cat([xgrad, ygrad], dim=1)  # [B, 2, H, W]
    
    def _sobelY(self, sobelXY: torch.Tensor, offset: float) -> torch.Tensor:
        """
        Vertical Sobel gradient + magnitude computation.
        Input: [B, 2, H, W] with [xgrad, ygrad]
        Output: [B, 1, H, W] edge magnitude
        """
        B, _, H, W = sobelXY.shape
        device = sobelXY.device
        
        yCords = torch.linspace(-1, 1, H, device=device, dtype=sobelXY.dtype)
        xCords = torch.linspace(-1, 1, W, device=device, dtype=sobelXY.dtype)
        gridY, gridX = torch.meshgrid(yCords, xCords, indexing='ij')

        dy = torch.tensor(2.0 * offset / H, device=device, dtype=sobelXY.dtype)
        
        gridTop = torch.stack([gridX, gridY - dy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        top = F.grid_sample(sobelXY, gridTop, mode='bilinear', padding_mode='border', align_corners=True)
        

        center = sobelXY

        gridBottom = torch.stack([gridX, gridY + dy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        bottom = F.grid_sample(sobelXY, gridBottom, mode='bilinear', padding_mode='border', align_corners=True)
        
        tx, ty = top[:, 0:1], top[:, 1:2]
        cx, _ = center[:, 0:1], center[:, 1:2]
        bx, by = bottom[:, 0:1], bottom[:, 1:2]
        
        xgrad = (tx + 2 * cx + bx) / 8.0
        ygrad = (-ty + by) / 8.0
        
        norm = torch.sqrt(xgrad * xgrad + ygrad * ygrad)
        return torch.pow(norm, 0.7)
    
    def _createGaussianKernel(self, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Create 1D Gaussian kernel with specified dtype and device."""
        kernelSize = max(int(math.ceil(sigma * 2.0)), 1) * 2 + 1
        half_size = kernelSize // 2
        
        x = torch.arange(-half_size, half_size + 1, dtype=dtype, device=device)
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel = kernel / kernel.sum()
        
        return kernel
    
    def _gaussianBlur(self, edge_map: torch.Tensor, sigma: float) -> torch.Tensor:
        """Apply separable Gaussian blur."""
        if sigma < 0.1:
            return edge_map
            
        kernel = self._createGaussianKernel(sigma, edge_map.device, edge_map.dtype)
        kernelSize = len(kernel)
        
        kernel_h = kernel.view(1, 1, 1, kernelSize)
        padded = F.pad(edge_map, (kernelSize // 2, kernelSize // 2, 0, 0), mode='replicate')
        blurred = F.conv2d(padded, kernel_h, groups=edge_map.shape[1])
        
        kernel_v = kernel.view(1, 1, kernelSize, 1)
        padded = F.pad(blurred, (0, 0, kernelSize // 2, kernelSize // 2), mode='replicate')
        blurred = F.conv2d(padded, kernel_v, groups=edge_map.shape[1])
        
        return blurred
    
    def _kernelX(self, edge_map: torch.Tensor, offset: float) -> torch.Tensor:
        """Second gradient pass - horizontal."""
        B, C, H, W = edge_map.shape
        device = edge_map.device
        
        yCords = torch.linspace(-1, 1, H, device=device, dtype=edge_map.dtype)
        xCords = torch.linspace(-1, 1, W, device=device, dtype=edge_map.dtype)
        gridY, gridX = torch.meshgrid(yCords, xCords, indexing='ij')
        
        dx = torch.tensor(2.0 * offset / W, device=device, dtype=edge_map.dtype)
        
        gridLeft = torch.stack([gridX - dx, gridY], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        left = F.grid_sample(edge_map, gridLeft, mode='bilinear', padding_mode='border', align_corners=True)
        
        center = edge_map
        
        gridRight = torch.stack([gridX + dx, gridY], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        right = F.grid_sample(edge_map, gridRight, mode='bilinear', padding_mode='border', align_corners=True)
        
        xgrad = -left + right
        ygrad = left + 2 * center + right
        
        return torch.cat([xgrad, ygrad], dim=1)  # [B, 2, H, W]
    
    def _kernelY(self, kernel_xy: torch.Tensor, offset: float) -> torch.Tensor:
        """Second gradient pass - vertical."""
        B, C, H, W = kernel_xy.shape
        device = kernel_xy.device
        
        yCords = torch.linspace(-1, 1, H, device=device, dtype=kernel_xy.dtype)
        xCords = torch.linspace(-1, 1, W, device=device, dtype=kernel_xy.dtype)
        gridY, gridX = torch.meshgrid(yCords, xCords, indexing='ij')
        
        dy = torch.tensor(2.0 * offset / H, device=device, dtype=kernel_xy.dtype)
        
        gridTop = torch.stack([gridX, gridY - dy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        top = F.grid_sample(kernel_xy, gridTop, mode='bilinear', padding_mode='border', align_corners=True)
        
        center = kernel_xy
        
        gridBottom = torch.stack([gridX, gridY + dy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        bottom = F.grid_sample(kernel_xy, gridBottom, mode='bilinear', padding_mode='border', align_corners=True)
        
        tx, ty = top[:, 0:1], top[:, 1:2]
        cx, _ = center[:, 0:1], center[:, 1:2]
        bx, by = bottom[:, 0:1], bottom[:, 1:2]
        
        xgrad = (tx + 2 * cx + bx) / 8.0
        ygrad = (-ty + by) / 8.0
        
        return torch.cat([xgrad, ygrad], dim=1)  # [B, 2, H, W]
    
    def _applyWarp(
        self,
        image: torch.Tensor,
        gradient_field: torch.Tensor,
        height: int,
    ) -> torch.Tensor:
        """Apply warp displacement based on gradient field."""
        B, C, H, W = image.shape
        device = image.device
        
        if gradient_field.shape[2] != H or gradient_field.shape[3] != W:
            gradient_field = F.interpolate(
                gradient_field, size=(H, W), mode='bilinear', align_corners=True
            )
        
        yCords = torch.linspace(-1, 1, H, device=device, dtype=image.dtype)
        xCords = torch.linspace(-1, 1, W, device=device, dtype=image.dtype)
        gridY, gridX = torch.meshgrid(yCords, xCords, indexing='ij')
        grid = torch.stack([gridX, gridY], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        
        dx = torch.tensor(2.0 / W, device=device, dtype=image.dtype)
        dy = torch.tensor(2.0 / H, device=device, dtype=image.dtype)
        
        relstr = height / 1080.0 * self.strength
        
        pos = grid.clone()
        for _ in range(self.iterations):
            dn = F.grid_sample(
                gradient_field, pos, mode='bilinear', 
                padding_mode='border', align_corners=True
            )  # [B, 2, H, W]
            
            dn = dn.permute(0, 2, 3, 1)  # [B, H, W, 2]
            
            dn_len = torch.sqrt(dn[..., 0:1]**2 + dn[..., 1:2]**2) + 0.01
            dn_normalized = dn / dn_len
            
            dd = dn_normalized * torch.tensor([dx.item(), dy.item()], device=device, dtype=image.dtype) * relstr
            
            pos = pos - dd
        
        return F.grid_sample(image, pos, mode='bilinear', padding_mode='border', align_corners=True)
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply line thinning to input image.
        
        Args:
            image: Input tensor [B, 3, H, W] in range [0, 1]
            
        Returns:
            Processed tensor [B, 3, H, W] in range [0, 1]
        """
        _, C, H, W = image.shape
        assert C == 3, "Expected RGB image with 3 channels"
        
        config = self.config
        
        luma_h = int(H * config['luma_scale'])
        luma_w = int(W * config['luma_scale'])
        sobel_h = int(H * config['sobel_scale'])
        sobel_w = int(W * config['sobel_scale'])
        
        if config['luma_scale'] < 1.0:
            image_scaled = F.interpolate(image, size=(luma_h, luma_w), mode='bilinear', align_corners=True)
            luma = self._getLuma(image_scaled)
        else:
            luma = self._getLuma(image)
        
        if self.variant == 'veryfast':
            luma = F.interpolate(luma, size=(sobel_h, sobel_w), mode='bilinear', align_corners=True)
        elif config['sobel_scale'] < 1.0:
            luma = F.interpolate(luma, size=(sobel_h, sobel_w), mode='bilinear', align_corners=True)
        
        sobelXY = self._sobelX(luma, config['sobel_offset'])
        edge_map = self._sobelY(sobelXY, config['sobel_offset'])
        
        spatial_sigma = config['spatial_sigma_factor'] * H / 1080.0
        edge_map = self._gaussianBlur(edge_map, spatial_sigma)
        
        kernel_xy = self._kernelX(edge_map, config['kernel_offset'])
        gradient_field = self._kernelY(kernel_xy, config['kernel_offset'])
        
        result = self._applyWarp(image, gradient_field, H)
        
        return result


class LineThin():
    def __init__(
        self,
        variant: Literal['lite', 'medium', 'heavy'] = 'medium',
        device: str = 'cuda',
        half: bool = False,
    ) -> torch.Tensor:
        self.variant = variant
        self.device = device
        self.half = half

        self.precision = torch.float16 if half else torch.float32
        self.model = LineThinner(variant=variant).to(device=device, dtype=self.precision)

        self.normStream = None
        if device == 'cuda':
            self.normStream = torch.cuda.Stream()

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply line thinning to input image.
        
        Args:
            image: Input tensor [B, 3, H, W] in range [0, 1]
            
        Returns:
            Processed tensor [B, 3, H, W] in range [0, 1]
        """
        if self.normStream is not None:
            with torch.cuda.stream(self.normStream):
                output = self.model(image.to(device=self.device, dtype=self.precision))
            self.normStream.synchronize()
        else:
            image = image.to(device=self.device, dtype=self.precision)
            output = self.model(image)

        return output

