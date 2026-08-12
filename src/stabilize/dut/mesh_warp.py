"""Native-resolution mesh warp.

Upstream (utils/WarpUtils.py) warps at the 640x480 model resolution and the
demo resizes the result back, costing sharpness. Here the model-resolution
grid displacement is scaled onto the source video's resolution and the
per-cell homography warp is evaluated directly on native pixels. Semantics
match upstream mesh_warp_frame: destination = source vertices + motion,
per-cell homography via HomoCalc, /W*2-1 normalization, align_corners=True,
zero padding.
"""

import torch
import torch.nn.functional as F

from . import config as cfg
from .projection import HomoCalc


class MeshWarper:
    """Per-frame mesh warp bound to one output resolution and device."""

    def __init__(self, width, height, device):
        self.width = width
        self.height = height
        self.device = device
        self.scaleX = width / float(cfg.WIDTH)
        self.scaleY = height / float(cfg.HEIGHT)

        gridW = cfg.WIDTH // cfg.PIXELS  # 40 vertices
        gridH = cfg.HEIGHT // cfg.PIXELS  # 30 vertices
        vx = torch.arange(gridW, device=device).float() * (cfg.PIXELS * self.scaleX)
        vy = torch.arange(gridH, device=device).float() * (cfg.PIXELS * self.scaleY)
        gy, gx = torch.meshgrid(vy, vx, indexing="ij")
        self.srcGrid = torch.stack([gx, gy], 0)  # [2, G_H, G_W] native px

        px = torch.arange(width, device=device).float()
        py = torch.arange(height, device=device).float()
        self.pixY, self.pixX = torch.meshgrid(py, px, indexing="ij")  # [H, W]
        # Cell lookup, clamped like upstream HomoProj (right/bottom edge strips
        # extrapolate the last cell's homography).
        self.cellX = (
            (self.pixX / (cfg.PIXELS * self.scaleX)).long().clamp(max=gridW - 2)
        )
        self.cellY = (
            (self.pixY / (cfg.PIXELS * self.scaleY)).long().clamp(max=gridH - 2)
        )

    def warp(self, frame, xMotion, yMotion):
        """
        @param frame [1, C, H, W] float tensor on self.device
        @param xMotion/yMotion [G_H, G_W] model-resolution px displacement
        @return warped frame [1, C, H, W]
        """
        des = self.srcGrid + torch.stack(
            [xMotion * self.scaleX, yMotion * self.scaleY], 0
        )
        homo = HomoCalc(self.srcGrid, des)  # [3, 3, G_H-1, G_W-1]

        h = homo[:, :, self.cellY, self.cellX]  # [3, 3, H, W]
        xProj = self.pixX * h[0, 0] + self.pixY * h[0, 1] + h[0, 2]
        yProj = self.pixX * h[1, 0] + self.pixY * h[1, 1] + h[1, 2]
        z = self.pixX * h[2, 0] + self.pixY * h[2, 1] + h[2, 2]
        xProj = xProj / z
        yProj = yProj / z

        grid = torch.stack(
            [
                xProj / self.width * 2.0 - 1.0,
                yProj / self.height * 2.0 - 1.0,
            ],
            -1,
        ).unsqueeze(0)  # [1, H, W, 2]
        return F.grid_sample(frame, grid, align_corners=True)
