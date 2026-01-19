"""
RIFE DirectML Architectures - Consolidated DirectML-compatible RIFE models.

This module provides DirectML-compatible versions of all RIFE architectures
using decomposed grid_sample operations.
"""

import torch
import torch.nn as nn
from torch.nn.functional import interpolate
import math

from .grid_sample_directml import grid_sample_directml


# =============================================================================
# Shared utilities
# =============================================================================


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        ),
        nn.LeakyReLU(0.2, True),
    )


class Head8(nn.Module):
    """Feature encoder that produces 8-channel features (for RIFE 4.15-4.22)."""

    def __init__(self):
        super(Head8, self).__init__()
        self.cnn0 = nn.Conv2d(3, 32, 3, 2, 1)
        self.cnn1 = nn.Conv2d(32, 32, 3, 1, 1)
        self.cnn2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(32, 8, 4, 2, 1)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x, feat=False):
        x0 = self.cnn0(x)
        x = self.relu(x0)
        x1 = self.cnn1(x)
        x = self.relu(x1)
        x2 = self.cnn2(x)
        x = self.relu(x2)
        x3 = self.cnn3(x)
        if feat:
            return [x0, x1, x2, x3]
        return x3


class Head4(nn.Module):
    """Feature encoder that produces 4-channel features (for RIFE 4.22-lite, 4.25)."""

    def __init__(self):
        super(Head4, self).__init__()
        self.cnn0 = nn.Conv2d(3, 16, 3, 2, 1)
        self.cnn1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(16, 4, 4, 2, 1)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x, feat=False):
        x0 = self.cnn0(x)
        x = self.relu(x0)
        x1 = self.cnn1(x)
        x = self.relu(x1)
        x2 = self.cnn2(x)
        x = self.relu(x2)
        x3 = self.cnn3(x)
        if feat:
            return [x0, x1, x2, x3]
        return x3


class ResConv(nn.Module):
    def __init__(self, c):
        super(ResConv, self).__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, padding=1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.relu(self.conv(x) * self.beta + x)


class IFBlock415(nn.Module):
    """IFBlock for RIFE 4.15-4.21 (returns flow and mask only)."""

    def __init__(self, in_planes, c=64):
        super(IFBlock415, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1),
            conv(c // 2, c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(*[ResConv(c) for _ in range(8)])
        self.lastconv = nn.Sequential(
            nn.ConvTranspose2d(c, 4 * 6, 4, 2, 1), nn.PixelShuffle(2)
        )

    def forward(self, x, scale=1):
        if scale != 1:
            x = interpolate(
                x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False
            )
        feat = self.conv0(x)
        feat = self.convblock(feat)
        tmp = self.lastconv(feat)
        if scale != 1:
            tmp = interpolate(
                tmp, scale_factor=scale, mode="bilinear", align_corners=False
            )
        flow = tmp[:, :4]
        mask = tmp[:, 4:5]
        if scale != 1:
            flow = flow * scale
        return flow, mask


class IFBlock422(nn.Module):
    """IFBlock for RIFE 4.22+ (returns flow, mask, and feat)."""

    def __init__(self, in_planes, c=64):
        super(IFBlock422, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1),
            conv(c // 2, c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(*[ResConv(c) for _ in range(8)])
        self.lastconv = nn.Sequential(
            nn.ConvTranspose2d(c, 4 * 13, 4, 2, 1), nn.PixelShuffle(2)
        )

    def forward(self, x, scale=1):
        if scale != 1:
            x = interpolate(
                x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False
            )
        feat = self.conv0(x)
        feat = self.convblock(feat)
        tmp = self.lastconv(feat)
        if scale != 1:
            tmp = interpolate(
                tmp, scale_factor=scale, mode="bilinear", align_corners=False
            )
        flow = tmp[:, :4]
        mask = tmp[:, 4:5]
        feat = tmp[:, 5:]
        if scale != 1:
            flow = flow * scale
        return flow, mask, feat


# =============================================================================
# RIFE 4.6 DirectML (non-head)
# =============================================================================
class ResConv46(nn.Module):
    def __init__(self, c, dilation=1):
        super(ResConv46, self).__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.relu(self.conv(x) * self.beta + x)


class IFBlock46(nn.Module):
    def __init__(self, in_planes, c=64):
        super(IFBlock46, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1),
            conv(c // 2, c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(*[ResConv46(c) for _ in range(8)])
        self.lastconv = nn.Sequential(
            nn.ConvTranspose2d(c, 4 * 6, 4, 2, 1), nn.PixelShuffle(2)
        )

    def forward(self, x, flow=None, scale=1):
        x = interpolate(x, scale_factor=1.0 / scale, mode="bilinear")
        if flow is not None:
            flow = (
                interpolate(flow, scale_factor=1.0 / scale, mode="bilinear") * 1.0 / scale
            )
            x = torch.cat((x, flow), 1)

        feat = self.conv0(x)
        feat = self.convblock(feat)
        tmp = self.lastconv(feat)
        tmp = interpolate(tmp, scale_factor=scale, mode="bilinear")

        flow = tmp[:, :4] * scale
        mask = tmp[:, 4:5]
        return flow, mask


def warp_directml_46(tenInput, tenFlow, tenFlowDiv, backWarp):
    # Normalize flow (pixel units -> grid units) then build sampling grid.
    tenFlow = torch.cat(
        [tenFlow[:, 0:1] / tenFlowDiv[0], tenFlow[:, 1:2] / tenFlowDiv[1]], 1
    )
    g = (backWarp + tenFlow).permute(0, 2, 3, 1)
    return grid_sample_directml(
        input_tensor=tenInput,
        grid=g,
        padding_mode="border",
        align_corners=True,
    )


class IFNet_46(nn.Module):
    """RIFE 4.6 DirectML - 4 blocks, flow/mask based, no head features."""

    def __init__(
        self,
        scale=1.0,
        ensemble=False,
        dtype=torch.float32,
        device="cuda",
        width=1920,
        height=1080,
    ):
        super(IFNet_46, self).__init__()
        self.block0 = IFBlock46(7, c=192)
        self.block1 = IFBlock46(8 + 4, c=128)
        self.block2 = IFBlock46(8 + 4, c=96)
        self.block3 = IFBlock46(8 + 4, c=64)

        self.scaleList = [8 / scale, 4 / scale, 2 / scale, 1 / scale]
        self.ensemble = ensemble
        self.dtype = dtype
        self.device = device
        self.width = width
        self.height = height
        self.blocks = [self.block0, self.block1, self.block2, self.block3]

        tmp = max(32, int(32 / 1.0))
        self.pw = math.ceil(self.width / tmp) * tmp
        self.ph = math.ceil(self.height / tmp) * tmp
        self.padding = (0, self.pw - self.width, 0, self.ph - self.height)

        # Flow divisor for normalizing optical flow to [-1, 1] grid units.
        self.register_buffer(
            "tenFlowDiv",
            torch.tensor(
                [(self.pw - 1.0) / 2.0, (self.ph - 1.0) / 2.0],
                dtype=dtype,
            ),
        )

        tenHorizontal = (
            torch.linspace(-1.0, 1.0, self.pw, dtype=dtype)
            .view(1, 1, 1, self.pw)
            .expand(-1, -1, self.ph, -1)
        )
        tenVertical = (
            torch.linspace(-1.0, 1.0, self.ph, dtype=dtype)
            .view(1, 1, self.ph, 1)
            .expand(-1, -1, -1, self.pw)
        )
        self.register_buffer("backWarp", torch.cat([tenHorizontal, tenVertical], 1))

    def warp(self, tenInput, tenFlow):
        return warp_directml_46(tenInput, tenFlow, self.tenFlowDiv, self.backWarp)

    def forward(self, img0, img1, timestep):
        warpedImg0, warpedImg1 = img0, img1
        flow = mask = None

        for i, block in enumerate(self.blocks):
            scale = self.scaleList[i]
            if flow is None:
                flow, mask = block(
                    torch.cat((img0[:, :3], img1[:, :3], timestep), 1),
                    None,
                    scale=scale,
                )

                if self.ensemble:
                    f1, m1 = block(
                        torch.cat((img1[:, :3], img0[:, :3], 1 - timestep), 1),
                        None,
                        scale=scale,
                    )
                    flow = (flow + torch.cat((f1[:, 2:4], f1[:, :2]), 1)) / 2
                    mask = (mask - m1) / 2
            else:
                f0, m0 = block(
                    torch.cat((warpedImg0[:, :3], warpedImg1[:, :3], timestep, mask), 1),
                    flow,
                    scale=scale,
                )

                if self.ensemble:
                    f1, m1 = block(
                        torch.cat(
                            (warpedImg1[:, :3], warpedImg0[:, :3], 1 - timestep, -mask),
                            1,
                        ),
                        torch.cat((flow[:, 2:4], flow[:, :2]), 1),
                        scale=scale,
                    )
                    f0 = (f0 + torch.cat((f1[:, 2:4], f1[:, :2]), 1)) / 2
                    m0 = (m0 - m1) / 2

                flow = flow + f0
                mask = mask + m0

            warpedImg0 = self.warp(img0, flow[:, :2])
            warpedImg1 = self.warp(img1, flow[:, 2:4])

        temp = torch.sigmoid(mask)
        return (warpedImg0 * temp + warpedImg1 * (1 - temp))[
            :, :, : self.height, : self.width
        ]


# =============================================================================
# RIFE 4.15/4.17/4.18 DirectML (8ch features, 4 blocks, no feat output)
# =============================================================================
class IFNet_415(nn.Module):
    """RIFE 4.15/4.17/4.18 DirectML - 4 blocks, 8ch features, no feat passthrough."""

    def __init__(
        self,
        scale=1.0,
        ensemble=False,
        dtype=torch.float32,
        device="cuda",
        width=1920,
        height=1080,
    ):
        super(IFNet_415, self).__init__()
        self.block0 = IFBlock415(7 + 16, c=192)
        self.block1 = IFBlock415(8 + 4 + 16, c=128)
        self.block2 = IFBlock415(8 + 4 + 16, c=96)
        self.block3 = IFBlock415(8 + 4 + 16, c=64)
        self.encode = Head8()
        self.device = device
        self.dtype = dtype
        self.scaleList = [8 / scale, 4 / scale, 2 / scale, 1 / scale]
        self.width = width
        self.height = height
        self.blocks = [self.block0, self.block1, self.block2, self.block3]

        tmp = max(32, int(32 / 1.0))
        self.pw = math.ceil(self.width / tmp) * tmp
        self.ph = math.ceil(self.height / tmp) * tmp
        self.padding = (0, self.pw - self.width, 0, self.ph - self.height)

        hMul = 2 / (self.pw - 1)
        vMul = 2 / (self.ph - 1)
        self.register_buffer(
            "tenFlowMul", torch.tensor([hMul, vMul], dtype=dtype).reshape(1, 2, 1, 1)
        )
        horizontal = (
            (torch.arange(self.pw, dtype=dtype) * hMul - 1)
            .reshape(1, 1, 1, -1)
            .expand(-1, -1, self.ph, -1)
        )
        vertical = (
            (torch.arange(self.ph, dtype=dtype) * vMul - 1)
            .reshape(1, 1, -1, 1)
            .expand(-1, -1, -1, self.pw)
        )
        self.register_buffer("backWarp", torch.cat([horizontal, vertical], dim=1))

    def forward(self, img0, img1, timestep):
        imgs = torch.cat([img0, img1], dim=1)
        imgs2 = imgs.view(2, 3, self.ph, self.pw)
        f0 = self.encode(img0[:, :3])
        f1 = self.encode(img1[:, :3])
        fs = torch.cat([f0, f1], dim=1)
        fs2 = fs.view(2, 8, self.ph, self.pw)

        wimg = None
        wf = None

        flows = None
        for block, scale in zip(self.blocks, self.scaleList):
            if flows is None:
                temp = torch.cat((imgs, fs, timestep), 1)
                flows, mask = block(temp, scale=scale)
            else:
                temp = torch.cat(
                    (
                        wimg,
                        wf,
                        timestep,
                        mask,
                        flows * (1 / scale) if scale != 1 else flows,
                    ),
                    1,
                )  # noqa
                fds, mask = block(temp, scale=scale)
                flows = flows + fds

            precomp = (
                self.backWarp
                + flows.reshape((2, 2, self.ph, self.pw)) * self.tenFlowMul
            ).permute(0, 2, 3, 1)
            if scale == 1:
                warpedImgs = grid_sample_directml(
                    imgs2, precomp, padding_mode="border", align_corners=True
                )
            else:
                warps = grid_sample_directml(
                    torch.cat((imgs2, fs2), 1),
                    precomp,
                    padding_mode="border",
                    align_corners=True,
                )
                wimg, wf = torch.split(warps, [3, 8], dim=1)
                wimg = wimg.reshape(1, 6, self.ph, self.pw)
                wf = wf.reshape(1, 16, self.ph, self.pw)

        mask = torch.sigmoid(mask)
        warpedImg0, warpedImg1 = torch.split(warpedImgs, [1, 1])
        return (warpedImg0 * mask + warpedImg1 * (1 - mask))[
            :, :, : self.height, : self.width
        ]


# =============================================================================
# RIFE 4.22 DirectML (full)
# =============================================================================
class IFNet_422(nn.Module):
    """RIFE 4.22 DirectML - 4 blocks, 8ch features, feat passthrough."""

    def __init__(
        self,
        scale=1.0,
        ensemble=False,
        dtype=torch.float32,
        device="cuda",
        width=1920,
        height=1080,
    ):
        super(IFNet_422, self).__init__()
        self.block0 = IFBlock422(7 + 16, c=256)
        self.block1 = IFBlock422(8 + 4 + 16 + 8, c=192)
        self.block2 = IFBlock422(8 + 4 + 16 + 8, c=96)
        self.block3 = IFBlock422(8 + 4 + 16 + 8, c=48)
        self.encode = Head8()
        self.device = device
        self.dtype = dtype
        self.scaleList = [8 / scale, 4 / scale, 2 / scale, 1 / scale]
        self.ensemble = ensemble
        self.width = width
        self.height = height
        self.blocks = [self.block0, self.block1, self.block2, self.block3]

        tmp = max(32, int(32 / 1.0))
        self.pw = math.ceil(self.width / tmp) * tmp
        self.ph = math.ceil(self.height / tmp) * tmp
        self.padding = (0, self.pw - self.width, 0, self.ph - self.height)

        hMul = 2 / (self.pw - 1)
        vMul = 2 / (self.ph - 1)
        self.register_buffer(
            "tenFlowMul", torch.tensor([hMul, vMul], dtype=dtype).reshape(1, 2, 1, 1)
        )
        horizontal = (
            (torch.arange(self.pw, dtype=dtype) * hMul - 1)
            .reshape(1, 1, 1, -1)
            .expand(-1, -1, self.ph, -1)
        )
        vertical = (
            (torch.arange(self.ph, dtype=dtype) * vMul - 1)
            .reshape(1, 1, -1, 1)
            .expand(-1, -1, -1, self.pw)
        )
        self.register_buffer("backWarp", torch.cat([horizontal, vertical], dim=1))

    def forward(self, img0, img1, timestep):
        imgs = torch.cat([img0, img1], dim=1)
        imgs2 = imgs.view(2, 3, self.ph, self.pw)

        f0 = self.encode(img0[:, :3])
        f1 = self.encode(img1[:, :3])
        fs = torch.cat([f0, f1], dim=1)
        fs2 = fs.view(2, 8, self.ph, self.pw)

        wimg = None
        wf = None

        flows = None
        for block, scale in zip(self.blocks, self.scaleList):
            if flows is None:
                temp = torch.cat((imgs, fs, timestep), 1)
                flows, mask, feat = block(temp, scale=scale)
            else:
                temp = torch.cat(
                    (
                        wimg,
                        wf,
                        timestep,
                        mask,
                        feat,
                        flows * (1 / scale) if scale != 1 else flows,
                    ),
                    1,
                )  # noqa: F821
                fds, mask, feat = block(temp, scale=scale)
                flows = flows + fds

            precomp = (
                self.backWarp
                + flows.reshape((2, 2, self.ph, self.pw)) * self.tenFlowMul
            ).permute(0, 2, 3, 1)

            if scale == 1:
                warpedImgs = grid_sample_directml(
                    imgs2, precomp, padding_mode="border", align_corners=True
                )
            else:
                warps = grid_sample_directml(
                    torch.cat((imgs2, fs2), 1),
                    precomp,
                    padding_mode="border",
                    align_corners=True,
                )
                wimg, wf = torch.split(warps, [3, 8], dim=1)
                wimg = wimg.reshape(1, 6, self.ph, self.pw)
                wf = wf.reshape(1, 16, self.ph, self.pw)

        mask = torch.sigmoid(mask)
        warpedImg0, warpedImg1 = torch.split(warpedImgs, [1, 1])
        return (warpedImg0 * mask + warpedImg1 * (1 - mask))[
            :, :, : self.height, : self.width
        ]


# =============================================================================
# RIFE 4.20/4.21 DirectML (8ch features, 4 blocks, larger first block)
# =============================================================================
class IFNet_420(nn.Module):
    """RIFE 4.20/4.21 DirectML - Same as 415 but with larger block0 (c=384)."""

    def __init__(
        self,
        scale=1.0,
        ensemble=False,
        dtype=torch.float32,
        device="cuda",
        width=1920,
        height=1080,
    ):
        super(IFNet_420, self).__init__()
        self.block0 = IFBlock415(7 + 16, c=384)
        self.block1 = IFBlock415(8 + 4 + 16, c=192)
        self.block2 = IFBlock415(8 + 4 + 16, c=96)
        self.block3 = IFBlock415(8 + 4 + 16, c=48)
        self.encode = Head8()
        self.device = device
        self.dtype = dtype
        self.scaleList = [8 / scale, 4 / scale, 2 / scale, 1 / scale]
        self.width = width
        self.height = height
        self.blocks = [self.block0, self.block1, self.block2, self.block3]

        tmp = max(32, int(32 / 1.0))
        self.pw = math.ceil(self.width / tmp) * tmp
        self.ph = math.ceil(self.height / tmp) * tmp
        self.padding = (0, self.pw - self.width, 0, self.ph - self.height)

        hMul = 2 / (self.pw - 1)
        vMul = 2 / (self.ph - 1)
        self.register_buffer(
            "tenFlowMul", torch.tensor([hMul, vMul], dtype=dtype).reshape(1, 2, 1, 1)
        )
        horizontal = (
            (torch.arange(self.pw, dtype=dtype) * hMul - 1)
            .reshape(1, 1, 1, -1)
            .expand(-1, -1, self.ph, -1)
        )
        vertical = (
            (torch.arange(self.ph, dtype=dtype) * vMul - 1)
            .reshape(1, 1, -1, 1)
            .expand(-1, -1, -1, self.pw)
        )
        self.register_buffer("backWarp", torch.cat([horizontal, vertical], dim=1))

    def forward(self, img0, img1, timestep):
        imgs = torch.cat([img0, img1], dim=1)
        imgs2 = imgs.view(2, 3, self.ph, self.pw)
        f0 = self.encode(img0[:, :3])
        f1 = self.encode(img1[:, :3])
        fs = torch.cat([f0, f1], dim=1)
        fs2 = fs.view(2, 8, self.ph, self.pw)

        wimg = None
        wf = None

        flows = None
        for block, scale in zip(self.blocks, self.scaleList):
            if flows is None:
                temp = torch.cat((imgs, fs, timestep), 1)
                flows, mask = block(temp, scale=scale)
            else:
                temp = torch.cat(
                    (
                        wimg,
                        wf,
                        timestep,
                        mask,
                        flows * (1 / scale) if scale != 1 else flows,
                    ),
                    1,
                )  # noqa
                fds, mask = block(temp, scale=scale)
                flows = flows + fds

            precomp = (
                self.backWarp
                + flows.reshape((2, 2, self.ph, self.pw)) * self.tenFlowMul
            ).permute(0, 2, 3, 1)
            if scale == 1:
                warpedImgs = grid_sample_directml(
                    imgs2, precomp, padding_mode="border", align_corners=True
                )
            else:
                warps = grid_sample_directml(
                    torch.cat((imgs2, fs2), 1),
                    precomp,
                    padding_mode="border",
                    align_corners=True,
                )
                wimg, wf = torch.split(warps, [3, 8], dim=1)
                wimg = wimg.reshape(1, 6, self.ph, self.pw)
                wf = wf.reshape(1, 16, self.ph, self.pw)

        mask = torch.sigmoid(mask)
        warpedImg0, warpedImg1 = torch.split(warpedImgs, [1, 1])
        return (warpedImg0 * mask + warpedImg1 * (1 - mask))[
            :, :, : self.height, : self.width
        ]


# =============================================================================
# RIFE 4.22-lite DirectML (4ch features, 4 blocks, feat passthrough)
# =============================================================================
class IFNet_422_lite(nn.Module):
    """RIFE 4.22-lite DirectML - 4 blocks, 4ch features."""

    def __init__(
        self,
        scale=1.0,
        ensemble=False,
        dtype=torch.float32,
        device="cuda",
        width=1920,
        height=1080,
    ):
        super(IFNet_422_lite, self).__init__()
        self.block0 = IFBlock422(7 + 8, c=192)
        self.block1 = IFBlock422(8 + 4 + 8 + 8, c=128)
        self.block2 = IFBlock422(8 + 4 + 8 + 8, c=64)
        self.block3 = IFBlock422(8 + 4 + 8 + 8, c=32)
        self.encode = Head4()
        self.device = device
        self.dtype = dtype
        self.scaleList = [8 / scale, 4 / scale, 2 / scale, 1 / scale]
        self.width = width
        self.height = height
        self.blocks = [self.block0, self.block1, self.block2, self.block3]

        tmp = max(32, int(32 / 1.0))
        self.pw = math.ceil(self.width / tmp) * tmp
        self.ph = math.ceil(self.height / tmp) * tmp
        self.padding = (0, self.pw - self.width, 0, self.ph - self.height)

        hMul = 2 / (self.pw - 1)
        vMul = 2 / (self.ph - 1)
        self.register_buffer(
            "tenFlowMul", torch.tensor([hMul, vMul], dtype=dtype).reshape(1, 2, 1, 1)
        )
        horizontal = (
            (torch.arange(self.pw, dtype=dtype) * hMul - 1)
            .reshape(1, 1, 1, -1)
            .expand(-1, -1, self.ph, -1)
        )
        vertical = (
            (torch.arange(self.ph, dtype=dtype) * vMul - 1)
            .reshape(1, 1, -1, 1)
            .expand(-1, -1, -1, self.pw)
        )
        self.register_buffer("backWarp", torch.cat([horizontal, vertical], dim=1))

    def forward(self, img0, img1, timestep):
        imgs = torch.cat([img0, img1], dim=1)
        imgs2 = imgs.view(2, 3, self.ph, self.pw)
        f0 = self.encode(img0[:, :3])
        f1 = self.encode(img1[:, :3])
        fs = torch.cat([f0, f1], dim=1)
        fs2 = fs.view(2, 4, self.ph, self.pw)

        wimg = None
        wf = None

        flows = None
        for block, scale in zip(self.blocks, self.scaleList):
            if flows is None:
                temp = torch.cat((imgs, fs, timestep), 1)
                flows, mask, feat = block(temp, scale=scale)
            else:
                temp = torch.cat(
                    (
                        wimg,
                        wf,
                        timestep,
                        mask,
                        feat,
                        flows * (1 / scale) if scale != 1 else flows,
                    ),
                    1,
                )  # noqa
                fds, mask, feat = block(temp, scale=scale)
                flows = flows + fds

            precomp = (
                self.backWarp
                + flows.reshape((2, 2, self.ph, self.pw)) * self.tenFlowMul
            ).permute(0, 2, 3, 1)
            if scale == 1:
                warpedImgs = grid_sample_directml(
                    imgs2, precomp, padding_mode="border", align_corners=True
                )
            else:
                warps = grid_sample_directml(
                    torch.cat((imgs2, fs2), 1),
                    precomp,
                    padding_mode="border",
                    align_corners=True,
                )
                wimg, wf = torch.split(warps, [3, 4], dim=1)
                wimg = wimg.reshape(1, 6, self.ph, self.pw)
                wf = wf.reshape(1, 8, self.ph, self.pw)

        mask = torch.sigmoid(mask)
        warpedImg0, warpedImg1 = torch.split(warpedImgs, [1, 1])
        return (warpedImg0 * mask + warpedImg1 * (1 - mask))[
            :, :, : self.height, : self.width
        ]


# =============================================================================
# RIFE 4.25 DirectML (4ch features, 5 blocks, mul=64)
# =============================================================================
class IFNet_425(nn.Module):
    """RIFE 4.25 DirectML - 5 blocks, 4ch features, mul=64."""

    def __init__(
        self,
        scale=1.0,
        ensemble=False,
        dtype=torch.float32,
        device="cuda",
        width=1920,
        height=1080,
    ):
        super(IFNet_425, self).__init__()
        self.block0 = IFBlock422(7 + 8, c=192)
        self.block1 = IFBlock422(8 + 4 + 8 + 8, c=128)
        self.block2 = IFBlock422(8 + 4 + 8 + 8, c=96)
        self.block3 = IFBlock422(8 + 4 + 8 + 8, c=64)
        self.block4 = IFBlock422(8 + 4 + 8 + 8, c=32)
        self.encode = Head4()
        self.device = device
        self.dtype = dtype
        self.scaleList = [16 / scale, 8 / scale, 4 / scale, 2 / scale, 1 / scale]
        self.width = width
        self.height = height
        self.blocks = [self.block0, self.block1, self.block2, self.block3, self.block4]

        tmp = max(64, int(64 / 1.0))
        self.pw = math.ceil(self.width / tmp) * tmp
        self.ph = math.ceil(self.height / tmp) * tmp
        self.padding = (0, self.pw - self.width, 0, self.ph - self.height)

        hMul = 2 / (self.pw - 1)
        vMul = 2 / (self.ph - 1)
        self.register_buffer(
            "tenFlowMul", torch.tensor([hMul, vMul], dtype=dtype).reshape(1, 2, 1, 1)
        )
        horizontal = (
            (torch.arange(self.pw, dtype=dtype) * hMul - 1)
            .reshape(1, 1, 1, -1)
            .expand(-1, -1, self.ph, -1)
        )
        vertical = (
            (torch.arange(self.ph, dtype=dtype) * vMul - 1)
            .reshape(1, 1, -1, 1)
            .expand(-1, -1, -1, self.pw)
        )
        self.register_buffer("backWarp", torch.cat([horizontal, vertical], dim=1))

    def forward(self, img0, img1, timestep):
        imgs = torch.cat([img0, img1], dim=1)
        imgs2 = imgs.view(2, 3, self.ph, self.pw)
        f0 = self.encode(img0[:, :3])
        f1 = self.encode(img1[:, :3])
        fs = torch.cat([f0, f1], dim=1)
        fs2 = fs.view(2, 4, self.ph, self.pw)

        wimg = None
        wf = None

        flows = None
        for block, scale in zip(self.blocks, self.scaleList):
            if flows is None:
                temp = torch.cat((imgs, fs, timestep), 1)
                flows, mask, feat = block(temp, scale=scale)
            else:
                temp = torch.cat(
                    (
                        wimg,
                        wf,
                        timestep,
                        mask,
                        feat,
                        flows * (1 / scale) if scale != 1 else flows,
                    ),
                    1,
                )  # noqa
                fds, mask, feat = block(temp, scale=scale)
                flows = flows + fds

            precomp = (
                self.backWarp
                + flows.reshape((2, 2, self.ph, self.pw)) * self.tenFlowMul
            ).permute(0, 2, 3, 1)
            if scale == 1:
                warpedImgs = grid_sample_directml(
                    imgs2, precomp, padding_mode="border", align_corners=True
                )
            else:
                warps = grid_sample_directml(
                    torch.cat((imgs2, fs2), 1),
                    precomp,
                    padding_mode="border",
                    align_corners=True,
                )
                wimg, wf = torch.split(warps, [3, 4], dim=1)
                wimg = wimg.reshape(1, 6, self.ph, self.pw)
                wf = wf.reshape(1, 8, self.ph, self.pw)

        mask = torch.sigmoid(mask)
        warpedImg0, warpedImg1 = torch.split(warpedImgs, [1, 1])
        return (warpedImg0 * mask + warpedImg1 * (1 - mask))[
            :, :, : self.height, : self.width
        ]


# =============================================================================
# RIFE 4.25-lite DirectML (4ch features, 5 blocks, mul=128)
# =============================================================================
class IFNet_425_lite(nn.Module):
    """RIFE 4.25-lite DirectML - 5 blocks, 4ch features, mul=128."""

    def __init__(
        self,
        scale=1.0,
        ensemble=False,
        dtype=torch.float32,
        device="cuda",
        width=1920,
        height=1080,
    ):
        super(IFNet_425_lite, self).__init__()
        self.block0 = IFBlock422(7 + 8, c=192)
        self.block1 = IFBlock422(8 + 4 + 8 + 8, c=128)
        self.block2 = IFBlock422(8 + 4 + 8 + 8, c=96)
        self.block3 = IFBlock422(8 + 4 + 8 + 8, c=64)
        self.block4 = IFBlock422(8 + 4 + 8 + 8, c=24)
        self.encode = Head4()
        self.device = device
        self.dtype = dtype
        self.scaleList = [16 / scale, 8 / scale, 4 / scale, 2 / scale, 1 / scale]
        self.width = width
        self.height = height
        self.blocks = [self.block0, self.block1, self.block2, self.block3, self.block4]

        tmp = max(128, int(128 / 1.0))
        self.pw = math.ceil(self.width / tmp) * tmp
        self.ph = math.ceil(self.height / tmp) * tmp
        self.padding = (0, self.pw - self.width, 0, self.ph - self.height)

        hMul = 2 / (self.pw - 1)
        vMul = 2 / (self.ph - 1)
        self.register_buffer(
            "tenFlowMul", torch.tensor([hMul, vMul], dtype=dtype).reshape(1, 2, 1, 1)
        )
        horizontal = (
            (torch.arange(self.pw, dtype=dtype) * hMul - 1)
            .reshape(1, 1, 1, -1)
            .expand(-1, -1, self.ph, -1)
        )
        vertical = (
            (torch.arange(self.ph, dtype=dtype) * vMul - 1)
            .reshape(1, 1, -1, 1)
            .expand(-1, -1, -1, self.pw)
        )
        self.register_buffer("backWarp", torch.cat([horizontal, vertical], dim=1))

    def forward(self, img0, img1, timestep):
        imgs = torch.cat([img0, img1], dim=1)
        imgs2 = imgs.view(2, 3, self.ph, self.pw)
        f0 = self.encode(img0[:, :3])
        f1 = self.encode(img1[:, :3])
        fs = torch.cat([f0, f1], dim=1)
        fs2 = fs.view(2, 4, self.ph, self.pw)

        wimg = None
        wf = None

        flows = None
        for block, scale in zip(self.blocks, self.scaleList):
            if flows is None:
                temp = torch.cat((imgs, fs, timestep), 1)
                flows, mask, feat = block(temp, scale=scale)
            else:
                temp = torch.cat(
                    (
                        wimg,
                        wf,
                        timestep,
                        mask,
                        feat,
                        flows * (1 / scale) if scale != 1 else flows,
                    ),
                    1,
                )  # noqa
                fds, mask, feat = block(temp, scale=scale)
                flows = flows + fds

            precomp = (
                self.backWarp
                + flows.reshape((2, 2, self.ph, self.pw)) * self.tenFlowMul
            ).permute(0, 2, 3, 1)
            if scale == 1:
                warpedImgs = grid_sample_directml(
                    imgs2, precomp, padding_mode="border", align_corners=True
                )
            else:
                warps = grid_sample_directml(
                    torch.cat((imgs2, fs2), 1),
                    precomp,
                    padding_mode="border",
                    align_corners=True,
                )
                wimg, wf = torch.split(warps, [3, 4], dim=1)
                wimg = wimg.reshape(1, 6, self.ph, self.pw)
                wf = wf.reshape(1, 8, self.ph, self.pw)

        mask = torch.sigmoid(mask)
        warpedImg0, warpedImg1 = torch.split(warpedImgs, [1, 1])
        return (warpedImg0 * mask + warpedImg1 * (1 - mask))[
            :, :, : self.height, : self.width
        ]


# =============================================================================
# RIFE 4.25-heavy DirectML (4ch features, 5 blocks, 2x channels, mul=64)
# =============================================================================
class IFNet_425_heavy(nn.Module):
    """RIFE 4.25-heavy DirectML - 5 blocks, 4ch features, doubled channel counts."""

    def __init__(
        self,
        scale=1.0,
        ensemble=False,
        dtype=torch.float32,
        device="cuda",
        width=1920,
        height=1080,
    ):
        super(IFNet_425_heavy, self).__init__()
        self.block0 = IFBlock422(7 + 8, c=192 * 2)
        self.block1 = IFBlock422(8 + 4 + 8 + 8, c=128 * 2)
        self.block2 = IFBlock422(8 + 4 + 8 + 8, c=96 * 2)
        self.block3 = IFBlock422(8 + 4 + 8 + 8, c=64 * 2)
        self.block4 = IFBlock422(8 + 4 + 8 + 8, c=32 * 2)
        self.encode = Head4()
        self.device = device
        self.dtype = dtype
        self.scaleList = [16 / scale, 8 / scale, 4 / scale, 2 / scale, 1 / scale]
        self.width = width
        self.height = height
        self.blocks = [self.block0, self.block1, self.block2, self.block3, self.block4]

        tmp = max(64, int(64 / 1.0))
        self.pw = math.ceil(self.width / tmp) * tmp
        self.ph = math.ceil(self.height / tmp) * tmp
        self.padding = (0, self.pw - self.width, 0, self.ph - self.height)

        hMul = 2 / (self.pw - 1)
        vMul = 2 / (self.ph - 1)
        self.register_buffer(
            "tenFlowMul", torch.tensor([hMul, vMul], dtype=dtype).reshape(1, 2, 1, 1)
        )
        horizontal = (
            (torch.arange(self.pw, dtype=dtype) * hMul - 1)
            .reshape(1, 1, 1, -1)
            .expand(-1, -1, self.ph, -1)
        )
        vertical = (
            (torch.arange(self.ph, dtype=dtype) * vMul - 1)
            .reshape(1, 1, -1, 1)
            .expand(-1, -1, -1, self.pw)
        )
        self.register_buffer("backWarp", torch.cat([horizontal, vertical], dim=1))

    def forward(self, img0, img1, timestep):
        imgs = torch.cat([img0, img1], dim=1)
        imgs2 = imgs.view(2, 3, self.ph, self.pw)
        f0 = self.encode(img0[:, :3])
        f1 = self.encode(img1[:, :3])
        fs = torch.cat([f0, f1], dim=1)
        fs2 = fs.view(2, 4, self.ph, self.pw)

        wimg = None
        wf = None

        flows = None
        for block, scale in zip(self.blocks, self.scaleList):
            if flows is None:
                temp = torch.cat((imgs, fs, timestep), 1)
                flows, mask, feat = block(temp, scale=scale)
            else:
                temp = torch.cat(
                    (
                        wimg,
                        wf,
                        timestep,
                        mask,
                        feat,
                        flows * (1 / scale) if scale != 1 else flows,
                    ),
                    1,
                )  # noqa
                fds, mask, feat = block(temp, scale=scale)
                flows = flows + fds

            precomp = (
                self.backWarp
                + flows.reshape((2, 2, self.ph, self.pw)) * self.tenFlowMul
            ).permute(0, 2, 3, 1)
            if scale == 1:
                warpedImgs = grid_sample_directml(
                    imgs2, precomp, padding_mode="border", align_corners=True
                )
            else:
                warps = grid_sample_directml(
                    torch.cat((imgs2, fs2), 1),
                    precomp,
                    padding_mode="border",
                    align_corners=True,
                )
                wimg, wf = torch.split(warps, [3, 4], dim=1)
                wimg = wimg.reshape(1, 6, self.ph, self.pw)
                wf = wf.reshape(1, 8, self.ph, self.pw)

        mask = torch.sigmoid(mask)
        warpedImg0, warpedImg1 = torch.split(warpedImgs, [1, 1])
        return (warpedImg0 * mask + warpedImg1 * (1 - mask))[
            :, :, : self.height, : self.width
        ]
