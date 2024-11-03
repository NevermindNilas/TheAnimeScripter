import torch
import torch.nn as nn
import torch.nn.functional as F
from .warplayer_v2 import warp
import math


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


class ResConv(nn.Module):
    def __init__(self, c, dilation=1):
        super(ResConv, self).__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.relu(self.conv(x) * self.beta + x)


class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1),
            conv(c // 2, c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
        )
        self.lastconv = nn.Sequential(
            nn.ConvTranspose2d(c, 4 * 6, 4, 2, 1), nn.PixelShuffle(2)
        )

    def forward(self, x, flow=None, scale=1):
        x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear")
        if flow is not None:
            flow = (
                F.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear")
                * 1.0
                / scale
            )
            x = torch.cat((x, flow), 1)
        feat = self.conv0(x)
        feat = self.convblock(feat)
        tmp = self.lastconv(feat)
        tmp = F.interpolate(tmp, scale_factor=scale, mode="bilinear")
        flow = tmp[:, :4] * scale
        mask = tmp[:, 4:5]
        return flow, mask


class IFNet(nn.Module):
    def __init__(
        self,
        scale=1.0,
        ensemble=False,
        dtype=torch.float32,
        device="cuda",
        width=1920,
        height=1080,
        backWarp=None,
        tenFlow=None,
    ):
        super(IFNet, self).__init__()
        self.block0 = IFBlock(7, c=192)
        self.block1 = IFBlock(8 + 4, c=128)
        self.block2 = IFBlock(8 + 4, c=96)
        self.block3 = IFBlock(8 + 4, c=64)
        self.scaleList = [8 / scale, 4 / scale, 2 / scale, 1 / scale]
        self.ensemble = ensemble
        self.dtype = dtype
        self.device = device
        self.width = width
        self.height = height
        self.blocks = [self.block0, self.block1, self.block2, self.block3]

        self.dtype = torch.float16 if self.half else torch.float32
        tmp = max(32, int(32 / 1.0))
        self.pw = math.ceil(self.width / tmp) * tmp
        self.ph = math.ceil(self.height / tmp) * tmp
        self.padding = (0, self.pw - self.width, 0, self.ph - self.height)
        self.tenFlow = torch.tensor(
            [(self.pw - 1.0) / 2.0, (self.ph - 1.0) / 2.0],
            dtype=self.dtype,
            device=self.device,
        )
        tenHorizontal = (
            torch.linspace(-1.0, 1.0, self.pw, dtype=self.dtype, device=self.device)
            .view(1, 1, 1, self.pw)
            .expand(-1, -1, self.ph, -1)
        ).to(dtype=self.dtype, device=self.device)
        tenVertical = (
            torch.linspace(-1.0, 1.0, self.ph, dtype=self.dtype, device=self.device)
            .view(1, 1, self.ph, 1)
            .expand(-1, -1, -1, self.pw)
        ).to(dtype=self.dtype, device=self.device)
        self.backWarp = torch.cat([tenHorizontal, tenVertical], 1)

    def forward(self, img0, img1, timeStep):
        warpedImg0, warpedImg1 = img0, img1
        flow = mask = None

        for i, block in enumerate(self.blocks):
            scale = self.scaleList[i]

            if flow is None:
                flow, mask = block(
                    torch.cat((img0[:, :3], img1[:, :3], timeStep), 1),
                    None,
                    scale=scale,
                )

                if self.ensemble:
                    f1, m1 = block(
                        torch.cat((img1[:, :3], img0[:, :3], 1 - timeStep), 1),
                        None,
                        scale=scale,
                    )
                    flow = (flow + torch.cat((f1[:, 2:4], f1[:, :2]), 1)) / 2
                    mask = (mask - m1) / 2
            else:
                f0, m0 = block(
                    torch.cat(
                        (warpedImg0[:, :3], warpedImg1[:, :3], timeStep, mask), 1
                    ),
                    flow,
                    scale=scale,
                )

                if self.ensemble:
                    f1, m1 = block(
                        torch.cat(
                            (warpedImg1[:, :3], warpedImg0[:, :3], 1 - timeStep, -mask),
                            1,
                        ),
                        torch.cat((flow[:, 2:4], flow[:, :2]), 1),
                        scale=scale,
                    )
                    f0 = (f0 + torch.cat((f1[:, 2:4], f1[:, :2]), 1)) / 2
                    m0 = (m0 - m1) / 2

                flow = flow + f0
                mask = mask + m0

            warpedImg0 = warp(img0, flow[:, :2], self.tenFlow, self.backWarp)
            warpedImg1 = warp(img1, flow[:, 2:4], self.tenFlow, self.backWarp)

        temp = torch.sigmoid(mask)
        return (warpedImg0 * temp + warpedImg1 * (1 - temp))[
            :, :, : self.height, : self.width
        ]
