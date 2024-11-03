import torch
import torch.nn as nn
import math

from torch.nn.functional import interpolate
from .warplayer_v2 import warp


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode="replicate",
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

    def forward(self, x, h, w, flow=None, scale=1):
        x = interpolate(x, scale_factor=1.0 / scale, mode="bilinear")

        if flow is not None:
            flow = interpolate(flow, scale_factor=1.0 / scale, mode="bilinear") / scale
            x = torch.cat((x, flow), 1)

        feat = self.conv0(x)
        feat = self.convblock(feat)
        tmp = self.lastconv(feat)
        tmp = interpolate(tmp, size=(h, w), mode="bilinear")

        flow = tmp[:, :4] * scale
        mask = tmp[:, 4:5]

        return flow, mask


class IFNet(nn.Module):
    def __init__(
        self,
        scale=1,
        ensemble=False,
        dtype=torch.float32,
        device="cuda",
        width=1920,
        height=1080,
    ):
        super(IFNet, self).__init__()
        self.block0 = IFBlock(7 + 8, c=192)
        self.block1 = IFBlock(8 + 4 + 8, c=128)
        self.block2 = IFBlock(8 + 4 + 8, c=96)
        self.block3 = IFBlock(8 + 4 + 8, c=64)
        self.encode = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1), nn.ConvTranspose2d(16, 4, 4, 2, 1)
        )

        self.width = width
        self.height = height
        self.device = device
        self.dtype = dtype

        self.scale_list = [
            32 / scale,
            16 / scale,
            8 / scale,
            4 / scale,
            2 / scale,
            1 / scale,
        ]
        self.ensemble = ensemble
        self.blocks = [
            self.block0,
            self.block1,
            self.block0,
            self.block1,
            self.block2,
            self.block3,
        ]
        tmp = max(64, int(64 / 1.0))
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

    def forward(self, img0, img1, timestep, f0):
        f1 = self.encode(img1)
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        large_flow = None
        mask = None

        for i in range(6):
            if flow is None:
                flow, mask = self.blocks[i](
                    torch.cat((img0, img1, f0, f1, timestep), 1),
                    self.ph,
                    self.pw,
                    None,
                    scale=self.scale_list[i],
                )

                if large_flow is not None:
                    magnitude = torch.sqrt(
                        large_flow[:, 0, :, :] ** 2 + large_flow[:, 1, :, :] ** 2
                    )

                    count = torch.sum(magnitude > 40)
                    mask_large = count > 1036800

                    flow = torch.where(mask_large, large_flow, flow)

            else:
                wf0 = warp(f0, flow[:, :2], self.tenFlow, self.backWarp)
                wf1 = warp(f1, flow[:, 2:4], self.tenFlow, self.backWarp)
                fd, mask = self.blocks[i](
                    torch.cat((warped_img0, warped_img1, wf0, wf1, timestep, mask), 1),
                    self.ph,
                    self.pw,
                    flow,
                    scale=self.scale_list[i],
                )
                flow = flow + fd

            warped_img0 = warp(img0, flow[:, :2], self.tenFlow, self.backWarp)
            warped_img1 = warp(img1, flow[:, 2:4], self.tenFlow, self.backWarp)

            if i == 1:
                large_flow = flow
                flow = None

        mask = torch.sigmoid(mask)
        return (warped_img0 * mask + warped_img1 * (1 - mask))[
            :, :, : self.height, : self.width
        ], f1
