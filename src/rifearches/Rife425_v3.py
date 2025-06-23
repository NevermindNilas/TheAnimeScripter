import torch
import torch.nn as nn
from torch.nn.functional import interpolate
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


class Head(nn.Module):
    def __init__(self):
        super(Head, self).__init__()
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


class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, out_planes=c // 2, kernel_size=3, stride=2, padding=1),
            conv(c // 2, out_planes=c, kernel_size=3, stride=2, padding=1),
        )
        self.convblock = nn.Sequential(*[ResConv(c) for _ in range(8)])
        self.lastconv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=c, out_channels=4 * 13, kernel_size=4, stride=2, padding=1
            ),
            nn.PixelShuffle(upscale_factor=2),
        )
        self.in_planes = in_planes

    def forward(self, x, scale=1):
        if scale != 1:
            x = interpolate(
                x, scale_factor=1 / scale, mode="bilinear", align_corners=False
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


class IFNet(nn.Module):
    def __init__(
        self,
        scale=1.0,
        ensemble=False,
        dtype=torch.float32,
        device="cuda",
        width=1920,
        height=1080,
    ):
        super(IFNet, self).__init__()
        self.block0 = IFBlock(7 + 8, c=192)
        self.block1 = IFBlock(8 + 4 + 8 + 8, c=128)
        self.block2 = IFBlock(8 + 4 + 8 + 8, c=96)
        self.block3 = IFBlock(8 + 4 + 8 + 8, c=64)
        self.block4 = IFBlock(8 + 4 + 8 + 8, c=32)

        self.encode = Head()
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
        self.tenFlow = (
            torch.Tensor([hMul, vMul])
            .to(device=self.device, dtype=self.dtype)
            .reshape(1, 2, 1, 1)
        )
        self.backWarp = torch.cat(
            (
                (torch.arange(self.pw) * hMul - 1)
                .reshape(1, 1, 1, -1)
                .expand(-1, -1, self.ph, -1),
                (torch.arange(self.ph) * vMul - 1)
                .reshape(1, 1, -1, 1)
                .expand(-1, -1, -1, self.pw),
            ),
            dim=1,
        ).to(device=self.device, dtype=self.dtype)

    def forward(self, img0, img1, timeStep, f0):
        warpedImg0, warpedImg1 = img0, img1
        imgs = torch.cat([img0, img1], dim=1)
        imgs2 = imgs.view(2, 3, self.ph, self.pw)
        f1 = self.encode(img1[:, :3])
        fs = torch.cat([f0, f1], dim=1)
        fs2 = fs.view(2, 4, self.ph, self.pw)

        flows = None
        for block, scale in zip(self.blocks, self.scaleList):
            if flows is None:
                temp = torch.cat((imgs, fs, timeStep), 1)
                flows, mask, feat = block(temp, scale=scale)
            else:
                temp = torch.cat(
                    (
                        wimg,  # noqa
                        wf,  # noqa
                        timeStep,
                        mask,
                        feat,
                        (flows * (1 / scale) if scale != 1 else flows),
                    ),
                    1,
                )
                fds, mask, feat = block(temp, scale=scale)
                flows = flows + fds

            if scale == 1:
                warpedImgs = torch.nn.functional.grid_sample(
                    imgs2,
                    (
                        self.backWarp
                        + flows.reshape((2, 2, self.ph, self.pw)) * self.tenFlow
                    ).permute(0, 2, 3, 1),
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=True,
                )
            else:
                warps = torch.nn.functional.grid_sample(
                    torch.cat((imgs2, fs2), 1),
                    (
                        self.backWarp
                        + flows.reshape((2, 2, self.ph, self.pw)) * self.tenFlow
                    ).permute(0, 2, 3, 1),
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=True,
                )
                wimg, wf = torch.split(warps, [3, 4], dim=1)
                wimg = torch.reshape(wimg, (1, 6, self.ph, self.pw))
                wf = torch.reshape(wf, (1, 8, self.ph, self.pw))

        mask = torch.sigmoid(mask)
        warpedImg0, warpedImg1 = torch.split(warpedImgs, [1, 1])

        return (warpedImg0 * mask + warpedImg1 * (1 - mask))[
            :, :, : self.height, : self.width
        ], f1
