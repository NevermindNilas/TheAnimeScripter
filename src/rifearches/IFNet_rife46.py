# IFNet_rife46.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .warplayer import warp
from .dynamic_scale import dynamicScale


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
        nn.LeakyReLU(0.2, inplace=True),
    )


def conv_bn(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        ),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True),
    )


class ResConv(nn.Module):
    def __init__(self, c, dilation=1):
        super(ResConv, self).__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

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
        x = F.interpolate(
            x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False
        )
        if flow is not None:
            flow = F.interpolate(
                flow, scale_factor=1.0 / scale, mode="bilinear", align_corners=False
            ) * (1.0 / scale)
            x = torch.cat((x, flow), 1)
        feat = self.conv0(x)
        feat = self.convblock(feat)
        tmp = self.lastconv(feat)
        tmp = F.interpolate(
            tmp, scale_factor=scale, mode="bilinear", align_corners=False
        )
        flow = tmp[:, :4] * scale
        mask = tmp[:, 4:5]
        return flow, mask


class IFNet(nn.Module):
    def __init__(
        self,
        ensemble=False,
        dynamicScale=False,
        scale=1,
        interpolateFactor=2,
        staticStep=False,
    ):
        super(IFNet, self).__init__()
        self.block0 = IFBlock(7, c=192)
        self.block1 = IFBlock(8 + 4, c=128)
        self.block2 = IFBlock(8 + 4, c=96)
        self.block3 = IFBlock(8 + 4, c=64)

        self.scale_list = [8 / scale, 4 / scale, 2 / scale, 1 / scale]
        self.ensemble = ensemble
        self.dynamicScale = dynamicScale
        self.interpolateFactor = interpolateFactor
        self.blocks = [self.block0, self.block1, self.block2, self.block3]
        self.staticStep = staticStep
        if self.staticStep:
            self.timesteps = None

    def cache(self):
        pass

    def cacheReset(self, frame):
        pass

    def bakedTimestep(self, img0, img1):
        warped_img0 = img0
        warped_img1 = img1
        outputs = []

        if self.timesteps is None:
            timesteps = []
            for i in range(1, self.interpolateFactor):
                t = i / self.interpolateFactor
                ts = torch.full(
                    (1, 1, img0.size(2), img0.size(3)),
                    t,
                    dtype=img0.dtype,
                    device=img0.device,
                )
                timesteps.append(ts)
            timesteps = torch.stack(timesteps, dim=0)
        else:
            timesteps = self.timesteps

        if self.dynamicScale:
            scale = dynamicScale(img0, img1)
            self.scale_list = [8 / scale, 4 / scale, 2 / scale, 1 / scale]

        for t in timesteps:
            warped_img0 = img0
            warped_img1 = img1

            flow, mask = self.blocks[0](
                torch.cat((img0[:, :3], img1[:, :3], t), 1),
                None,
                scale=self.scale_list[0],
            )
            if self.ensemble:
                f1, m1 = self.blocks[0](
                    torch.cat((img1[:, :3], img0[:, :3], 1 - t), 1),
                    None,
                    scale=self.scale_list[0],
                )
                flow = (flow + torch.cat((f1[:, 2:4], f1[:, :2]), 1)) / 2
                mask = (mask + (-m1)) / 2

            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])

            for j, block in enumerate(self.blocks[1:], start=1):
                f, m = block(
                    torch.cat((warped_img0[:, :3], warped_img1[:, :3], t, mask), 1),
                    flow,
                    scale=self.scale_list[j],
                )
                if self.ensemble:
                    f1, m1 = block(
                        torch.cat(
                            (warped_img1[:, :3], warped_img0[:, :3], 1 - t, -mask), 1
                        ),
                        torch.cat((flow[:, 2:4], flow[:, :2]), 1),
                        scale=self.scale_list[j],
                    )
                    f = (f + torch.cat((f1[:, 2:4], f1[:, :2]), 1)) / 2
                    m = (m + (-m1)) / 2
                flow = flow + f
                mask = mask + m

                warped_img0 = warp(img0, flow[:, :2])
                warped_img1 = warp(img1, flow[:, 2:4])
            mask = torch.sigmoid(mask)
            outputs.append(warped_img0 * mask + warped_img1 * (1 - mask))

        return outputs

    def forward(self, img0, img1, timestep):
        if self.staticStep:
            return self.bakedTimestep(img0, img1)

        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1

        if self.dynamicScale:
            scale = dynamicScale(img0, img1)
            self.scale_list = [8 / scale, 4 / scale, 2 / scale, 1 / scale]

        flow = None
        for i in range(4):
            if flow is None:
                flow, mask = self.blocks[i](
                    torch.cat((img0[:, :3], img1[:, :3], timestep), 1),
                    None,
                    scale=self.scale_list[i],
                )
                if self.ensemble:
                    f1, m1 = self.blocks[i](
                        torch.cat((img1[:, :3], img0[:, :3], 1 - timestep), 1),
                        None,
                        scale=self.scale_list[i],
                    )
                    flow = (flow + torch.cat((f1[:, 2:4], f1[:, :2]), 1)) / 2
                    mask = (mask + (-m1)) / 2
            else:
                f0, m0 = self.blocks[i](
                    torch.cat(
                        (warped_img0[:, :3], warped_img1[:, :3], timestep, mask), 1
                    ),
                    flow,
                    scale=self.scale_list[i],
                )
                if self.ensemble:
                    f1, m1 = self.blocks[i](
                        torch.cat(
                            (
                                warped_img1[:, :3],
                                warped_img0[:, :3],
                                1 - timestep,
                                -mask,
                            ),
                            1,
                        ),
                        torch.cat((flow[:, 2:4], flow[:, :2]), 1),
                        scale=self.scale_list[i],
                    )
                    f0 = (f0 + torch.cat((f1[:, 2:4], f1[:, :2]), 1)) / 2
                    m0 = (m0 + (-m1)) / 2
                flow = flow + f0
                mask = mask + m0
            mask_list.append(mask)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged.append((warped_img0, warped_img1))
        mask_list[3] = torch.sigmoid(mask_list[3])
        return merged[3][0] * mask_list[3] + merged[3][1] * (1 - mask_list[3])
