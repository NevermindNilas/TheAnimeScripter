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
        nn.LeakyReLU(0.2, True),
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
        x = F.interpolate(
            x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False
        )
        if flow is not None:
            flow = (
                F.interpolate(
                    flow, scale_factor=1.0 / scale, mode="bilinear", align_corners=False
                )
                * 1.0
                / scale
            )
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
    def __init__(self, ensemble=False, dynamicScale=False, scale=1):
        super(IFNet, self).__init__()
        self.block0 = IFBlock(7 + 8, c=128)
        self.block1 = IFBlock(8 + 4 + 8, c=96)
        self.block2 = IFBlock(8 + 4 + 8, c=64)
        self.block3 = IFBlock(8 + 4 + 8, c=48)
        self.encode = Head()
        self.f0 = None
        self.f1 = None
        self.scale_list = [8 / scale, 4 / scale, 2 / scale, 1 / scale]
        self.ensemble = ensemble
        self.dynamicScale = dynamicScale
        self.counter = 1

    def cache(self):
        self.f0.copy_(self.f1, non_blocking=True)

    def cacheReset(self, frame):
        self.f0 = self.encode(frame[:, :3])

    def forward(self, img0, img1, timestep, interpolateFactor=2):
        # Overengineered but it seems to work
        if interpolateFactor == 2:
            if self.f0 is None:
                self.f0 = self.encode(img0[:, :3])
            self.f1 = self.encode(img1[:, :3])
        else:
            if self.counter == interpolateFactor:
                self.counter = 1
                if self.f0 is None:
                    self.f0 = self.encode(img0[:, :3])
                self.f1 = self.encode(img1[:, :3])
            else:
                if self.f0 is None or self.f1 is None:
                    self.f0 = self.encode(img0[:, :3])
                    self.f1 = self.encode(img1[:, :3])
            self.counter += 1

        merged = []
        warped_img0 = img0
        warped_img1 = img1

        if self.dynamicScale:
            scale = dynamicScale(img0, img1)
            self.scale_list = [8 / scale, 4 / scale, 2 / scale, 1 / scale]
        flow = None
        block = [self.block0, self.block1, self.block2, self.block3]
        for i in range(4):
            if flow is None:
                flow, mask = block[i](
                    torch.cat(
                        (img0[:, :3], img1[:, :3], self.f0, self.f1, timestep), 1
                    ),
                    None,
                    scale=self.scale_list[i],
                )
                if self.ensemble:
                    f_, m_ = block[i](
                        torch.cat(
                            (img1[:, :3], img0[:, :3], self.f1, self.f0, 1 - timestep),
                            1,
                        ),
                        None,
                        scale=self.scale_list[i],
                    )
                    flow = (flow + torch.cat((f_[:, 2:4], f_[:, :2]), 1)) / 2
                    mask = (mask + (-m_)) / 2
            else:
                wf0 = warp(self.f0, flow[:, :2])
                wf1 = warp(self.f1, flow[:, 2:4])
                fd, m0 = block[i](
                    torch.cat(
                        (
                            warped_img0[:, :3],
                            warped_img1[:, :3],
                            wf0,
                            wf1,
                            timestep,
                            mask,
                        ),
                        1,
                    ),
                    flow,
                    scale=self.scale_list[i],
                )
                if self.ensemble:
                    f_, m_ = block[i](
                        torch.cat(
                            (
                                warped_img1[:, :3],
                                warped_img0[:, :3],
                                wf1,
                                wf0,
                                1 - timestep,
                                -mask,
                            ),
                            1,
                        ),
                        torch.cat((flow[:, 2:4], flow[:, :2]), 1),
                        scale=self.scale_list[i],
                    )
                    fd = (fd + torch.cat((f_[:, 2:4], f_[:, :2]), 1)) / 2
                    mask = (m0 + (-m_)) / 2
                else:
                    mask = m0
                flow = flow + fd
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged.append((warped_img0, warped_img1))
        mask = torch.sigmoid(mask)
        return warped_img0 * mask + warped_img1 * (1 - mask)
