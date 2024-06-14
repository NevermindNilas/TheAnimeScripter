"""PerVFI: Fully-Adaptive mask

"""
import accelerate
import torch
import torch.nn.functional as F
import logging as logger
from torch import Tensor

from .msfusion import MultiscaleFuse
from .normalizing_flow import *
from .softsplatnet import Basic, Encode, Softmetric
from .softsplatnet.softsplat import softsplat


def resize(x, size: tuple, scale: bool):
    H, W = x.shape[-2:]
    h, w = size
    scale_ = h / H
    x_ = F.interpolate(x, size, mode="bilinear", align_corners=False)
    if scale:
        return x_ * scale_
    return x_


def binary_hole(flow):
    n, _, h, w = flow.shape
    mask = softsplat(
        tenIn=torch.ones((n, 1, h, w), device=flow.device),
        tenFlow=flow,
        tenMetric=None,
        strMode="avg",
    )
    ones = torch.ones_like(mask, device=mask.device)
    zeros = torch.zeros_like(mask, device=mask.device)
    out = torch.where(mask <= 0.5, ones, zeros)
    return out


def warp_pyramid(features: list, metric, flow):
    outputs = []
    masks = []
    for lv in range(3):
        fea = features[lv]
        if lv != 0:
            h, w = fea.shape[-2:]
            metric = resize(metric, (h, w), scale=False)
            flow = resize(flow, (h, w), scale=True)
        outputs.append(softsplat(fea, flow, metric.neg().clip(-20.0, 20.0), "soft"))
        masks.append(binary_hole(flow))
    return outputs, masks


class FeaturePyramid(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.netEncode = Encode()
        self.netSoftmetric = Softmetric()

    def forward(
        self,
        tenOne,
        tenTwo=None,
        tenFlows: list[Tensor] = None,
        time: float = 0.5,
    ):
        x1s = self.netEncode(tenOne)
        x2s = self.netEncode(tenTwo)
        if tenTwo is None:  # just encode
            return x1s
        F12, F21 = tenFlows
        m1t = self.netSoftmetric(x1s, x2s, F12) * 2 * time
        F1t = time * F12
        m2t = self.netSoftmetric(x2s, x1s, F21) * 2 * (1 - time)
        F2t = (1 - time) * F21
        x1s, bmasks1 = warp_pyramid(x1s, m1t, F1t)
        x2s, bmasks2 = warp_pyramid(x2s, m2t, F2t)
        return list(zip(x1s, x2s)), bmasks1, bmasks2


class SoftBinary(torch.nn.Module):
    def __init__(self, cin, dilate_size=7) -> None:
        super().__init__()
        channel = 64
        reduction = 8
        self.conv1 = torch.nn.Sequential(
            *[
                torch.nn.Conv2d(2, channel, dilate_size, 1, padding="same", bias=False),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(channel, channel, 3, 1, padding="same", bias=False),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(channel, channel, 1, 1, padding="same", bias=False),
            ]
        )
        self.att = torch.nn.Conv2d(cin * 2, channel, 3, 1, padding="same")
        self.avg = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(channel, channel // reduction, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(channel // reduction, channel, bias=False),
            torch.nn.Sigmoid(),
        )
        self.conv2 = torch.nn.Conv2d(channel, 1, 1, 1, padding="same", bias=False)

    def forward(self, bmask: list, feaL, feaR):
        m_fea = self.conv1(torch.cat(bmask, dim=1))
        x = self.att(torch.cat([feaL, feaR], dim=1))
        b, c, _, _ = x.size()
        x = self.avg(x).view(b, c)
        x = self.fc(x).view(b, c, 1, 1)
        x = m_fea * x.expand_as(x)
        x = self.conv2(x)
        return torch.sigmoid(x)


class AttentionMerge(torch.nn.Module):
    def __init__(self, dilate_size=7, **kwargs):
        super().__init__()
        self.softbinary = torch.nn.ModuleDict()
        channels = [35, 64, 96]
        for i in range(2, -1, -1):
            level = f"{i}"
            c = channels[i]
            self.softbinary[level] = SoftBinary(c, dilate_size, **kwargs)

    def forward(self, feaL, feaR, bmask1, bmask2):
        outs = []
        soft_masks = []
        for i in range(2, -1, -1):
            level = f"{i}"
            sm = self.softbinary[level]([bmask1[i], bmask2[i]], feaL[i], feaR[i])
            soft_masks.append(sm)
            x = feaL[i] * (1 - sm) + feaR[i] * sm
            outs.append(x)
        return outs, soft_masks


class Decoder(torch.nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.conv3 = Basic("conv-relu-conv", [cin[2], 128, cin[1]], True)
        self.conv2 = Basic("conv-relu-conv", [cin[1] * 2, 256, cin[0]], True)
        self.conv1 = Basic("conv-relu-conv", [cin[0] * 2, 96, 128], True)
        self.tail = torch.nn.Conv2d(128 // 4, 3, 3, 1, padding="same")
        self.up = torch.nn.UpsamplingBilinear2d(scale_factor=2)
        logger.info(
            f"Parameter of decoder: {sum(p.numel() for p in self.parameters())}"
        )

    def forward(self, xs):
        lv1, lv2, lv3 = xs
        lv3 = self.conv3(lv3)
        lv3 = self.up(lv3)
        lv2 = self.conv2(torch.cat([lv3, lv2], dim=1))
        lv2 = self.up(lv2)
        lv1 = self.conv1(torch.cat([lv2, lv1], dim=1))
        lv1 = unsqueeze2d(lv1, factor=2)
        return self.tail(lv1)


class Network(torch.torch.nn.Module):
    def __init__(self, dilate_size=9, **kwargs):
        super().__init__()
        cond_c = [35, 64, 96]
        self.featurePyramid = FeaturePyramid()
        self.attentionMerge = AttentionMerge(dilate_size=dilate_size, **kwargs)
        self.multiscaleFuse = MultiscaleFuse(cond_c)
        self.generator = Decoder(cond_c)

    def get_cond(self, inps: list, time: float = 0.5):
        tenOne, tenTwo, fflow, bflow = inps
        with accelerate.Accelerator().autocast():
            feas, bmasks1, bmasks2 = self.featurePyramid(
                tenOne, tenTwo, [fflow, bflow], time
            )
            feaL = [feas[i][0] for i in range(3)]
            feaR = [feas[i][1] for i in range(3)]
            feas, smasks = self.attentionMerge(feaL, feaR, bmasks1, bmasks2)
            feas = self.multiscaleFuse(feas[::-1])  # downscale by 2
        return feas, smasks

    def normalize(self, x, reverse=False):
        # x in [0, 1]
        if not reverse:
            return x * 2 - 1
        else:
            return (x + 1) / 2

    def forward(self, inps=[], time=0.5, **kwargs):
        img0, img1 = [self.normalize(x) for x in inps[:2]]
        cond = [img0, img1] + inps[-2:]

        conds, smasks = self.get_cond(cond, time=time)
        with accelerate.Accelerator().autocast():
            pred = self.generator(conds)
        pred = self.normalize(pred, reverse=True)
        return pred, smasks
