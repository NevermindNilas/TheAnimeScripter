"""
DistilDRBA TensorRT Export Architecture
Simplified version for ONNX/TensorRT export - handles only backward interpolation (t < 1.0)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


backwarpTenGrid = {}


def warp(tenInput, tenFlow):
    """
    Backward warping using grid_sample.
    """
    k = (str(tenFlow.device), str(tenFlow.size()))
    if k not in backwarpTenGrid:
        tenHorizontal = (
            torch.linspace(-1.0, 1.0, tenFlow.shape[3])
            .view(1, 1, 1, tenFlow.shape[3])
            .expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        )
        tenVertical = (
            torch.linspace(-1.0, 1.0, tenFlow.shape[2])
            .view(1, 1, tenFlow.shape[2], 1)
            .expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
        )
        backwarpTenGrid[k] = torch.cat([tenHorizontal, tenVertical], 1).to(
            tenFlow.device
        )

    tenFlow = torch.cat(
        [
            tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
            tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0),
        ],
        1,
    )

    g = (backwarpTenGrid[k] + tenFlow).permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(
        input=tenInput,
        grid=g.to(tenInput.dtype),
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


def conv(inPlanes, outPlanes, kernelSize=3, stride=1, padding=1, dilation=1):
    """Conv2d + LeakyReLU"""
    return nn.Sequential(
        nn.Conv2d(
            inPlanes,
            outPlanes,
            kernel_size=kernelSize,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        ),
        nn.LeakyReLU(0.2, True),
    )


class Head(nn.Module):
    """Feature encoder head - encodes RGB to 16-channel features"""

    def __init__(self):
        super(Head, self).__init__()
        self.cnn0 = nn.Conv2d(3, 16, 3, 2, 1)
        self.cnn1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(16, 16, 4, 2, 1)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x0 = self.cnn0(x)
        x = self.relu(x0)
        x1 = self.cnn1(x)
        x = self.relu(x1)
        x2 = self.cnn2(x)
        x = self.relu(x2)
        x3 = self.cnn3(x)
        return x3


class ResConv(nn.Module):
    """Residual convolution block"""

    def __init__(self, c, dilation=1):
        super(ResConv, self).__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.relu(self.conv(x) * self.beta + x)


class IFBlockV2Lite(nn.Module):
    """Flow estimation block for v2_lite model (3 blocks)"""

    def __init__(self, inPlanes, c=64):
        super(IFBlockV2Lite, self).__init__()
        self.conv0 = nn.Sequential(
            conv(inPlanes, c // 2, 3, 2, 1),
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
            nn.ConvTranspose2d(c, 4 * (4 + 1 + 8), 4, 2, 1), nn.PixelShuffle(2)
        )

    def forward(self, x, flow=None):
        if flow is not None:
            x = torch.cat((x, flow), 1)
        feat = self.conv0(x)
        feat = self.convblock(feat)
        tmp = self.lastconv(feat)
        flow = tmp[:, :4]
        mask = tmp[:, 4:5]
        feat = tmp[:, 5:]
        return flow, mask, feat


class IFNetLiteTRT(nn.Module):
    """
    DistilDRBA v2_lite TensorRT export version.

    This version handles only backward interpolation (timestep in [0.5, 1.0))
    which is the standard case for TAS interpolation between consecutive frames.

    The encoder (Head) is included inside this model for proper feature resolution handling.

    Inputs to forward:
        - img0, img1, img2: Input frames [B, 3, H, W] at full resolution
        - timestep: Timestep tensor [B, 1, H, W] in range [0.5, 1.0)

    Output:
        - Interpolated frame [B, 3, H, W]
    """

    def __init__(self, scale=1.0):
        super(IFNetLiteTRT, self).__init__()
        self.scale = scale
        self.scaleListDefault = [16, 8, 4]

        self.encode = Head()
        self.block0 = IFBlockV2Lite(9 + 48 + 1, c=192)
        self.block1 = IFBlockV2Lite(9 + 48 + 8 + 4 + 8 + 32, c=128)
        self.block2 = IFBlockV2Lite(9 + 48 + 8 + 4 + 8 + 32, c=96)

    def batchInterpolate(
        self, *tensor, scaleFactor=1.0, mode="bilinear", alignCorners=False
    ):
        if scaleFactor != 1:
            return [
                F.interpolate(
                    x, scale_factor=scaleFactor, mode=mode, align_corners=alignCorners
                )
                for x in tensor
            ]
        return tensor

    def batchWarp(self, *tensor, flow):
        return [warp(x, flow) for x in tensor]

    def forward(self, img0, img1, img2, timestep):
        """
        Forward for backward interpolation (t in [0.5, 1.0)).
        For backward: inp0=img1, inp1=img0, h0=f1, h1=f0

        Args:
            img0: Previous frame [B, 3, H, W]
            img1: Current frame [B, 3, H, W]
            img2: Next frame [B, 3, H, W]
            timestep: Timestep tensor [B, 1, H, W] in range [0.5, 1.0)
        """
        scaleList = [s / self.scale for s in self.scaleListDefault]
        encodeScale = min(1 / scaleList[-1], 1)

        # Encode features at encodeScale resolution
        img0Encode, img1Encode, img2Encode = self.batchInterpolate(
            img0, img1, img2, scaleFactor=encodeScale
        )
        f0 = self.encode(img0Encode)
        f1 = self.encode(img1Encode)
        f2 = self.encode(img2Encode)

        # For backward interpolation: use img1 as source, img0 as target
        inp0, inp1 = img1, img0
        h0, h1 = f1, f0

        block = [self.block0, self.block1, self.block2]
        flow, mask = None, None

        for scaleIdx in range(len(scaleList)):
            currentScale = 1 / scaleList[scaleIdx]
            f0S, f1S, f2S = self.batchInterpolate(
                f0, f1, f2, scaleFactor=currentScale / encodeScale
            )
            img0S, img1S, img2S = self.batchInterpolate(
                img0, img1, img2, scaleFactor=currentScale
            )

            if flow is None:
                # Create timestep tensor at current scale
                timestepS = F.interpolate(
                    timestep,
                    scale_factor=currentScale,
                    mode="bilinear",
                    align_corners=False,
                )
                flow, mask, feat = block[scaleIdx](
                    torch.cat(
                        (
                            img0S,
                            img1S,
                            img2S,
                            f0S,
                            f1S,
                            f2S,
                            timestepS,
                        ),
                        1,
                    )
                )
            else:
                h0S, h1S = self.batchInterpolate(
                    h0, h1, scaleFactor=currentScale / encodeScale
                )
                inp0S, inp1S = self.batchInterpolate(
                    inp0, inp1, scaleFactor=currentScale
                )

                if currentScale <= 1:
                    wf0, warpedImg0 = self.batchWarp(h0S, inp0S, flow=flow[:, :2])
                    wf1, warpedImg1 = self.batchWarp(h1S, inp1S, flow=flow[:, 2:4])
                else:
                    wf0, warpedImg0 = self.batchWarp(h0, inp0, flow=flow[:, :2])
                    wf1, warpedImg1 = self.batchWarp(h1, inp1, flow=flow[:, 2:4])
                    wf0, wf1, warpedImg0, warpedImg1 = self.batchInterpolate(
                        wf0, wf1, warpedImg0, warpedImg1, scaleFactor=currentScale
                    )
                    flow, mask, feat = self.batchInterpolate(
                        flow, mask, feat, scaleFactor=currentScale
                    )
                    flow = flow * currentScale

                timestepS = F.interpolate(
                    timestep,
                    scale_factor=currentScale,
                    mode="bilinear",
                    align_corners=False,
                )
                flow, mask, feat = block[scaleIdx](
                    torch.cat(
                        (
                            img0S,
                            img1S,
                            img2S,
                            f0S,
                            f1S,
                            f2S,
                            warpedImg0,
                            warpedImg1,
                            wf0,
                            wf1,
                            timestepS,
                            mask,
                            feat,
                        ),
                        1,
                    ),
                    flow,
                )

            # Scale flow for next iteration
            if scaleIdx == len(scaleList) - 1:
                flowScale = scaleList[scaleIdx]
            elif scaleList[scaleIdx + 1] >= 1:
                flowScale = scaleList[scaleIdx] / scaleList[scaleIdx + 1]
            else:
                flowScale = 1

            if flowScale != 1:
                flow, mask, feat = self.batchInterpolate(
                    flow, mask, feat, scaleFactor=flowScale
                )
                flow = flow * flowScale

        warpedImg0 = warp(inp0, flow[:, :2])
        warpedImg1 = warp(inp1, flow[:, 2:4])
        mask = torch.sigmoid(mask)
        merged = warpedImg0 * mask + warpedImg1 * (1 - mask)
        return merged
