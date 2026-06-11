"""
DistilDRBA - Distilled Distance Ratio-Based Approach for Video Frame Interpolation
Ported from: https://github.com/routineLife1/DistilDRBA

This module implements both v1 (full, 5 blocks) and v2_lite (lite, 3 blocks) architectures.
The architecture selection is explicit via the `lite` parameter.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

backwarpTenGrid = {}


def warp(tenInput, tenFlow):
    """
    Backward warping using grid_sample.

    Args:
        tenInput: Input tensor to warp [B, C, H, W]
        tenFlow: Optical flow [B, 2, H, W]

    Returns:
        Warped tensor [B, C, H, W]
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
    """
    Feature encoder head.
    Encodes 3-channel RGB image to 16-channel feature map.
    Architecture: stride-2 conv -> conv -> conv -> transpose conv (back to original resolution)
    """

    def __init__(self):
        super(Head, self).__init__()
        self.cnn0 = nn.Conv2d(3, 16, 3, 2, 1)
        self.cnn1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(16, 16, 4, 2, 1)
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
    """Residual convolution block with learnable scale parameter"""

    def __init__(self, c, dilation=1):
        super(ResConv, self).__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.relu(self.conv(x) * self.beta + x)


class IFBlockV1(nn.Module):
    """
    Flow estimation block for v1 (full) model.
    Outputs: flow(4) + mask(1) + feat(8) + tmap(1) = 14 channels
    """

    def __init__(self, inPlanes, c=64):
        super(IFBlockV1, self).__init__()
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
        # Output: 4*(4+1+8+1) = 56 channels before PixelShuffle -> 14 channels after
        self.lastconv = nn.Sequential(
            nn.ConvTranspose2d(c, 4 * (4 + 1 + 8 + 1), 4, 2, 1), nn.PixelShuffle(2)
        )

    def forward(self, x, flow=None):
        if flow is not None:
            x = torch.cat((x, flow), 1)
        feat = self.conv0(x)
        feat = self.convblock(feat)
        tmp = self.lastconv(feat)
        flow = tmp[:, :4]
        mask = tmp[:, 4:5]
        feat = tmp[:, 5:13]
        tmap = torch.sigmoid(tmp[:, 13:])  # Must be in 0-1 range
        return flow, mask, feat, tmap


class IFBlockV2Lite(nn.Module):
    """
    Flow estimation block for v2_lite model.
    Outputs: flow(4) + mask(1) + feat(8) = 13 channels
    """

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
        # Output: 4*(4+1+8) = 52 channels before PixelShuffle -> 13 channels after
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


class IFNet(nn.Module):
    """
    DistilDRBA IFNet - main interpolation network.

    Supports both v1 (full, 5 blocks) and v2_lite (lite, 3 blocks) architectures.

    Args:
        lite (bool): If True, use v2_lite architecture (3 blocks).
                     If False, use v1 architecture (5 blocks).
        scale (float): Scale factor for processing (0.5 for 4K, 1.0 for 1080p).
    """

    def __init__(self, lite=False, scale=1.0):
        super(IFNet, self).__init__()
        self.lite = lite
        self.scale = scale

        if lite:
            # v2_lite: 3 blocks
            self.block0 = IFBlockV2Lite(9 + 48 + 1, c=192)
            self.block1 = IFBlockV2Lite(9 + 48 + 8 + 4 + 8 + 32, c=128)
            self.block2 = IFBlockV2Lite(9 + 48 + 8 + 4 + 8 + 32, c=96)
            self.scaleListDefault = [16, 8, 4]
        else:
            # v1: 5 blocks
            self.block0 = IFBlockV1(3 * 3 + 16 * 3 + 1, c=192)
            self.block1 = IFBlockV1(8 + 4 + 8 + 32, c=128)
            self.block2 = IFBlockV1(8 + 4 + 8 + 32, c=96)
            self.block3 = IFBlockV1(8 + 4 + 8 + 32, c=64)
            self.block4 = IFBlockV1(8 + 4 + 8 + 32, c=32)
            self.scaleListDefault = [16, 8, 4, 2, 1]

        self.encode = Head()

    def batchInterpolate(
        self, *tensor, scaleFactor=1.0, mode="bilinear", alignCorners=False
    ):
        """Batch interpolate multiple tensors"""
        if scaleFactor != 1:
            return [
                F.interpolate(
                    x, scale_factor=scaleFactor, mode=mode, align_corners=alignCorners
                )
                for x in tensor
            ]
        return tensor

    def batchWarp(self, *tensor, flow):
        """Batch warp multiple tensors with the same flow"""
        return [warp(x, flow) for x in tensor]

    def forward(
        self, img0, img1, img2, f0=None, f1=None, f2=None, timestep=0.5, scaleList=None
    ):
        """
        Forward pass for frame interpolation.

        Args:
            img0: Previous frame [B, 3, H, W], normalized [0, 1]
            img1: Current/middle frame [B, 3, H, W]
            img2: Next frame [B, 3, H, W]
            f0: Cached features for img0 (optional)
            f1: Cached features for img1 (optional)
            f2: Cached features for img2 (optional)
            timestep: Interpolation timestep in range [0.5, 1.5], where 1.0 = img1
            scaleList: List of scale factors for multi-scale processing

        Returns:
            merged: Interpolated frame [B, 3, H, W]
            f0, f1, f2: Updated feature caches for reuse
        """
        if scaleList is None:
            # Apply scale factor to default scale list
            # Original: scale_list = [x / scale for x in default_list]
            # So with scale=0.5 (4K), [16,8,4] -> [32,16,8] (larger values = more downscaling)
            scaleList = [s / self.scale for s in self.scaleListDefault]

        # Encode scale is capped at 1.0
        encodeScale = min(1 / scaleList[-1], 1)

        # Encode features if not cached
        img0Encode, img1Encode, img2Encode = self.batchInterpolate(
            img0[:, :3], img1[:, :3], img2[:, :3], scaleFactor=encodeScale
        )
        f0 = self.encode(img0Encode) if f0 is None else f0
        f1 = self.encode(img1Encode) if f1 is None else f1
        f2 = self.encode(img2Encode) if f2 is None else f2
        del img0Encode, img1Encode, img2Encode

        # Select input pair based on timestep
        # Original logic: 0.5 <= t < 1.0 uses (img1, img0), 1.0 < t <= 1.5 uses (img1, img2)
        # t = 1.0 exactly is invalid (would just return img1)
        if 0.5 <= timestep < 1.0:
            inp0, inp1 = img1.clone(), img0.clone()
            h0, h1 = f1.clone(), f0.clone()
        elif 1.0 < timestep <= 1.5:
            inp0, inp1 = img1.clone(), img2.clone()
            h0, h1 = f1.clone(), f2.clone()
        else:
            raise ValueError(
                "timestep should be in range [0.5, 1.0) or (1.0, 1.5], not exactly 1.0"
            )

        if self.lite:
            return self.forwardLite(
                img0,
                img1,
                img2,
                f0,
                f1,
                f2,
                h0,
                h1,
                inp0,
                inp1,
                timestep,
                scaleList,
                encodeScale,
            )
        else:
            return self.forwardV1(
                img0,
                img1,
                img2,
                f0,
                f1,
                f2,
                h0,
                h1,
                inp0,
                inp1,
                timestep,
                scaleList,
                encodeScale,
            )

    def forwardLite(
        self,
        img0,
        img1,
        img2,
        f0,
        f1,
        f2,
        h0,
        h1,
        inp0,
        inp1,
        timestep,
        scaleList,
        encodeScale,
    ):
        """Forward pass for v2_lite architecture (3 blocks)"""
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
                flow, mask, feat = block[scaleIdx](
                    torch.cat(
                        (
                            img0S[:, :3],
                            img1S[:, :3],
                            img2S[:, :3],
                            f0S,
                            f1S,
                            f2S,
                            (img0S[:, :1].clone() * 0 + 1) * timestep,
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
                    flow *= currentScale

                flow, mask, feat = block[scaleIdx](
                    torch.cat(
                        (
                            img0S[:, :3],
                            img1S[:, :3],
                            img2S[:, :3],
                            f0S,
                            f1S,
                            f2S,
                            warpedImg0[:, :3],
                            warpedImg1[:, :3],
                            wf0,
                            wf1,
                            img0S[:, :1].clone() * 0 + timestep,
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
                flow *= flowScale

        warpedImg0 = warp(inp0, flow[:, :2])
        warpedImg1 = warp(inp1, flow[:, 2:4])
        mask = torch.sigmoid(mask)
        merged = warpedImg0 * mask + warpedImg1 * (1 - mask)
        return merged, f0, f1, f2

    def forwardV1(
        self,
        img0,
        img1,
        img2,
        f0,
        f1,
        f2,
        h0,
        h1,
        inp0,
        inp1,
        timestep,
        scaleList,
        encodeScale,
    ):
        """Forward pass for v1 architecture (5 blocks)"""
        block = [self.block0, self.block1, self.block2, self.block3, self.block4]
        flow, mask = None, None

        for scaleIdx in range(len(scaleList)):
            currentScale = 1 / scaleList[scaleIdx]

            if flow is None:
                f0S, f1S, f2S = self.batchInterpolate(
                    f0, f1, f2, scaleFactor=currentScale / encodeScale
                )
                img0S, img1S, img2S = self.batchInterpolate(
                    img0, img1, img2, scaleFactor=currentScale
                )

                flow, mask, feat, tmap = block[scaleIdx](
                    torch.cat(
                        (
                            img0S[:, :3],
                            img1S[:, :3],
                            img2S[:, :3],
                            f0S,
                            f1S,
                            f2S,
                            (img0S[:, :1].clone() * 0 + 1) * timestep,
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
                    flow, mask, feat, tmap = self.batchInterpolate(
                        flow, mask, feat, tmap, scaleFactor=currentScale
                    )
                    flow *= currentScale

                fd, mask, feat, tmap = block[scaleIdx](
                    torch.cat(
                        (
                            warpedImg0[:, :3],
                            warpedImg1[:, :3],
                            wf0,
                            wf1,
                            tmap,
                            mask,
                            feat,
                        ),
                        1,
                    ),
                    flow,
                )

                flow = flow + fd

            # Scale flow for next iteration
            if scaleIdx == len(scaleList) - 1:
                flowScale = scaleList[scaleIdx]
            elif scaleList[scaleIdx + 1] >= 1:
                flowScale = scaleList[scaleIdx] / scaleList[scaleIdx + 1]
            else:
                flowScale = 1

            if flowScale != 1:
                flow, mask, feat, tmap = self.batchInterpolate(
                    flow, mask, feat, tmap, scaleFactor=flowScale
                )
                flow *= flowScale

        warpedImg0 = warp(inp0, flow[:, :2])
        warpedImg1 = warp(inp1, flow[:, 2:4])
        mask = torch.sigmoid(mask)
        merged = warpedImg0 * mask + warpedImg1 * (1 - mask)
        return merged, f0, f1, f2
