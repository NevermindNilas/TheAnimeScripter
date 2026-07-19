"""
DistilDRBA DirectML/OpenVINO export architecture.

Mirrors ``IFNet_distildrba_tensorrt`` (backward interpolation only, timestep in
[0.5, 1.0), 4 inputs -> 1 output) with one substitution: every ``grid_sample``
becomes a decomposition into floor/gather/weighted-sum, because DirectML has no
GridSample kernel and would silently fall the whole warp back to the CPU. Same
reason ``Rife_directml`` exists as a hand-copy of the RIFE archs.

Weights load from the same ``v1.pkl`` / ``v2_lite.pkl`` checkpoints as the CUDA
and TensorRT paths.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

_gridCache = {}


def _baseGrid(b, h, w, device):
    """Identity sampling grid for a [b, 2, h, w] flow, cached per shape.

    Held in fp32 regardless of model dtype: see ``warp``.
    """
    k = (str(device), b, h, w)
    if k not in _gridCache:
        tenHorizontal = (
            torch.linspace(-1.0, 1.0, w, dtype=torch.float32)
            .view(1, 1, 1, w)
            .expand(b, -1, h, -1)
        )
        tenVertical = (
            torch.linspace(-1.0, 1.0, h, dtype=torch.float32)
            .view(1, 1, h, 1)
            .expand(b, -1, -1, w)
        )
        _gridCache[k] = torch.cat([tenHorizontal, tenVertical], 1).to(device)
    return _gridCache[k]


def warp(tenInput, tenFlow):
    """Backward warp, grid_sample decomposed into ops DirectML can run.

    The coordinate math stays in fp32 even when the model runs in fp16. A
    normalised coordinate near the frame edge has an fp16 spacing of ~1e-3,
    which at 1080p is a ~0.5px sampling error -- enough to visibly smear the
    warp. Only the 2-channel grid pays the fp32 cost; the gathers and the
    bilinear blend run in the model dtype.

    ``tenInput`` and ``tenFlow`` may differ in spatial size (the coarse levels
    warp a full-resolution frame with a low-resolution flow); the output takes
    the flow's size, matching ``F.grid_sample`` semantics.
    """
    b, _, hOut, wOut = tenFlow.shape
    _, c, hIn, wIn = tenInput.shape

    flow = tenFlow.float()
    grid = _baseGrid(b, hOut, wOut, tenFlow.device) + torch.cat(
        [
            flow[:, 0:1] / ((wIn - 1.0) / 2.0),
            flow[:, 1:2] / ((hIn - 1.0) / 2.0),
        ],
        1,
    )

    pixelX = (grid[:, 0] + 1.0) * 0.5 * (wIn - 1.0)
    pixelY = (grid[:, 1] + 1.0) * 0.5 * (hIn - 1.0)

    x0f = torch.floor(pixelX)
    y0f = torch.floor(pixelY)

    dx = (pixelX - x0f).unsqueeze(1).to(tenInput.dtype)
    dy = (pixelY - y0f).unsqueeze(1).to(tenInput.dtype)

    # padding_mode="border": clamp the taps, do not zero them.
    x0 = torch.clamp(x0f, 0, wIn - 1).long()
    x1 = torch.clamp(x0f + 1, 0, wIn - 1).long()
    y0 = torch.clamp(y0f, 0, hIn - 1).long()
    y1 = torch.clamp(y0f + 1, 0, hIn - 1).long()

    flat = tenInput.reshape(b, c, hIn * wIn)

    def gather(yi, xi):
        idx = (yi * wIn + xi).reshape(b, 1, hOut * wOut).expand(-1, c, -1)
        return torch.gather(flat, 2, idx).reshape(b, c, hOut, wOut)

    v00 = gather(y0, x0)
    v01 = gather(y1, x0)
    v10 = gather(y0, x1)
    v11 = gather(y1, x1)

    return (
        v00 * ((1 - dx) * (1 - dy))
        + v10 * (dx * (1 - dy))
        + v01 * ((1 - dx) * dy)
        + v11 * (dx * dy)
    )


def conv(inPlanes, outPlanes, kernelSize=3, stride=1, padding=1, dilation=1):
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
        super().__init__()
        self.cnn0 = nn.Conv2d(3, 16, 3, 2, 1)
        self.cnn1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(16, 16, 4, 2, 1)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x = self.relu(self.cnn0(x))
        x = self.relu(self.cnn1(x))
        x = self.relu(self.cnn2(x))
        return self.cnn3(x)


class ResConv(nn.Module):
    def __init__(self, c, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.relu(self.conv(x) * self.beta + x)


class _IFNetBase(nn.Module):
    """Shared scale/warp plumbing for the two DistilDRBA variants."""

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


class IFBlockV2Lite(nn.Module):
    """Flow estimation block for v2_lite model (3 blocks)"""

    def __init__(self, inPlanes, c=64):
        super().__init__()
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


class IFNetLiteDML(_IFNetBase):
    """
    DistilDRBA v2_lite, DirectML/OpenVINO export version.

    Backward interpolation only (timestep in [0.5, 1.0)), which is the case TAS
    drives between consecutive frames. The encoder (Head) is inside the graph so
    the exported ONNX is self-contained: 4 inputs, 1 output, no recurrent state.

    Inputs:  img0, img1, img2 [B, 3, H, W]; timestep [B, 1, H, W]
    Output:  interpolated frame [B, 3, H, W]
    """

    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale
        self.scaleListDefault = [16, 8, 4]

        self.encode = Head()
        self.block0 = IFBlockV2Lite(9 + 48 + 1, c=192)
        self.block1 = IFBlockV2Lite(9 + 48 + 8 + 4 + 8 + 32, c=128)
        self.block2 = IFBlockV2Lite(9 + 48 + 8 + 4 + 8 + 32, c=96)

    def forward(self, img0, img1, img2, timestep):
        scaleList = [s / self.scale for s in self.scaleListDefault]
        encodeScale = min(1 / scaleList[-1], 1)

        img0Encode, img1Encode, img2Encode = self.batchInterpolate(
            img0, img1, img2, scaleFactor=encodeScale
        )
        f0 = self.encode(img0Encode)
        f1 = self.encode(img1Encode)
        f2 = self.encode(img2Encode)

        # Backward interpolation: img1 is the source, img0 the target.
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
                timestepS = F.interpolate(
                    timestep,
                    scale_factor=currentScale,
                    mode="bilinear",
                    align_corners=False,
                )
                flow, mask, feat = block[scaleIdx](
                    torch.cat((img0S, img1S, img2S, f0S, f1S, f2S, timestepS), 1)
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
        return warpedImg0 * mask + warpedImg1 * (1 - mask)


class IFBlockV1(nn.Module):
    """
    Flow estimation block for v1 (full) model.
    Outputs: flow(4) + mask(1) + feat(8) + tmap(1) = 14 channels
    """

    def __init__(self, inPlanes, c=64):
        super().__init__()
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


class IFNetFullDML(_IFNetBase):
    """
    DistilDRBA v1 (Full), DirectML/OpenVINO export version. See IFNetLiteDML.
    """

    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale
        self.scaleListDefault = [16, 8, 4, 2, 1]

        self.encode = Head()
        self.block0 = IFBlockV1(3 * 3 + 16 * 3 + 1, c=192)
        self.block1 = IFBlockV1(8 + 4 + 8 + 32, c=128)
        self.block2 = IFBlockV1(8 + 4 + 8 + 32, c=96)
        self.block3 = IFBlockV1(8 + 4 + 8 + 32, c=64)
        self.block4 = IFBlockV1(8 + 4 + 8 + 32, c=32)

    def forward(self, img0, img1, img2, timestep):
        scaleList = [s / self.scale for s in self.scaleListDefault]
        encodeScale = min(1 / scaleList[-1], 1)

        img0Encode, img1Encode, img2Encode = self.batchInterpolate(
            img0, img1, img2, scaleFactor=encodeScale
        )
        f0 = self.encode(img0Encode)
        f1 = self.encode(img1Encode)
        f2 = self.encode(img2Encode)

        inp0, inp1 = img1, img0
        h0, h1 = f1, f0

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
                timestepS = F.interpolate(
                    timestep,
                    scale_factor=currentScale,
                    mode="bilinear",
                    align_corners=False,
                )

                flow, mask, feat, tmap = block[scaleIdx](
                    torch.cat((img0S, img1S, img2S, f0S, f1S, f2S, timestepS), 1)
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
                    flow = flow * currentScale

                fd, mask, feat, tmap = block[scaleIdx](
                    torch.cat((warpedImg0, warpedImg1, wf0, wf1, tmap, mask, feat), 1),
                    flow,
                )
                flow = flow + fd

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
                flow = flow * flowScale

        warpedImg0 = warp(inp0, flow[:, :2])
        warpedImg1 = warp(inp1, flow[:, 2:4])
        mask = torch.sigmoid(mask)
        return warpedImg0 * mask + warpedImg1 * (1 - mask)
