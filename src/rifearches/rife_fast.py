"""
Fast CUDA path for all RIFE variants (4.6, 4.15-lite, 4.16-lite, 4.17, 4.18, 4.20,
4.21, 4.22, 4.22-lite, 4.25, 4.25-lite, 4.25-heavy). Math-equivalent to baselines
modulo fp16 noise.

Optimizations (all load-time, free):
  1. ResConv beta multiply folded into conv weight.
  2. ConvTranspose2d(k=4,s=2,p=1) -> Conv2d(k=3,s=1,p=1, out=4C) + PixelShuffle(2)
     via weight rearrangement (math identical, faster cuDNN tactic).
  3. Backwarp grid + flow-divider precomputed once on first forward
     (kills per-call cat in warplayer.warp).

Keeps v1 API: cache(), cacheReset(frame), forward(img0, img1, timestep).

Exports IFNet46, IFNet415Lite, IFNet416Lite, IFNet417, IFNet418, IFNet420,
        IFNet421, IFNet422, IFNet422Lite, IFNet425, IFNet425Lite, IFNet425Heavy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# shared modules
# -----------------------------------------------------------------------------


def _conv(inPlanes, outPlanes, kernelSize=3, stride=1, padding=1, dilation=1):
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
        nn.LeakyReLU(0.2, inplace=True),
    )


_DECONV_MAPPING = {
    (0, 0): [((0, 0), (3, 3)), ((0, 1), (3, 1)), ((1, 0), (1, 3)), ((1, 1), (1, 1))],
    (0, 1): [((0, 1), (3, 2)), ((0, 2), (3, 0)), ((1, 1), (1, 2)), ((1, 2), (1, 0))],
    (1, 0): [((1, 0), (2, 3)), ((1, 1), (2, 1)), ((2, 0), (0, 3)), ((2, 1), (0, 1))],
    (1, 1): [((1, 1), (2, 2)), ((1, 2), (2, 0)), ((2, 1), (0, 2)), ((2, 2), (0, 0))],
}


def _repackDeconv(deconvW, deconvB):
    """ConvTranspose2d(Cin, Cout, 4, 2, 1) -> Conv2d(Cin, 4*Cout, 3, 1, 1)."""
    Cin, Cout, _, _ = deconvW.shape
    device, dtype = deconvW.device, deconvW.dtype
    newK = torch.zeros(4 * Cout, Cin, 3, 3, dtype=dtype, device=device)
    newB = (
        torch.zeros(4 * Cout, dtype=dtype, device=device)
        if deconvB is not None
        else None
    )
    for (s, t), positions in _DECONV_MAPPING.items():
        sub = s * 2 + t
        for (a, b), (ky, kx) in positions:
            newK[sub::4, :, a, b] = deconvW[:, :, ky, kx].t()
        if deconvB is not None:
            newB[sub::4] = deconvB
    return newK, newB


class _ResConv(nn.Module):
    def __init__(self, c, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self._betaFolded = False

    def forward(self, x):
        if self._betaFolded:
            return self.relu(self.conv(x) + x)
        return self.relu(self.conv(x) * self.beta + x)

    def foldBeta(self):
        if self._betaFolded:
            return
        with torch.no_grad():
            bf = self.beta.view(-1)
            self.conv.weight.mul_(bf.view(-1, 1, 1, 1))
            self.conv.bias.mul_(bf)
        self._betaFolded = True


class _IFBlock(nn.Module):
    """IFBlock with deconv->Conv2d+double-PS rewrite. lastOutCh: 6 or 13."""

    def __init__(self, inPlanes, c=64, lastOutCh=13):
        super().__init__()
        self.conv0 = nn.Sequential(
            _conv(inPlanes, c // 2, 3, 2, 1),
            _conv(c // 2, c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(*[_ResConv(c) for _ in range(8)])
        # baseline: ConvTranspose2d(c, 4*lastOutCh, 4, 2, 1) + PixelShuffle(2) -> lastOutCh ch at 4x
        # fast: Conv2d(c, 4*(4*lastOutCh), 3, 1, 1) + PS(2) -> 4*lastOutCh ch at 2x -> PS(2) -> lastOutCh ch at 4x
        self.lastconv = nn.Conv2d(c, 4 * (4 * lastOutCh), 3, 1, 1)
        self.lastps0 = nn.PixelShuffle(2)
        self.lastps1 = nn.PixelShuffle(2)
        self.lastOutCh = lastOutCh

    def forward(self, x, flow=None, scale=1):
        if scale != 1:
            x = F.interpolate(
                x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False
            )
        if flow is not None:
            if scale != 1:
                flow = F.interpolate(
                    flow, scale_factor=1.0 / scale, mode="bilinear", align_corners=False
                ) * (1.0 / scale)
            x = torch.cat((x, flow), 1)
        return self._tail(x, scale)

    def forwardPrescaled(self, x, scale=1):
        """x is already at block resolution with flow channels appended
        (downsample-then-cat == cat-then-downsample: bilinear is per-channel)."""
        return self._tail(x, scale)

    def _tail(self, x, scale):
        feat = self.conv0(x)
        feat = self.convblock(feat)
        tmp = self.lastps1(self.lastps0(self.lastconv(feat)))
        if scale != 1:
            tmp = F.interpolate(
                tmp, scale_factor=scale, mode="bilinear", align_corners=False
            )
        flow_out = tmp[:, :4]
        if scale != 1:
            flow_out = flow_out * scale
        mask = tmp[:, 4:5]
        if self.lastOutCh >= 13:
            feat_out = tmp[:, 5:]  # 8 ch
            return flow_out, mask, feat_out
        return flow_out, mask


class _Head(nn.Module):
    """Head encoder. midC=16/32, outC=4/8. cnn3 is deconv replaced with Conv2d+PS."""

    def __init__(self, midC=16, outC=4):
        super().__init__()
        self.cnn0 = nn.Conv2d(3, midC, 3, 2, 1)
        self.cnn1 = nn.Conv2d(midC, midC, 3, 1, 1)
        self.cnn2 = nn.Conv2d(midC, midC, 3, 1, 1)
        # baseline: ConvTranspose2d(midC, outC, 4, 2, 1). Repacked: Conv2d(midC, 4*outC, 3, 1, 1) + PS(2)
        self.cnn3 = nn.Conv2d(midC, 4 * outC, 3, 1, 1)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x, feat=False):
        x0 = self.cnn0(x)
        x = self.relu(x0)
        x1 = self.cnn1(x)
        x = self.relu(x1)
        x2 = self.cnn2(x)
        x = self.relu(x2)
        x3 = F.pixel_shuffle(self.cnn3(x), 2)
        if feat:
            return [x0, x1, x2, x3]
        return x3


# -----------------------------------------------------------------------------
# generic fast IFNet, parameterized by config
# -----------------------------------------------------------------------------


class _FastIFNet(nn.Module):
    """Generic fast IFNet supporting all RIFE 4.x variants.

    Config (passed to __init__):
        channels:     list of per-block c (len = numBlocks)
        inPlanesList: list of per-block inPlanes (post-flow concat for non-first)
        lastOutCh:    6 (older) or 13 (newer w/ feat); also sets usesFeat
        hasHead:      True except for rife4.6
        headMidC:     16 or 32
        headOutC:     4 or 8 (ignored if not hasHead)
        scaleBase:    [16,8,4,2,1] (5 blocks) or [8,4,2,1] (4 blocks)
        maskMode:     "replace" (most) or "add" (rife4.6 only)
    """

    def __init__(
        self,
        *,
        channels,
        inPlanesList,
        lastOutCh,
        hasHead,
        headMidC,
        headOutC,
        scaleBase,
        maskMode,
        ensemble=False,
        dynamicScale=False,
        scale=1,
        interpolateFactor=2,
        staticStep=False,
    ):
        super().__init__()
        assert len(channels) == len(inPlanesList) == len(scaleBase)
        self._hasHead = hasHead
        self._usesFeat = lastOutCh >= 13
        self._maskMode = maskMode
        self._headOutC = headOutC if hasHead else 0

        for i, (cin, c) in enumerate(zip(inPlanesList, channels)):
            setattr(self, f"block{i}", _IFBlock(cin, c=c, lastOutCh=lastOutCh))
        self.blocks = [getattr(self, f"block{i}") for i in range(len(channels))]

        if hasHead:
            self.encode = _Head(midC=headMidC, outC=headOutC)
        else:
            self.encode = None

        self.scale_list = [s / scale for s in scaleBase]
        self.ensemble = ensemble
        self.dynamicScale = dynamicScale
        self.interpolateFactor = interpolateFactor
        self.staticStep = staticStep
        if self.staticStep:
            self.timesteps = None
        self.counter = 1

        self.f0 = None
        self.f1 = None

        self._backWarp = None
        self._tenFlow = None
        self._gridShape = None
        self._pendingDeconvWeights = {}
        self._repacked = False

    # ---- weight loading / repack -------------------------------------------

    def load_state_dict(self, stateDict, strict=True):
        remapped = {}
        for k, v in stateDict.items():
            nk = k
            if k.endswith(".lastconv.0.weight") or k.endswith(".lastconv.0.bias"):
                nk = k.replace(".lastconv.0.", ".lastconvRaw.")
            remapped[nk] = v
        cleaned = {}
        for k, v in remapped.items():
            if (
                ".lastconvRaw." in k
                or k == "encode.cnn3.weight"
                or k == "encode.cnn3.bias"
            ):
                self._pendingDeconvWeights[k] = v
            else:
                cleaned[k] = v
        missing, unexpected = super().load_state_dict(cleaned, strict=False)
        self._repackDeconvs()
        return type("R", (), {"missing_keys": missing, "unexpected_keys": unexpected})()

    def _repackDeconvs(self):
        prefixes = set()
        for k in self._pendingDeconvWeights:
            if k.endswith(".weight"):
                prefixes.add(k[: -len(".weight")])
            elif k.endswith(".bias"):
                prefixes.add(k[: -len(".bias")])
        for pfx in prefixes:
            W = self._pendingDeconvWeights[pfx + ".weight"]
            b = self._pendingDeconvWeights.get(pfx + ".bias")
            newK, newB = _repackDeconv(W, b)
            if pfx == "encode.cnn3":
                target = self.encode.cnn3
            elif pfx.endswith(".lastconvRaw"):
                blkName = pfx.split(".")[0]
                target = getattr(self, blkName).lastconv
            else:
                raise RuntimeError(f"unknown deconv prefix: {pfx}")
            with torch.no_grad():
                target.weight.data.copy_(newK.to(target.weight.dtype))
                if b is not None:
                    target.bias.data.copy_(newB.to(target.bias.dtype))
        self._pendingDeconvWeights = {}

    def repackWeights(self):
        if self._repacked:
            return
        for m in self.modules():
            if isinstance(m, _ResConv):
                m.foldBeta()
        self._repacked = True

    # ---- runtime helpers ---------------------------------------------------

    def _ensureGrid(self, ph, pw, dtype, device):
        key = (ph, pw, dtype, device)
        if self._gridShape == key:
            return
        hMul = 2.0 / (pw - 1)
        vMul = 2.0 / (ph - 1)
        self._tenFlow = torch.tensor([hMul, vMul], dtype=dtype, device=device).view(
            1, 2, 1, 1
        )
        bx32 = (
            torch.linspace(-1.0, 1.0, pw, device=device, dtype=torch.float32)
            .view(1, 1, 1, -1)
            .expand(-1, -1, ph, -1)
        )
        by32 = (
            torch.linspace(-1.0, 1.0, ph, device=device, dtype=torch.float32)
            .view(1, 1, -1, 1)
            .expand(-1, -1, -1, pw)
        )
        self._backWarp = torch.cat([bx32, by32], dim=1).to(dtype=dtype)
        self._gridShape = key

    def _warp(self, inp, flow2):
        # addcmul fuses backWarp + flow*tenFlow into one elementwise pass
        grid = torch.addcmul(self._backWarp, flow2, self._tenFlow).permute(0, 2, 3, 1)
        return F.grid_sample(
            inp, grid, mode="bilinear", padding_mode="border", align_corners=True
        )

    def cache(self):
        if self._hasHead and self.f0 is not None and self.f1 is not None:
            self.f0.copy_(self.f1, non_blocking=True)

    def cacheReset(self, frame):
        if self._hasHead:
            self.f0 = self.encode(frame[:, :3])

    # ---- forward -----------------------------------------------------------

    def forward(self, img0, img1, timestep):
        ph, pw = img0.shape[-2], img0.shape[-1]
        self._ensureGrid(ph, pw, img0.dtype, img0.device)

        if self._hasHead:
            if self.interpolateFactor == 2 or self.counter == self.interpolateFactor:
                if self.interpolateFactor != 2:
                    self.counter = 1
                if self.f0 is None:
                    self.f0 = self.encode(img0[:, :3])
                self.f1 = self.encode(img1[:, :3])
            else:
                if self.f0 is None or self.f1 is None:
                    self.f0 = self.encode(img0[:, :3])
                    self.f1 = self.encode(img1[:, :3])
            if self.interpolateFactor != 2:
                self.counter += 1

        warped_img0 = img0
        warped_img1 = img1
        wf0 = self.f0
        wf1 = self.f1
        flow = None
        mask = None
        feat = None

        # img and feature warps share the same flow -> one grid_sample per side
        # on the channel-concat (grid_sample is channel-independent, exact).
        # img0/f0 are constant across blocks, so concat once per forward.
        if self._hasHead:
            imgf0 = torch.cat((img0[:, :3], self.f0), 1)
            imgf1 = torch.cat((img1[:, :3], self.f1), 1)

        lastIdx = len(self.blocks) - 1

        for i, blk in enumerate(self.blocks):
            scale = self.scale_list[i]
            # Downsample components individually and concat at block resolution:
            # bilinear is per-channel, so downsample(cat(...)) == cat(downsample
            # of each part) -- this skips writing+reading the full-res concat.
            lowResCat = (not self.ensemble) and scale != 1
            inv = 1.0 / scale

            def ds(t):
                return F.interpolate(
                    t, scale_factor=inv, mode="bilinear", align_corners=False
                )

            if flow is None:
                if lowResCat:
                    if self._hasHead:
                        a, b = ds(imgf0), ds(imgf1)
                        parts = [a[:, :3], b[:, :3], a[:, 3:], b[:, 3:], ds(timestep)]
                    else:
                        parts = [ds(img0[:, :3]), ds(img1[:, :3]), ds(timestep)]
                    out = blk.forwardPrescaled(torch.cat(parts, 1), scale)
                else:
                    inp_parts = [img0[:, :3], img1[:, :3]]
                    if self._hasHead:
                        inp_parts += [self.f0, self.f1]
                    inp_parts.append(timestep)
                    inp = torch.cat(inp_parts, 1)
                    out = blk(inp, flow=None, scale=scale)
                if self._usesFeat:
                    flow, mask, feat = out
                else:
                    flow, mask = out
                if self.ensemble:
                    rev_parts = [img1[:, :3], img0[:, :3]]
                    if self._hasHead:
                        rev_parts += [self.f1, self.f0]
                    rev_parts.append(1 - timestep)
                    inpR = torch.cat(rev_parts, 1)
                    outR = blk(inpR, flow=None, scale=scale)
                    if self._usesFeat:
                        fR, mR, featR = outR
                        feat = (feat + featR) / 2
                    else:
                        fR, mR = outR
                    flow = (flow + torch.cat((fR[:, 2:4], fR[:, :2]), 1)) / 2
                    mask = (mask + (-mR)) / 2
            else:
                if lowResCat:
                    if self._hasHead:
                        a, b = ds(w0), ds(w1)
                        parts = [a[:, :3], b[:, :3], a[:, 3:], b[:, 3:]]
                    else:
                        parts = [ds(warped_img0[:, :3]), ds(warped_img1[:, :3])]
                    parts += [ds(timestep), ds(mask)]
                    if self._usesFeat:
                        parts.append(ds(feat))
                    parts.append(ds(flow) * inv)
                    out = blk.forwardPrescaled(torch.cat(parts, 1), scale)
                else:
                    inp_parts = [warped_img0[:, :3], warped_img1[:, :3]]
                    if self._hasHead:
                        inp_parts += [wf0, wf1]
                    inp_parts += [timestep, mask]
                    if self._usesFeat:
                        inp_parts.append(feat)
                    inp = torch.cat(inp_parts, 1)
                    out = blk(inp, flow=flow, scale=scale)
                if self._usesFeat:
                    fd, m0, feat = out
                else:
                    fd, m0 = out
                if self.ensemble:
                    rev_parts = [warped_img1[:, :3], warped_img0[:, :3]]
                    if self._hasHead:
                        rev_parts += [wf1, wf0]
                    rev_parts += [1 - timestep, -mask]
                    if self._usesFeat:
                        rev_parts.append(feat)
                    inpR = torch.cat(rev_parts, 1)
                    flowR = torch.cat((flow[:, 2:4], flow[:, :2]), 1)
                    outR = blk(inpR, flow=flowR, scale=scale)
                    if self._usesFeat:
                        fdR, mR, featR = outR
                        feat = (feat + featR) / 2
                    else:
                        fdR, mR = outR
                    fd = (fd + torch.cat((fdR[:, 2:4], fdR[:, :2]), 1)) / 2
                    m0 = (m0 + (-mR)) / 2

                flow = flow + fd
                if self._maskMode == "add":
                    mask = mask + m0
                else:
                    mask = m0

            flow_a = flow[:, :2]
            flow_b = flow[:, 2:4]
            # last block: wf0/wf1 are never read again -> warp only the images
            # (matches the baseline arch, which warps features at loop start)
            if self._hasHead and i < lastIdx:
                w0 = self._warp(imgf0, flow_a)
                w1 = self._warp(imgf1, flow_b)
                warped_img0, wf0 = w0[:, :3], w0[:, 3:]
                warped_img1, wf1 = w1[:, :3], w1[:, 3:]
            else:
                warped_img0 = self._warp(img0, flow_a)
                warped_img1 = self._warp(img1, flow_b)

        mask = torch.sigmoid(mask)
        # lerp(b, a, m) == a*m + b*(1-m): two elementwise passes instead of four
        return torch.lerp(warped_img1, warped_img0, mask)


# -----------------------------------------------------------------------------
# concrete variants
# -----------------------------------------------------------------------------


def _make(
    channels, inPlanesList, lastOutCh, hasHead, headMidC, headOutC, scaleBase, maskMode
):
    """Factory that returns a class binding the config to __init__ defaults."""
    cfg = dict(
        channels=channels,
        inPlanesList=inPlanesList,
        lastOutCh=lastOutCh,
        hasHead=hasHead,
        headMidC=headMidC,
        headOutC=headOutC,
        scaleBase=scaleBase,
        maskMode=maskMode,
    )

    class _Concrete(_FastIFNet):
        def __init__(
            self,
            ensemble=False,
            dynamicScale=False,
            scale=1,
            interpolateFactor=2,
            staticStep=False,
        ):
            super().__init__(
                **cfg,
                ensemble=ensemble,
                dynamicScale=dynamicScale,
                scale=scale,
                interpolateFactor=interpolateFactor,
                staticStep=staticStep,
            )

    return _Concrete


# rife4.6: 4 blocks, no Head, lastOutCh=6, additive mask
IFNet46 = _make(
    channels=[192, 128, 96, 64],
    inPlanesList=[7, 8 + 4, 8 + 4, 8 + 4],
    lastOutCh=6,
    hasHead=False,
    headMidC=0,
    headOutC=0,
    scaleBase=[8, 4, 2, 1],
    maskMode="add",
)

# rife4.15-lite: 4 blocks, Head(16,4), lastOutCh=6, no feat, replace mask
IFNet415Lite = _make(
    channels=[128, 96, 64, 48],
    inPlanesList=[7 + 8, 8 + 4 + 8, 8 + 4 + 8, 8 + 4 + 8],
    lastOutCh=6,
    hasHead=True,
    headMidC=16,
    headOutC=4,
    scaleBase=[8, 4, 2, 1],
    maskMode="replace",
)

# rife4.16-lite: same as 4.15-lite
IFNet416Lite = _make(
    channels=[128, 96, 64, 48],
    inPlanesList=[7 + 8, 8 + 4 + 8, 8 + 4 + 8, 8 + 4 + 8],
    lastOutCh=6,
    hasHead=True,
    headMidC=16,
    headOutC=4,
    scaleBase=[8, 4, 2, 1],
    maskMode="replace",
)

# rife4.17: 4 blocks, Head(32,8), lastOutCh=6, no feat
IFNet417 = _make(
    channels=[192, 128, 96, 64],
    inPlanesList=[7 + 16, 8 + 4 + 16, 8 + 4 + 16, 8 + 4 + 16],
    lastOutCh=6,
    hasHead=True,
    headMidC=32,
    headOutC=8,
    scaleBase=[8, 4, 2, 1],
    maskMode="replace",
)

# rife4.18: same as 4.17
IFNet418 = _make(
    channels=[192, 128, 96, 64],
    inPlanesList=[7 + 16, 8 + 4 + 16, 8 + 4 + 16, 8 + 4 + 16],
    lastOutCh=6,
    hasHead=True,
    headMidC=32,
    headOutC=8,
    scaleBase=[8, 4, 2, 1],
    maskMode="replace",
)

# rife4.20: 4 blocks, Head(32,8), block0 c=384, lastOutCh=6, no feat
IFNet420 = _make(
    channels=[384, 192, 96, 48],
    inPlanesList=[7 + 16, 8 + 4 + 16, 8 + 4 + 16, 8 + 4 + 16],
    lastOutCh=6,
    hasHead=True,
    headMidC=32,
    headOutC=8,
    scaleBase=[8, 4, 2, 1],
    maskMode="replace",
)

# rife4.21: 4 blocks, Head(32,8), c=[256,192,96,48], lastOutCh=13, feat used
IFNet421 = _make(
    channels=[256, 192, 96, 48],
    inPlanesList=[7 + 16, 8 + 4 + 16 + 8, 8 + 4 + 16 + 8, 8 + 4 + 16 + 8],
    lastOutCh=13,
    hasHead=True,
    headMidC=32,
    headOutC=8,
    scaleBase=[8, 4, 2, 1],
    maskMode="replace",
)

# rife4.22: same channel/in as 4.21
IFNet422 = _make(
    channels=[256, 192, 96, 48],
    inPlanesList=[7 + 16, 8 + 4 + 16 + 8, 8 + 4 + 16 + 8, 8 + 4 + 16 + 8],
    lastOutCh=13,
    hasHead=True,
    headMidC=32,
    headOutC=8,
    scaleBase=[8, 4, 2, 1],
    maskMode="replace",
)

# rife4.22-lite: 4 blocks, Head(16,4), c=[192,128,64,32], lastOutCh=13, feat used
IFNet422Lite = _make(
    channels=[192, 128, 64, 32],
    inPlanesList=[7 + 8, 8 + 4 + 8 + 8, 8 + 4 + 8 + 8, 8 + 4 + 8 + 8],
    lastOutCh=13,
    hasHead=True,
    headMidC=16,
    headOutC=4,
    scaleBase=[8, 4, 2, 1],
    maskMode="replace",
)

# rife4.25: 5 blocks, Head(16,4), c=[192,128,96,64,32], lastOutCh=13
IFNet425 = _make(
    channels=[192, 128, 96, 64, 32],
    inPlanesList=[7 + 8, 8 + 4 + 8 + 8, 8 + 4 + 8 + 8, 8 + 4 + 8 + 8, 8 + 4 + 8 + 8],
    lastOutCh=13,
    hasHead=True,
    headMidC=16,
    headOutC=4,
    scaleBase=[16, 8, 4, 2, 1],
    maskMode="replace",
)

# rife4.25-lite: 5 blocks, smaller last channel
IFNet425Lite = _make(
    channels=[192, 128, 96, 64, 24],
    inPlanesList=[7 + 8, 8 + 4 + 8 + 8, 8 + 4 + 8 + 8, 8 + 4 + 8 + 8, 8 + 4 + 8 + 8],
    lastOutCh=13,
    hasHead=True,
    headMidC=16,
    headOutC=4,
    scaleBase=[16, 8, 4, 2, 1],
    maskMode="replace",
)

# rife4.25-heavy: 5 blocks, channels x2
IFNet425Heavy = _make(
    channels=[384, 256, 192, 128, 64],
    inPlanesList=[7 + 8, 8 + 4 + 8 + 8, 8 + 4 + 8 + 8, 8 + 4 + 8 + 8, 8 + 4 + 8 + 8],
    lastOutCh=13,
    hasHead=True,
    headMidC=16,
    headOutC=4,
    scaleBase=[16, 8, 4, 2, 1],
    maskMode="replace",
)
