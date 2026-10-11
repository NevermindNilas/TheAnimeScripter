"""
The one RIFE 4.x implementation (4.6, 4.15-lite, 4.16-lite, 4.17, 4.18, 4.20,
4.21, 4.22, 4.22-lite, 4.25, 4.25-lite, 4.25-heavy, 4.26, 4.26-heavy).

- CUDA / ROCm / MPS, fp16 and fp32: the variant classes below (_FastIFNet),
  v1 API: cache(), cacheReset(frame), forward(img0, img1, timestep).
- TensorRT: exportArch(variant) (_ExportIFNet), the same blocks and rewrites
  lowered for ONNX; forward(img0, img1, timestep[, f0]).

Load-time transforms (original checkpoints load unchanged): ResConv beta folded
into the conv; ConvTranspose2d(4, 2, 1) tails repacked into Conv2d + a single
PixelShuffle(4); conv0 input channels permuted / zero-padded per the lowering
profile (see _FastIFNet). Forward rewrites -- block inputs concatenated at block
resolution in persistent channels_last buffers, timestep planes sliced, mask and
feature kept at block resolution between blocks, warps sampled only where the
next block's downsample reads -- are exact in real arithmetic; fp32 matches the
reference arches (IFNet_rife*) to rounding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dynamic_scale import pickScale

# -----------------------------------------------------------------------------
# shared modules
# -----------------------------------------------------------------------------


def _conv(
    inPlanes,
    outPlanes,
    kernelSize=3,
    stride=1,
    padding=1,
    dilation=1,
    paddingMode="zeros",
):
    return nn.Sequential(
        nn.Conv2d(
            inPlanes,
            outPlanes,
            kernel_size=kernelSize,
            stride=stride,
            padding=padding,
            padding_mode=paddingMode,
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


# -----------------------------------------------------------------------------
# depth-to-space
#
# After _repackDeconv a lastconv emits 16*C channels laid out for two nested
# PixelShuffle(2)s: channel (c*4 + 2*i2 + j2)*4 + 2*i1 + j1 lands on output pixel
# (4y + 2*i1 + i2, 4x + 2*j1 + j2). Permuting the conv's output channels at load
# to c*16 + 4a + b turns that into ONE PixelShuffle(4) (ONNX: one DepthToSpace)
# -- math identical, a conv's output channels are independent. One shuffle
# instead of two strided permute-copies on CUDA; one DepthToSpace instead of
# DepthToSpace+Shuffle+DepthToSpace in TensorRT. (A channel order giving a
# coalesced channels_last copy was measured -11..-15% end to end: the strided
# result slowed every consumer.)
# -----------------------------------------------------------------------------


def _d2sPerm(C):
    """perm[dst] = src: repacked 4x4/s2 deconv tail -> single PixelShuffle(4)."""
    perm = [0] * (16 * C)
    for c in range(C):
        for a in range(4):
            for b in range(4):
                i1, i2 = divmod(a, 2)
                j1, j2 = divmod(b, 2)
                perm[c * 16 + 4 * a + b] = (c * 4 + 2 * i2 + j2) * 4 + 2 * i1 + j1
    return perm


def _clView(t):
    """A 1-channel NCHW-contiguous plane re-strided as channels_last (same memory).

    torch.cat emits NCHW as soon as one input does not suggest channels_last, and
    a [N,1,H,W] NCHW tensor never does -- the timestep plane was turning every
    block-input concat NCHW, which cuDNN then copied back to channels_last."""
    if t.dim() == 4 and t.size(1) == 1 and t.is_contiguous():
        n, _, h, w = t.shape
        return t.as_strided((n, 1, h, w), (h * w, 1, w, 1))
    return t


def _isPow2Int(s):
    return s >= 1 and s == int(s) and (int(s) & (int(s) - 1)) == 0


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
    """IFBlock with deconv->Conv2d+depth-to-space rewrite. lastOutCh: 6 or 13."""

    def __init__(self, inPlanes, c=64, lastOutCh=13, convPad="zeros"):
        super().__init__()
        self.conv0 = nn.Sequential(
            _conv(inPlanes, c // 2, 3, 2, 1, paddingMode=convPad),
            _conv(c // 2, c, 3, 2, 1, paddingMode=convPad),
        )
        self.convblock = nn.Sequential(*[_ResConv(c) for _ in range(8)])
        # baseline: ConvTranspose2d(c, 4*lastOutCh, 4, 2, 1) + PixelShuffle(2) -> lastOutCh ch at 4x
        # fast: Conv2d(c, 16*lastOutCh, 3, 1, 1) + one depth-to-space(4) (see _d2sPerm)
        self.lastconv = nn.Conv2d(c, 4 * (4 * lastOutCh), 3, 1, 1)
        self.lastOutCh = lastOutCh
        self.inPlanes = inPlanes
        # input channels after zero-padding (see _FastIFNet._applyLowering)
        self.paddedIn = inPlanes
        # flow scale already folded into lastconv's flow rows at load
        self.flowScale = 1.0

    def padInput(self, x):
        if x.size(1) < self.paddedIn:
            z = x.new_zeros(
                (x.size(0), self.paddedIn - x.size(1), x.size(2), x.size(3))
            )
            x = torch.cat((x, z), 1)
        return x

    def core(self, x):
        """Block-resolution input (already padded) -> lastOutCh channels at block res."""
        feat = self.convblock(self.conv0(x))
        if isinstance(self.lastconv, nn.ConvTranspose2d):
            return F.pixel_shuffle(self.lastconv(feat), 2)
        return F.pixel_shuffle(self.lastconv(feat), 4)

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
        tmp = self.core(self.padInput(x))
        if scale != 1:
            tmp = F.interpolate(
                tmp, scale_factor=scale, mode="bilinear", align_corners=False
            )
        flow_out = tmp[:, :4]
        if scale != self.flowScale:
            flow_out = flow_out * (scale / self.flowScale)
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
        x3 = self.cnn3(x)
        if not isinstance(self.cnn3, nn.ConvTranspose2d):
            x3 = F.pixel_shuffle(x3, 2)
        if feat:
            return [x0, x1, x2, x3]
        return x3


class _HeadLite(nn.Module):
    """rife_elexor's encoder: Conv2d(3, 16, s2) -> ConvTranspose2d(16, outC, 4, 2, 1),
    no activation; the deconv is repacked into Conv2d + PixelShuffle(2) like
    _Head.cnn3. Checkpoint keys encode.0.* / encode.1.* map to cnn0 / cnn3."""

    def __init__(self, midC=16, outC=4):
        super().__init__()
        self.cnn0 = nn.Conv2d(3, midC, 3, 2, 1)
        self.cnn3 = nn.Conv2d(midC, 4 * outC, 3, 1, 1)

    def forward(self, x):
        x = self.cnn3(self.cnn0(x))
        if not isinstance(self.cnn3, nn.ConvTranspose2d):
            x = F.pixel_shuffle(x, 2)
        return x


# -----------------------------------------------------------------------------
# generic fast IFNet, parameterized by config
# -----------------------------------------------------------------------------


class _FastIFNet(nn.Module):
    """Generic fast IFNet supporting all RIFE 4.x variants.

    Lowering profile (class attributes, applied at load time / in forward):
        alignIn:   zero-pad every block's input channels to this multiple (zero
                   weight rows), so cuDNN's NHWC fp16 kernels run without their
                   own per-call nhwcAddPadding pass. 1 = off.
        lowResTail: keep a block's mask/feature outputs at block resolution and
                   resample them straight to the next block's resolution instead
                   of upsampling to full res and downsampling again (bilinear,
                   align_corners=False, power-of-two integer factors: the two are
                   equal in exact arithmetic, see _forwardFast).
        sparseWarp: for a next block at scale >= 4, warp only the 2x2 taps its
                   bilinear downsample reads (bit-identical).
        deconvTail: keep the checkpoint's ConvTranspose2d tails (no repack).
        foldFlowScale: fold each block's flow "* scale" into lastconv's flow
                   rows (weight + bias) at load. Scales are powers of two, so this
                   is bit-exact; off under dynamicScale (scale changes per pair).
        sideMajor: block inputs laid out [img0|f0, img1|f1, ...] instead of the
                   checkpoint's [img0, img1, f0, f1, ...] (conv0 input channels
                   permuted at load), so each warped side lands in a block's input
                   with one copy instead of two.

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

    alignIn = 8
    lowResTail = True
    sparseWarp = True
    deconvTail = False
    sideMajor = True
    foldFlowScale = True

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
        encoder="head",
        convPad="zeros",
        schedule=None,
        resetAfter=None,
        rescueAt=None,
        rescueMag=40,
        rescueCount=1036800,
        ensemble=False,
        dynamicScale=False,
        scale=1,
        interpolateFactor=2,
        staticStep=False,
    ):
        super().__init__()
        assert len(channels) == len(inPlanesList)
        # pass i runs blocks[schedule[i]] at scale_list[i]; every variant but
        # rife_elexor runs each block once, coarse to fine
        self.schedule = (
            list(schedule) if schedule is not None else list(range(len(channels)))
        )
        assert len(self.schedule) == len(scaleBase)
        # rife_elexor restarts the pyramid after pass resetAfter and, after pass
        # rescueAt, keeps the first pyramid's flow when more than rescueCount
        # pixels of it move by more than rescueMag px (side 0)
        self.resetAfter = resetAfter
        self.rescueAt = rescueAt
        self.rescueMag = rescueMag
        self.rescueCount = rescueCount
        if schedule is not None:
            dynamicScale = False  # no dynamic-scale path for scheduled variants
        self._hasHead = hasHead
        self._usesFeat = lastOutCh >= 13
        self._maskMode = maskMode
        self._headOutC = headOutC if hasHead else 0
        self._encoderType = encoder

        for i, (cin, c) in enumerate(zip(inPlanesList, channels, strict=False)):
            setattr(
                self,
                f"block{i}",
                _IFBlock(cin, c=c, lastOutCh=lastOutCh, convPad=convPad),
            )
        self.blocks = [getattr(self, f"block{i}") for i in range(len(channels))]

        if hasHead and encoder == "lite":
            self.encode = _HeadLite(midC=headMidC, outC=headOutC)
        elif hasHead:
            self.encode = _Head(midC=headMidC, outC=headOutC)
        else:
            self.encode = None

        # kept so forward() can rebuild scale_list when dynamicScale re-picks the
        # scale per frame; scale_list itself is overwritten in that case.
        self._scaleBase = tuple(scaleBase)
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
        self._lowered = False
        # persistent channels_last block-input buffers, keyed by shape; their
        # zero pad channels are written once at allocation
        self._inBufs = {}
        self._latticeGrids = {}

    # ---- weight loading / repack -------------------------------------------

    def load_state_dict(self, stateDict, strict=True):
        remapped = {}
        for k, v in stateDict.items():
            nk = k
            if self._encoderType == "lite":
                if k.startswith("encode.0."):
                    nk = k.replace("encode.0.", "encode.cnn0.", 1)
                elif k.startswith("encode.1."):
                    nk = k.replace("encode.1.", "encode.cnn3.", 1)
                k = nk
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
        self._applyLowering()
        return type("R", (), {"missing_keys": missing, "unexpected_keys": unexpected})()

    @torch.no_grad()
    def _applyLowering(self):
        """Load-time weight transforms of the lowering profile (once per load)."""
        if self._lowered:
            return
        for i, blk in enumerate(self.blocks):
            if not isinstance(blk.lastconv, nn.ConvTranspose2d):
                perm = torch.tensor(_d2sPerm(blk.lastOutCh), dtype=torch.long)
                blk.lastconv.weight.data = blk.lastconv.weight.data[perm].contiguous()
                blk.lastconv.bias.data = blk.lastconv.bias.data[perm].contiguous()
            conv = blk.conv0[0][0]
            e = self._headOutC
            if self.sideMajor and e:
                perm = (
                    [0, 1, 2]
                    + list(range(6, 6 + e))
                    + [3, 4, 5]
                    + list(range(6 + e, 6 + 2 * e))
                    + list(range(6 + 2 * e, conv.in_channels))
                )
                conv.weight.data = conv.weight.data[:, perm].contiguous()
            pad = (-conv.in_channels) % self.alignIn
            if pad:
                w = conv.weight.data
                conv.weight.data = torch.cat(
                    (w, w.new_zeros((w.size(0), pad, *w.shape[2:]))), 1
                ).contiguous()
                conv.in_channels += pad
            blk.paddedIn = conv.in_channels
            uses = [j for j, b in enumerate(self.schedule) if b == i]
            s = float(self.scale_list[uses[0]]) if len(uses) == 1 else 1.0
            if (
                self.foldFlowScale
                and not self.dynamicScale
                and not isinstance(blk.lastconv, nn.ConvTranspose2d)
                and s != 1
            ):
                # after _d2sPerm, output channel c*16 + k lands on channel c of
                # the shuffled tail: rows 0..63 are the 4 flow channels
                blk.lastconv.weight.data[:64] *= s
                blk.lastconv.bias.data[:64] *= s
                blk.flowScale = s
        self._lowered = True

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
            if self.deconvTail:
                # keep the checkpoint's ConvTranspose2d(4, 2, 1) as-is
                deconv = nn.ConvTranspose2d(W.size(0), W.size(1), 4, 2, 1)
                with torch.no_grad():
                    deconv.weight.data.copy_(W)
                    deconv.bias.data.copy_(b)
                if pfx == "encode.cnn3":
                    self.encode.cnn3 = deconv
                else:
                    getattr(self, pfx.split(".")[0]).lastconv = deconv
                continue
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
        self.prepare(img0, img1)
        return self.flow(img0, img1, timestep)

    def prepare(self, img0, img1):
        """Per-pair work: refresh the encoder features and stage the warp sources.

        Everything here depends only on the frame pair, so a caller emitting
        several frames per pair (--interpolate_factor > 2, motion blur, wide
        --smooth_dedup gaps) runs it once and then flow() once per timestep."""
        self._refreshFeatures(img0, img1)
        if self._hasHead:
            src0, src1 = self._srcBufs(img0)
            src0[:, :3].copy_(img0[:, :3])
            src0[:, 3:].copy_(self.f0)
            src1[:, :3].copy_(img1[:, :3])
            src1[:, 3:].copy_(self.f1)

    def _srcBufs(self, ref):
        n, _, h, w = ref.shape
        key = ("src", n, h, w, ref.dtype, ref.device)
        bufs = self._inBufs.get(key)
        if bufs is None:
            with torch.inference_mode(False):
                bufs = tuple(
                    torch.zeros(
                        (n, 3 + self._headOutC, h, w),
                        dtype=ref.dtype,
                        device=ref.device,
                    ).contiguous(memory_format=torch.channels_last)
                    for _ in range(2)
                )
            self._inBufs[key] = bufs
        return bufs

    def flow(self, img0, img1, timestep):
        """Per-timestep work; prepare() must have run for this pair."""
        ph, pw = img0.shape[-2], img0.shape[-1]
        self._ensureGrid(ph, pw, img0.dtype, img0.device)

        if self.dynamicScale:
            # Same contract as the reference arches: the per-pair scale replaces
            # the constructor's, so scale_list is rebuilt from scaleBase. Block
            # resolutions therefore change with the pick, so a caller that wants
            # CUDA graphs needs one capture per DYNAMIC_SCALES entry (RifeCuda
            # does) and must pad for the coarsest pick -- see
            # src/interpolate/_padding.py.
            s = pickScale(self, img0[:, :3], img1[:, :3])
            self.scale_list = [b / s for b in self._scaleBase]

        if not self.ensemble and all(_isPow2Int(s) for s in self.scale_list):
            return self._forwardFast(img0, img1, timestep)
        if self.resetAfter is not None or len(self.schedule) != len(self.blocks):
            raise NotImplementedError(
                "scheduled variants (rife_elexor) run only on the fast path; "
                "ensemble uses the reference arch"
            )
        return self._forwardGeneric(img0, img1, timestep)

    def _refreshFeatures(self, img0, img1):
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

    def _pair(self, i0, i1, f0=None, f1=None):
        """The two frames' image and feature parts in the blocks' channel order."""
        if f0 is None:
            return [i0, i1]
        return [i0, f0, i1, f1] if self.sideMajor else [i0, i1, f0, f1]

    def _sides(self, a, b):
        """Per-side [img|feat] tensors -> parts in the blocks' channel order."""
        if not self._hasHead or self.sideMajor:
            return [a, b] if self._hasHead else [a[:, :3], b[:, :3]]
        return [a[:, :3], b[:, :3], a[:, 3:], b[:, 3:]]

    # ---- fast path (non-ensemble, power-of-two integer block scales) ---------

    def _inBuf(self, i, c, h, w, ref):
        key = (i, ref.size(0), c, h, w, ref.dtype, ref.device)
        buf = self._inBufs.get(key)
        if buf is None:
            # A normal (non-inference) tensor so callers may mix inference_mode
            # and no_grad; zero pad channels stay zero (never written).
            with torch.inference_mode(False):
                buf = torch.zeros(
                    (ref.size(0), c, h, w), dtype=ref.dtype, device=ref.device
                ).contiguous(memory_format=torch.channels_last)
            self._inBufs[key] = buf
        return buf

    def _fill(self, i, blk, parts, h, w, ref):
        """Write parts side by side into block i's persistent channels_last input.

        One pass per part instead of cat (which goes NCHW whenever a part --
        grid_sample output, the timestep plane -- does) followed by cuDNN's
        NCHW->NHWC copy and its channel-padding copy."""
        buf = self._inBuf(i, blk.paddedIn, h, w, ref)
        o = 0
        for p in parts:
            k = p.size(1)
            buf[:, o : o + k].copy_(p)
            o += k
        assert o == blk.inPlanes, (o, blk.inPlanes)
        return buf

    def _lattice(self, t, s):
        """The 2x2 taps a bilinear 1/s downsample (align_corners=False, integer
        s >= 2) reads, as a 2x-block-res tensor: rows/cols s*j + s/2 - 1 + {0,1}."""
        n, c, h, w = t.shape
        o = s // 2 - 1
        t = t.unflatten(2, (h // s, s)).unflatten(4, (w // s, s))
        return t[:, :, :, o : o + 2, :, o : o + 2].reshape(n, c, 2 * h // s, 2 * w // s)

    def _warpDs(self, inp, flow2, s):
        """== downsample(warp(inp, flow2), 1/s), sampling only the taps it reads."""
        key = (s, self._gridShape)
        bw = self._latticeGrids.get(key)
        if bw is None:
            bw = self._lattice(self._backWarp, s).contiguous()
            self._latticeGrids[key] = bw
        grid = torch.addcmul(bw, self._lattice(flow2, s), self._tenFlow)
        w = F.grid_sample(
            inp,
            grid.permute(0, 2, 3, 1),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return F.interpolate(w, scale_factor=0.5, mode="bilinear", align_corners=False)

    def _forwardFast(self, img0, img1, timestep):
        """Same maths as _forwardGeneric, restructured for memory traffic.

        Exact rewrites (bit-identical): timestep planes taken as strided slices
        (the buffer is a constant fill, so any sampling of it is that constant);
        block inputs written straight into persistent channels_last buffers;
        upsampling only the channels each consumer needs (bilinear is
        per-channel); sparseWarp.

        lowResTail is exact in real arithmetic but rounds once instead of twice:
        the next block (scale s' >= 2) reads downsample_{s'}(upsample_s(x)), and
        for bilinear/align_corners=False with s/s' an even power of two the two
        taps of every downsample sit on one linear segment of the upsample, so
        the composite equals upsample_{s/s'}(x) -- including at the clamped
        borders."""
        n, _, ph, pw = img0.shape
        ts = _clView(timestep)
        hasHead = self._hasHead
        if hasHead:
            src0, src1 = self._srcBufs(img0)  # staged by prepare()
        else:
            src0, src1 = img0[:, :3], img1[:, :3]
        addMask = self._maskMode == "add"
        lowRes = self.lowResTail and not addMask
        npass = len(self.schedule)
        scales = [int(s) for s in self.scale_list]

        def ds(t, s):
            return F.interpolate(
                t, scale_factor=1.0 / s, mode="bilinear", align_corners=False
            )

        def up(t, r):
            return F.interpolate(
                t, scale_factor=float(r), mode="bilinear", align_corners=False
            )

        flow = mask = feat = None  # flow full-res; mask|feat low-res when maskLow
        maskLow = False
        largeFlow = None
        w0 = w1 = None
        prevS = None
        for i, bi in enumerate(self.schedule):
            blk = self.blocks[bi]
            s = scales[i]
            h, w = ph // s, pw // s
            tsI = ts if s == 1 else ts[:, :, ::s, ::s]
            if flow is None:  # first pass of a pyramid
                a, b = (src0, src1) if s == 1 else (ds(src0, s), ds(src1, s))
                parts = self._sides(a, b)
                parts.append(tsI)
            else:
                if s == 1:
                    a, b = w0, w1
                elif w0 is None:  # sparse
                    a = self._warpDs(src0, flow[:, :2], s)
                    b = self._warpDs(src1, flow[:, 2:4], s)
                else:
                    a, b = ds(w0, s), ds(w1, s)
                parts = self._sides(a, b)
                parts.append(tsI)
                if maskLow:
                    # mask and feat are adjacent in both the block output and
                    # the block input: one resample for both
                    parts.append(up(mask, prevS // s))
                else:
                    parts.append(mask if s == 1 else ds(mask, s))
                    if self._usesFeat:
                        parts.append(feat if s == 1 else ds(feat, s))
                parts.append(flow if s == 1 else ds(flow, s) * (1.0 / s))
            tmp = blk.core(self._fill(i, blk, parts, h, w, img0))

            last = i == npass - 1
            # The tail is back at block resolution unless the block resolution
            # was not a multiple of 4 (rife_elexor's 1/32 pass at 1088 rows:
            # 34 -> 17 -> 9 -> 36); then it is resized straight to full res, as
            # the reference does, and the low-res tail cannot apply.
            exact = tmp.shape[-2] == h and tmp.shape[-1] == w
            lowThis = lowRes and exact and not last
            # flow (+ mask when it accumulates / is needed at full res) to full res
            fullC = 5 if (addMask or last or not lowThis) else 4
            if not lowThis and not last and self._usesFeat:
                fullC = tmp.size(1)
            if not exact:
                t = F.interpolate(
                    tmp[:, :fullC], size=(ph, pw), mode="bilinear", align_corners=False
                )
            else:
                t = tmp[:, :fullC] if s == 1 else up(tmp[:, :fullC], s)
            r = s / blk.flowScale
            fd = t[:, :4] if r == 1 else t[:, :4] * r
            flow = fd if flow is None else flow + fd
            if addMask:
                m0 = t[:, 4:5]
                mask = m0 if mask is None else mask + m0
            elif lowThis:
                mask = tmp[:, 4:] if self._usesFeat else tmp[:, 4:5]  # mask|feat
            else:
                mask = t[:, 4:5]
                feat = t[:, 5:] if self._usesFeat and not last else None
            maskLow = lowThis
            prevS = s
            if i == self.rescueAt and largeFlow is not None:
                mag = torch.sqrt(largeFlow[:, 0] ** 2 + largeFlow[:, 1] ** 2)
                flow = torch.where(
                    torch.sum(mag > self.rescueMag) > self.rescueCount,
                    largeFlow,
                    flow,
                )

            if last:
                break
            if i == self.resetAfter:  # restart the pyramid; no warps needed
                largeFlow, flow = flow, None
                continue
            nextS = scales[i + 1]
            if self.sparseWarp and nextS >= 4:
                w0 = w1 = None
            else:
                w0 = self._warp(src0, flow[:, :2])
                w1 = self._warp(src1, flow[:, 2:4])

        wi0 = self._warp(img0[:, :3], flow[:, :2])
        wi1 = self._warp(img1[:, :3], flow[:, 2:4])
        return torch.lerp(wi1, wi0, torch.sigmoid(mask))

    # ---- generic path (ensemble, non-power-of-two dynamic scales) ------------

    def _forwardGeneric(self, img0, img1, timestep):
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
                # B023: `inv` is read immediately within this iteration, never deferred.
                return F.interpolate(
                    t,
                    scale_factor=inv,  # noqa: B023
                    mode="bilinear",
                    align_corners=False,
                )

            if flow is None:
                if lowResCat:
                    if self._hasHead:
                        a, b = ds(imgf0), ds(imgf1)
                        parts = self._sides(a, b) + [ds(timestep)]
                    else:
                        parts = [ds(img0[:, :3]), ds(img1[:, :3]), ds(timestep)]
                    out = blk.forwardPrescaled(torch.cat(parts, 1), scale)
                else:
                    inp_parts = self._pair(img0[:, :3], img1[:, :3], self.f0, self.f1)
                    inp_parts.append(timestep)
                    inp = torch.cat(inp_parts, 1)
                    out = blk(inp, flow=None, scale=scale)
                if self._usesFeat:
                    flow, mask, feat = out
                else:
                    flow, mask = out
                if self.ensemble:
                    rev_parts = self._pair(img1[:, :3], img0[:, :3], self.f1, self.f0)
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
                        # w0/w1 are loop-carried from the prior block iteration (set below);
                        # iteration 0 takes the `flow is None` branch and never reaches here.
                        a, b = ds(w0), ds(w1)  # noqa: F821
                        parts = self._sides(a, b)
                    else:
                        parts = [ds(warped_img0[:, :3]), ds(warped_img1[:, :3])]
                    parts += [ds(timestep), ds(mask)]
                    if self._usesFeat:
                        parts.append(ds(feat))
                    parts.append(ds(flow) * inv)
                    out = blk.forwardPrescaled(torch.cat(parts, 1), scale)
                else:
                    inp_parts = self._pair(
                        warped_img0[:, :3], warped_img1[:, :3], wf0, wf1
                    )
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
                    rev_parts = self._pair(
                        warped_img1[:, :3], warped_img0[:, :3], wf1, wf0
                    )
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
    channels,
    inPlanesList,
    lastOutCh,
    hasHead,
    headMidC,
    headOutC,
    scaleBase,
    maskMode,
    **extra,
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
        **extra,
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

    _Concrete.cfg = cfg
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

# rife4.26: same architecture as rife4.25, new weights
IFNet426 = IFNet425

# rife4.26-heavy: rife4.25 block widths, but a 16-channel Head (4.25-heavy
# widened the blocks instead), so every block takes 2*12 more feature channels
IFNet426Heavy = _make(
    channels=[192, 128, 96, 64, 32],
    inPlanesList=[
        7 + 32,
        8 + 4 + 32 + 8,
        8 + 4 + 32 + 8,
        8 + 4 + 32 + 8,
        8 + 4 + 32 + 8,
    ],
    lastOutCh=13,
    hasHead=True,
    headMidC=16,
    headOutC=16,
    scaleBase=[16, 8, 4, 2, 1],
    maskMode="replace",
)


# rife_elexor: 4.7-style blocks and 4-ch features, 6 passes over 4 blocks
# ([b0, b1] at 32/16, then the pyramid restarts: [b0, b1, b2, b3] at 8..1) with a
# large-motion rescue after the restart; replicate-padded conv0; two-layer encoder
IFNetElexor = _make(
    channels=[192, 128, 96, 64],
    inPlanesList=[7 + 8, 8 + 4 + 8, 8 + 4 + 8, 8 + 4 + 8],
    lastOutCh=6,
    hasHead=True,
    headMidC=16,
    headOutC=4,
    scaleBase=[32, 16, 8, 4, 2, 1],
    maskMode="replace",
    encoder="lite",
    convPad="replicate",
    schedule=[0, 1, 0, 1, 2, 3],
    resetAfter=1,
    rescueAt=2,
)


# -----------------------------------------------------------------------------
# export profile (TensorRT / DirectML / OpenVINO ONNX)
#
# Same variant table, same blocks, same load-time transforms and the same
# rewrites as the CUDA path (single DepthToSpace tail, side-major block inputs,
# timestep slices, low-res mask/feature tail, sparse warps, flow scale folded
# into the tail weights); only the lowering
# that is specific to an executor differs:
#   - both frames are warped by ONE grid_sample on a batch-2 stack, and that stack
#     folded into channels IS the side-major block input;
#   - alignIn stays 1: the zero channels exist to skip cuDNN's own NHWC padding
#     pass, and TensorRT measured slower with padded block inputs;
#   - warpPad zero-pads the warp input to a multiple of warpPad channels.
#     TensorRT 10.16 miscompiled a 7-channel GridSample -> Resize in one older
#     graph shape (34 dB vs eager); the current graph is valid unpadded on 10.16
#     and 11.3, and the padding costs ~13.5% on 11.3 (see _shared._tensorrtProfile).
# -----------------------------------------------------------------------------


class _ExportIFNet(_FastIFNet):
    alignIn = 1
    lowResTail = True
    sparseWarp = True
    deconvTail = False
    warpPad = 1
    # True: forward(img0, img1, timestep, f0) -> (output, f1), encoder run on
    # img1 only (TensorRT keeps f0 outside the engine). False: forward(img0,
    # img1, timestep) -> output, both frames encoded inside (DirectML/OpenVINO).
    featureIO = True

    def __init__(
        self,
        cfg,
        scale=1.0,
        ensemble=False,
        dtype=torch.float32,
        device="cuda",
        width=1920,
        height=1080,
        decomposedWarp=False,
    ):
        super().__init__(
            **cfg,
            ensemble=ensemble,
            dynamicScale=False,
            scale=scale,
            interpolateFactor=2,
        )
        self.width = width
        self.height = height
        self.dtype = dtype
        self.device = device
        self.encodeChannels = cfg["headOutC"] if cfg["hasHead"] else 0
        if decomposedWarp:
            from .grid_sample_directml import grid_sample_directml

            self._gsFn = grid_sample_directml
        else:
            self._gsFn = None

    def _gs(self, inp, grid):
        if self._gsFn is not None:
            return self._gsFn(inp, grid, padding_mode="border", align_corners=True)
        return F.grid_sample(
            inp, grid, mode="bilinear", padding_mode="border", align_corners=True
        )

    def _grid2(self, flow, ph, pw):
        return (self._backWarp + flow.reshape(2, 2, ph, pw) * self._tenFlow).permute(
            0, 2, 3, 1
        )

    def _latticeGrid2(self, flow, s):
        key = (s, self._gridShape)
        bw = self._latticeGrids.get(key)
        if bw is None:
            bw = self._lattice(self._backWarp, s).contiguous()
            self._latticeGrids[key] = bw
        f = self._lattice(flow, s)
        h2, w2 = f.shape[-2], f.shape[-1]
        return (bw + f.reshape(2, 2, h2, w2) * self._tenFlow).permute(0, 2, 3, 1)

    def _src2(self, img0, img1, f0, f1):
        """Batch-2 warp source [img|feat] for both frames, channel-padded."""
        imgs2 = torch.cat((img0[:, :3], img1[:, :3]), 0)
        src2 = torch.cat((imgs2, torch.cat((f0, f1), 0)), 1) if self._hasHead else imgs2
        padC = (-src2.size(1)) % self.warpPad
        if padC:
            src2 = torch.cat((src2, imgs2[:, :1].expand(-1, padC, -1, -1) * 0), 1)
        return src2

    def forward(self, img0, img1, timestep, f0=None):
        """Single-graph form: pairForward + stepForward."""
        if self._hasHead:
            src2, f1 = self.pairForward(img0, img1, f0)
        else:
            src2, f1 = self.pairForward(img0, img1), None
        out = self.stepForward(src2, timestep)
        if self.featureIO and self._hasHead:
            return out, f1
        return out

    def pairForward(self, img0, img1, f0=None):
        """Per-pair graph: encode img1 and build the batch-2 warp source.

        -> (src2, f1) for head-bearing variants, src2 alone for headless ones."""
        if not self._hasHead:
            return self._src2(img0, img1, None, None)
        f1 = self.encode(img1[:, :3])
        if f0 is None:
            f0 = self.encode(img0[:, :3])
        return self._src2(img0, img1, f0, f1), f1

    def stepForward(self, src2, timestep):
        """Per-timestep graph: flow blocks + warps + blend, output cropped."""
        ph, pw = src2.shape[-2], src2.shape[-1]
        self._ensureGrid(ph, pw, src2.dtype, src2.device)
        if self.ensemble:
            out = self._forwardEnsemble(src2, timestep, ph, pw)
        else:
            out = self._forwardExport(src2, timestep, ph, pw)
        return out[:, :, : self.height, : self.width]

    def _forwardExport(self, src2, timestep, ph, pw):
        hasHead = self._hasHead
        e = self.encodeChannels
        C = 3 + e
        imgs2 = src2[:, :3]
        addMask = self._maskMode == "add"
        lowRes = self.lowResTail and not addMask
        npass = len(self.schedule)
        scales = [int(s) for s in self.scale_list]

        def ds(t, s):
            return F.interpolate(
                t, scale_factor=1.0 / s, mode="bilinear", align_corners=False
            )

        def up(t, r):
            return F.interpolate(
                t, scale_factor=float(r), mode="bilinear", align_corners=False
            )

        def split2(t):
            """[2,C(+pad),h,w] warp/source stack -> the blocks' image+feature parts.

            Side-major is just the batch folded into channels ([img0|f0, img1|f1])."""
            h, w = t.shape[-2], t.shape[-1]
            if self.sideMajor or not hasHead:
                return [t[:, :C].reshape(1, 2 * C, h, w)]
            return [
                t[:, :3].reshape(1, 6, h, w),
                t[:, 3:C].reshape(1, 2 * e, h, w),
            ]

        flow = mask = feat = None
        maskLow = False
        largeFlow = None
        warps = None
        prevS = None
        for i, bi in enumerate(self.schedule):
            blk = self.blocks[bi]
            s = scales[i]
            h, w = ph // s, pw // s
            tsI = timestep if s == 1 else timestep[:, :, ::s, ::s]
            if flow is None:  # first pass of a pyramid
                parts = split2(src2[:, :C] if s == 1 else ds(src2[:, :C], s))
                parts.append(tsI)
            else:
                if warps is None:  # sparse: warp only the taps the downsample reads
                    wl = self._gs(src2, self._latticeGrid2(flow, s))
                    parts = split2(ds(wl[:, :C], 2))
                else:
                    parts = split2(warps[:, :C] if s == 1 else ds(warps[:, :C], s))
                parts.append(tsI)
                if maskLow:
                    # mask|feat are adjacent in block output and input alike
                    parts.append(up(mask, prevS // s))
                else:
                    parts.append(mask if s == 1 else ds(mask, s))
                    if self._usesFeat:
                        parts.append(feat if s == 1 else ds(feat, s))
                parts.append(flow if s == 1 else ds(flow * (1.0 / s), s))
            tmp = blk.core(blk.padInput(torch.cat(parts, 1)))

            last = i == npass - 1
            exact = tmp.shape[-2] == h and tmp.shape[-1] == w  # see _forwardFast
            lowThis = lowRes and exact and not last
            fullC = 5 if (addMask or last or not lowThis) else 4
            if not lowThis and not last and self._usesFeat:
                fullC = tmp.size(1)
            if not exact:
                t = F.interpolate(
                    tmp[:, :fullC], size=(ph, pw), mode="bilinear", align_corners=False
                )
            else:
                t = tmp[:, :fullC] if s == 1 else up(tmp[:, :fullC], s)
            r = s / blk.flowScale
            fd = t[:, :4] if r == 1 else t[:, :4] * r
            flow = fd if flow is None else flow + fd
            if addMask:
                m0 = t[:, 4:5]
                mask = m0 if mask is None else mask + m0
            elif lowThis:
                mask = tmp[:, 4:] if self._usesFeat else tmp[:, 4:5]  # mask|feat
            else:
                mask = t[:, 4:5]
                feat = t[:, 5:] if self._usesFeat and not last else None
            maskLow = lowThis
            prevS = s
            if i == self.rescueAt and largeFlow is not None:
                mag = torch.sqrt(largeFlow[:, 0] ** 2 + largeFlow[:, 1] ** 2)
                flow = torch.where(
                    torch.sum(mag > self.rescueMag) > self.rescueCount,
                    largeFlow,
                    flow,
                )
            if last:
                break
            if i == self.resetAfter:  # restart the pyramid; no warps needed
                largeFlow, flow = flow, None
                continue
            if self.sparseWarp and scales[i + 1] >= 4:
                warps = None
            else:
                warps = self._gs(src2, self._grid2(flow, ph, pw))

        wi = self._gs(imgs2, self._grid2(flow, ph, pw))
        mask = torch.sigmoid(mask)
        return wi[0:1] * mask + wi[1:2] * (1 - mask)

    def _block(self, blk, x, s):
        """Full-resolution block input -> (flow, mask, feat) at full res."""
        if s != 1:
            x = F.interpolate(
                x, scale_factor=1.0 / s, mode="bilinear", align_corners=False
            )
        tmp = blk.core(blk.padInput(x))
        if s != 1:
            tmp = F.interpolate(
                tmp, scale_factor=s, mode="bilinear", align_corners=False
            )
        r = s / blk.flowScale
        flow = tmp[:, :4] * r if r != 1 else tmp[:, :4]
        feat = tmp[:, 5:] if self._usesFeat else None
        return flow, tmp[:, 4:5], feat

    def _forwardEnsemble(self, src2, timestep, ph, pw):
        """Forward + reversed pass, averaged (reference maths)."""
        if self.resetAfter is not None or len(self.schedule) != len(self.blocks):
            raise NotImplementedError(
                "scheduled variants (rife_elexor) export without ensemble only"
            )
        hasHead = self._hasHead
        e = self.encodeChannels
        C = 3 + e
        imgs2 = src2[:, :3]
        img0, img1 = src2[0:1, :3], src2[1:2, :3]
        f0, f1 = (src2[0:1, 3:C], src2[1:2, 3:C]) if hasHead else (None, None)
        addMask = self._maskMode == "add"

        def swap(t, k):  # swap the two k-channel halves
            return torch.cat((t[:, k : 2 * k], t[:, :k]), 1)

        flow = mask = feat = None
        warps = None
        for i, blk in enumerate(self.blocks):
            s = self.scale_list[i]
            if i == 0:
                fwd = self._pair(img0[:, :3], img1[:, :3], f0, f1) + [timestep]
                rev = self._pair(img1[:, :3], img0[:, :3], f1, f0) + [1 - timestep]
            else:
                h, w = warps.shape[-2], warps.shape[-1]
                if self.sideMajor or not hasHead:
                    side = warps[:, :C].reshape(1, 2 * C, h, w)
                    fwd = [side]
                    rev = [swap(side, C)]
                else:
                    wimg = warps[:, :3].reshape(1, 6, h, w)
                    wf = warps[:, 3:C].reshape(1, 2 * e, h, w)
                    fwd = [wimg, wf]
                    rev = [swap(wimg, 3), swap(wf, e)]
                fwd += [timestep, mask]
                rev += [1 - timestep, -mask]
                if self._usesFeat:
                    fwd.append(feat)
                    rev.append(feat)
                fwd.append(flow * (1.0 / s) if s != 1 else flow)
                rflow = swap(flow, 2)
                rev.append(rflow * (1.0 / s) if s != 1 else rflow)
            # The reversed pass reads the forward pass's fresh feature output
            # (reference arches and the CUDA path do the same), so the two
            # passes run one after the other rather than as one batch of 2.
            flF, mF, ftF = self._block(blk, torch.cat(fwd, 1), s)
            if self._usesFeat and i > 0:
                rev[-2] = ftF
            flR, mR, ftR = self._block(blk, torch.cat(rev, 1), s)
            fd = (flF + swap(flR, 2)) / 2
            m0 = (mF - mR) / 2
            if self._usesFeat:
                feat = (ftF + ftR) / 2
            flow = fd if flow is None else flow + fd
            mask = mask + m0 if (addMask and mask is not None) else m0
            if i < len(self.blocks) - 1:
                warps = self._gs(src2, self._grid2(flow, ph, pw))
        wi = self._gs(imgs2, self._grid2(flow, ph, pw))
        mask = torch.sigmoid(mask)
        return wi[0:1] * mask + wi[1:2] * (1 - mask)


def exportArch(fastCls, featureIO=True, name=None, **profile):
    """Export-profile class for one variant: IFNet(scale, ensemble, dtype, device,
    width, height[, decomposedWarp]) with the variant's config bound."""
    cfg = fastCls.cfg

    def __init__(self, *args, **kwargs):
        _ExportIFNet.__init__(self, cfg, *args, **kwargs)

    attrs = {"featureIO": featureIO, "__init__": __init__, **profile}
    return type(name or f"Export{fastCls.__name__}", (_ExportIFNet,), attrs)
