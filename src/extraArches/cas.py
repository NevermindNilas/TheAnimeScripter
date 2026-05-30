# Vendored from https://github.com/NevermindNilas/AutoCAS (cas.py) @ main.
# PyTorch port of AMD FidelityFX Contrast Adaptive Sharpening (CAS) with
# automatic strength detection. Upstream author: jamyl. Kept close to upstream
# so it can be re-synced; the TAS restore wrapper (AutoCAS) lives in
# src/unifiedRestore.py and is registered as the `autocas` restore method.
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 19:30:28 2023

@author: jamyl

PyTorch port of AMD FidelityFX Contrast Adaptive Sharpening (CAS).
Reference: https://github.com/GPUOpen-Effects/FidelityFX-CAS/blob/master/ffx-cas/ffx_cas.h

This file fixes four deviations from the reference (diagonal soft-min/max,
clamp-to-edge padding, reciprocal-space sharpness lerp, host-syncing asserts)
and adds an optional auto-tuning mode so the caller never has to set `amount`.
"""

import torch as th
import torch.nn.functional as F

EPSILON = 1e-6

# ---------------------------------------------------------------------------
# Auto-tune calibration constants. These are BUILD-TIME constants, not user
# knobs: fit them once to your pipeline with calibrate.py. They replace the
# end-user `amount` slider, they do not add a new one.
# ---------------------------------------------------------------------------
AMOUNT_MAX = 0.9          # hard ceiling so a mis-estimate can never fully over-sharpen
# Blur band calibrated on D:\sisr\v3\sr_out (clean anime 2x SR), 2026-05-29.
LO, HI = -2.31, -1.19     # log blur-score band, fitted to the p5/p95 of that set
GAMMA = 1.0               # blur-demand curve shape (raise to be more conservative)
K = 7                     # local-contrast window (odd)
C0 = 1e-3                 # contrast-normalization floor
# NOTE: CAS is a sharpener, not a denoiser. The amount is auto-tuned purely from a
# blur signal; there is deliberately no noise estimate or noise gate. Broadband noise
# self-limits (it inflates the blur score -> reads as sharp -> demand drops). Structured
# / chroma noise must be handled UPSTREAM (denoise/deband before CAS).


# ---------------------------------------------------------------------------
# Shape helpers: accept [H,W], [C,H,W] or [B,C,H,W] and process internally as
# 4D. replicate padding and the auto-tuner both need a channel/batch dim.
# ---------------------------------------------------------------------------
def _to_4d(x):
    nd = x.dim()
    if nd == 2:      # [H, W]            -> [1, 1, H, W]
        return x[None, None], nd
    if nd == 3:      # [C, H, W]         -> [1, C, H, W]
        return x[None], nd
    return x, nd     # [B, C, H, W]


def _from_4d(x, nd):
    if nd == 2:
        return x[0, 0]
    if nd == 3:
        return x[0]
    return x


def _luma(x):
    """x: [B, C, H, W] -> [B, 1, H, W]. Rec.601 luma for >=3 channels, else mean."""
    if x.shape[1] >= 3:
        w = x.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        return (x[:, :3] * w).sum(1, keepdim=True)
    return x.mean(1, keepdim=True)


def _rep(t, p):
    return F.pad(t, (p, p, p, p), mode='replicate')


def _box(t, k):
    return F.avg_pool2d(_rep(t, k // 2), k, stride=1)


# Fused Laplacian-of-(1-2-1 smooth). The old estimator ran 3 separable convs
# (1-2-1 h, 1-2-1 v, 3x3 Laplacian); all linear -> one 5x5 = lap (X) smooth2d.
_LAP5 = th.tensor([[0.,    0.0625,  0.125,  0.0625, 0.],
                   [0.0625, 0.,    -0.125,  0.,     0.0625],
                   [0.125, -0.125, -0.5,   -0.125,  0.125],
                   [0.0625, 0.,    -0.125,  0.,     0.0625],
                   [0.,     0.0625, 0.125,  0.0625, 0.]]).view(1, 1, 5, 5)


def _feature_maps(x4):
    """Return (hf, contrast) maps used by the estimator and calibration.

    hf       : contrast-normalized high-frequency (blur) energy, [B,1,H,W]
    contrast : local RMS contrast, [B,1,H,W]
    """
    # The blur/contrast maths must be fp32 (hf squares a large ratio; fp16 overflows
    # >65504 -> inf/NaN). But everything downstream needs only the 1-channel luma, so
    # take luma in the input dtype FIRST and upcast that one channel -- not the full
    # 3-channel input. For fp16 in, this is 1/3 the upcast traffic and allocation; the
    # luma weighted-sum stays in [0,1] (no fp16 overflow) and its rounding is washed out
    # by the global blur average. fp32 in: both .float() calls are no-ops -> identical.
    Y = _luma(x4).float()

    # Blur: Laplacian on a pre-smoothed luma so broadband noise does not read as detail.
    # The old path ran 3 convs (1-2-1 h, 1-2-1 v, then a 3x3 Laplacian). All three are
    # linear -> their composition is a single 5x5 conv = lap (X) smooth2d (_LAP5). One
    # conv + one replicate-pad(2): 1/3 the launches/passes; interior-identical (only the
    # replicate border differs by pad-order, negligible in the global blur score).
    lap = F.conv2d(_rep(Y, 2), _LAP5.to(device=Y.device, dtype=Y.dtype))

    mu = _box(Y, K)
    mu2 = _box(Y * Y, K)
    contrast = th.sqrt(th.clamp(mu2 - mu * mu, min=0.0) + EPSILON)
    hf = (lap / (contrast + C0)) ** 2
    return hf, contrast


def _to_amount(score):
    """Map a blur score to an amount in [0, AMOUNT_MAX]. Blurrier -> more sharpening.

    No noise term by design: CAS is a sharpener. Broadband noise self-limits (it raises
    the blur score, so demand drops); structured/chroma noise belongs to an upstream stage.
    """
    t = th.clamp((th.log(score + 1e-4) - LO) / (HI - LO), 0.0, 1.0)
    demand = th.clamp(1.0 - t, 0.0, 1.0) ** GAMMA           # blurrier -> more demand
    return th.clamp(demand, 0.0, AMOUNT_MAX)


def estimate_amount(x4, tiles=0):
    """Auto-estimate the CAS amount from the image itself (no host sync).

    x4    : [B, C, H, W] in [0, 1].
    tiles : 0 -> per-image scalar amount [B,1,1,1] (fast default).
            n>0 -> per-region amount map [B,1,H,W] from an n x n tile grid
                   (recommended for mixed-focus / locally-noisy content).
    """
    dt = x4.dtype
    if x4.shape[-1] < 3 or x4.shape[-2] < 3:
        # too small to measure; fall back to a conservative constant.
        return x4.new_full((x4.shape[0], 1, 1, 1), min(0.5, AMOUNT_MAX))

    hf, contrast = _feature_maps(x4)                        # computed in fp32

    if tiles > 0:
        num = F.adaptive_avg_pool2d(hf * contrast, tiles)
        den = F.adaptive_avg_pool2d(contrast, tiles) + EPSILON
        amt = _to_amount(num / den)                         # per-tile blur demand
        amt = F.interpolate(amt, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        return amt.to(dt)

    # contrast-weighted pooling so flat regions don't dominate the score.
    score = (hf * contrast).sum((1, 2, 3), keepdim=True) / (contrast.sum((1, 2, 3), keepdim=True) + EPSILON)
    return _to_amount(score).to(dt)                         # [B,1,1,1]


def image_stats(x):
    """Public helper for calibration: returns the log blur-score, [B,1,1,1].

    Used by calibrate.py to fit LO/HI (the blur band).
    """
    x4, _ = _to_4d(x)
    hf, contrast = _feature_maps(x4)
    score = (hf * contrast).sum((1, 2, 3), keepdim=True) / (contrast.sum((1, 2, 3), keepdim=True) + EPSILON)
    return th.log(score + 1e-4)


def contrast_adaptive_sharpening(x, amount=0.8, better_diagonals=True, auto_tiles=0):
    """
    Performs a contrast adaptive sharpening on the batch of images x.
    The algorithm is directly implemented from FidelityFX's source code,
    that can be found here
    https://github.com/GPUOpen-Effects/FidelityFX-CAS/blob/master/ffx-cas/ffx_cas.h

    Parameters
    ----------
    x : Tensor
        Image or stack of images, of shape [burst, channels, ny, nx].
        Burst and channel dimensions can be ommited.
    amount : float in [0, 1], Tensor, or None
        Amount of sharpening, 0 = minimum, 1 = maximum.
        Pass None to auto-tune the amount from the image content (see auto_tiles).
        A broadcastable Tensor is also accepted (e.g. a per-pixel amount map).
    better_diagonals : bool, optional
        If False, the algorithm runs slightly faster, but
        won't consider diagonals. The default is True.
    auto_tiles : int, optional
        Only used when amount is None. 0 -> single auto amount for the whole
        image. n>0 -> per-region amount map from an n x n tile grid.

    Returns
    -------
    Tensor
        Processed stack of images (same rank as the input).
    """
    assert x.dim() >= 2

    x4, nd = _to_4d(x)

    # Reference samples the 3x3 neighborhood with CLAMP-TO-EDGE addressing.
    # torch's default pad mode is constant-zero, which injects a fake black
    # border; 'replicate' matches the GPU behaviour.
    x_padded = F.pad(x4, pad=(1, 1, 1, 1), mode='replicate')

    # Extracting the 3x3 neighborhood around each pixel
    # a b c
    # d e f
    # g h i
    b = x_padded[..., :-2, 1:-1]
    d = x_padded[..., 1:-1, :-2]
    e = x4                          # center == input interior (replicate pad leaves it unchanged)
    f = x_padded[..., 1:-1, 2:]
    h = x_padded[..., 2:, 1:-1]

    if better_diagonals:
        a = x_padded[..., :-2, :-2]
        c = x_padded[..., :-2, 2:]
        g = x_padded[..., 2:, :-2]
        i = x_padded[..., 2:, 2:]

    # Soft min / max over the neighborhood. Pairwise minimum/maximum instead of
    # th.stack(list).min(dim=0): avoids materializing a [N,B,C,H,W] tensor + strided
    # gather (~1.8x faster), and is bit-exact (min/max never round). The diagonal term
    # reuses the cross result -- min over all 9 == min(min(cross5), min(diag4)) -- so
    # there is no 9-way re-reduction.
    mn = th.minimum(th.minimum(th.minimum(th.minimum(b, d), e), f), h)
    mx = th.maximum(th.maximum(th.maximum(th.maximum(b, d), e), f), h)

    if better_diagonals:
        mn_d = th.minimum(th.minimum(th.minimum(a, c), g), i)
        mx_d = th.maximum(th.maximum(th.maximum(a, c), g), i)
        mn = mn + th.minimum(mn, mn_d)     # == cross5 + min(all9), matching ffx_cas.h
        mx = mx + th.maximum(mx, mx_d)     # == cross5 + max(all9)
        lim = 2 - mx
    else:
        lim = 1 - mx

    # Local weight. Divide (not reciprocal*x): minimum(mn,lim) <= mn <= mx, so the
    # ratio is <=1 by construction and stays finite. reciprocal(mx+EPS) instead
    # OVERFLOWS in fp16 for near-black pixels (mx~0 -> 1e6 > 65504 -> inf, then
    # inf*0 = NaN). The min=1e-4 floor is an fp16-normal denominator guard (replaces
    # the old EPSILON; for mx>=1e-4 the ratio is exact, matching ffx_cas.h's rcp(mx)).
    # saturate(.) before sqrt matches the reference (ASatF1).
    amp = th.sqrt(th.clamp(th.minimum(mn, lim) / mx.clamp(min=1e-4), 0.0, 1.0))

    # Resolve the sharpening amount.
    if amount is None:
        amount = estimate_amount(x4, tiles=auto_tiles)
    elif not th.is_tensor(amount):
        amount = x4.new_tensor(float(amount))
    amount = th.clamp(amount, 0.0, 1.0)

    # peak = -1 / lerp(8, 5, amount); interpolation happens in the DENOMINATOR
    # (reciprocal space), matching ffx_cas.h. denom in [5, 8] -> no EPSILON.
    # Fold the sign into the (tiny, broadcast [B,1,1,1]) coefficient so the per-pixel
    # step is a single multiply instead of neg + mul.
    w = amp * (-th.reciprocal(8.0 - 3.0 * amount))

    # The local conv filter is
    # 0 w 0
    # w 1 w
    # 0 w 0
    # Fused tail: addcmul gives (e + sum4*w) in one pass; a single divide replaces
    # reciprocal(den) + mul. Both are fewer full-tensor passes than the split form and,
    # being single-rounded, are no less accurate than the reference arithmetic.
    # w = -amp/(8-3a) in [-0.2, 0] (amp<=1, 8-3a in [5,8]) -> 1+4w in [0.2, 1] > 0, so
    # the divide is always safe with no EPSILON, matching ffx_cas.h.
    sum4 = b + d + f + h
    den = w.mul(4.0).add_(1.0)                     # 1 + 4w
    output = th.addcmul(e, sum4, w).div_(den)      # in-place div: no extra peak buffer

    # Clipping between 0 and 1. It fixes previous divisions by 0 too. In-place: output
    # is the private div_ result, so the final saturate needs no new full-tensor buffer.
    return _from_4d(output.clamp_(0, 1), nd)
