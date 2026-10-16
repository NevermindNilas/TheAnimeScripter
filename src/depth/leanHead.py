"""Memory-lean forwards for the DPT depth heads.

At ``--depth_quality high`` the heads, not the ViT, set the VRAM peak: their
last stages upsample a wide feature map to the full input resolution and run
3x3 convs on it (large_v3 from a 1080p source builds a 1.07 GB fp32 map there,
and a 705 MB one a stage earlier). ``stripedUpsampleConv`` runs such a stage a
band of output rows at a time, so the full-resolution map never exists.

A map that already fits ``STRIPE_BUDGET`` takes the original single-pass ops,
so low/medium output is unchanged. Striped output matches within fp rounding:
the bilinear blend runs height-then-width instead of width-then-height, and
cuDNN may pick a different conv algorithm for a band.
"""

import torch
import torch.nn.functional as F

# Bytes of upsampled feature map materialized per band. Above every map that
# --depth_quality low/medium builds from a 1080p source (the largest, VDA's
# 700 px tail, is 111 MB), so only high is striped.
STRIPE_BUDGET = 128 * 2**20


def _exporting():
    return torch.jit.is_tracing() or torch.onnx.is_in_onnx_export()


def stripedUpsampleConv(
    x, size, heads, interpolate=F.interpolate, addRows=None, budget=None
):
    """``[head(up) for head in heads]`` for ``up = interpolate(x, size,
    bilinear, align_corners=True)``, computed a band of rows at a time.

    Every head may read at most one row above and below each output row (one
    3x3/pad-1 conv, everything else per pixel). ``addRows(band, r0, r1)``, when
    given, adds a per-pixel term to rows ``r0:r1`` of the upsampled map before
    the heads run (DA3's UV position embedding).

    ``x`` may be handed over as a one-element list, which is emptied: then the
    single-pass path frees ``x`` once it is upsampled and a single head gets
    the only reference to the upsampled map, matching the original ops' peak
    (callers' own references would otherwise pin both).
    """
    if isinstance(x, list):
        x = x.pop()
    B, C, hIn, wIn = x.shape
    hOut, wOut = size
    # The band is blended at fp32 (the kernel's accumulator) whatever the
    # input dtype, so size it on 4-byte elements.
    rows = (STRIPE_BUDGET if budget is None else budget) // max(1, B * C * wOut * 4)
    if rows >= hOut or _exporting():
        up = [interpolate(x, size, mode="bilinear", align_corners=True)]
        del x
        if addRows is not None:
            up = [addRows(up.pop(), 0, hOut)]
        if len(heads) == 1:
            return [heads[0](up.pop())]
        return [head(up[0]) for head in heads]
    rows = max(rows, 16)

    # Source row and blend weight of every output row, computed the way
    # upsample_bilinear2d does: scale (in-1)/(out-1) at its accumulation
    # precision (fp32, fp64 for double), times the row index.
    # Built on the CPU and uploaded once, so the band loop never reads the GPU
    # back (a per-band sync stalls the launch queue).
    acc = torch.float64 if x.dtype == torch.float64 else torch.float32
    scale = torch.tensor(hIn - 1, dtype=acc) / (hOut - 1)
    srcRow = torch.arange(hOut, dtype=acc) * scale
    h1 = srcRow.long().clamp_(max=hIn - 1)
    h1p = torch.where(h1 < hIn - 1, h1 + 1, h1)
    lam = (srcRow - h1.to(acc)).to(x.device).view(1, 1, -1, 1)
    h1Dev, h1pDev = h1.to(x.device), h1p.to(x.device)

    outs = [None] * len(heads)
    for r0 in range(0, hOut, rows):
        r1 = min(hOut, r0 + rows)
        a, b = max(0, r0 - 1), min(hOut, r1 + 1)  # one halo row each side
        s0, s1 = int(h1[a]), int(h1p[b - 1]) + 1
        # Blend the band's rows at the narrow source width, then one
        # width-only upsample (height scale exactly 1), all at the kernel's
        # accumulation precision and rounded to the input dtype once, like the
        # fused kernel; only the blend order differs. Rounding the row blend
        # to fp16 before the width pass cost ~23 dB on small_v2.
        src = x[:, :, s0:s1]
        band = torch.lerp(
            src.index_select(2, h1Dev[a:b] - s0).to(acc),
            src.index_select(2, h1pDev[a:b] - s0).to(acc),
            lam[:, :, a:b],
        )
        del src
        band = F.interpolate(
            band, size=(b - a, wOut), mode="bilinear", align_corners=True
        ).to(x.dtype)
        if addRows is not None:
            band = addRows(band, a, b)
        top = r0 - a
        for i, head in enumerate(heads):
            y = head(band)[:, :, top : top + (r1 - r0)]
            if outs[i] is None:
                # Preallocated rather than torch.cat'd, so the output is never
                # held twice.
                outs[i] = y.new_empty((*y.shape[:2], hOut, y.shape[3]))
            outs[i][:, :, r0:r1] = y
        del band
    return outs


# ---------------------------------------------------------------------------
# Depth Anything 3 heads (DPT for the mono large models, DualDPT for
# small/base/limbo). Bound onto a loaded model by installLeanHead.
# ---------------------------------------------------------------------------


def _resizedFeats(head, feats, H, W, patchStartIdx, contiguous):
    B, _, C = feats[0].shape
    ph, pw = H // head.patch_size, W // head.patch_size
    resized = []
    for stageIdx, takeIdx in enumerate(head.intermediate_layer_idx):
        x = feats[takeIdx][:, patchStartIdx:]
        x = head.norm(x)
        # DPT reshapes a contiguous copy, DualDPT the permuted view; the conv
        # picks its kernel by layout, so each keeps its own to stay bit-exact.
        x = x.permute(0, 2, 1)
        if contiguous:
            x = x.contiguous()
        x = x.reshape(B, C, ph, pw)
        x = head.projects[stageIdx](x)
        if head.pos_embed:
            x = head._add_pos_embed(x, W, H)
        resized.append(head.resize_layers[stageIdx](x))
    return resized, ph, pw


def mainFusion(s, resized, interpolate=F.interpolate):
    """A DPT scratch's main fusion chain through output_conv1, with
    refinenet1's x2 upsample, 1x1 out_conv and output_conv1 striped.
    Empties ``resized`` so each stage map is freed once it is consumed."""
    l1, l2, l3, l4 = resized
    resized.clear()
    l1 = s.layer1_rn(l1)
    l2 = s.layer2_rn(l2)
    l3 = s.layer3_rn(l3)
    l4 = s.layer4_rn(l4)
    out = s.refinenet4(l4, size=l3.shape[2:])
    out = s.refinenet3(out, l3, size=l2.shape[2:])
    out = s.refinenet2(out, l2, size=l1.shape[2:])
    del l2, l3, l4
    held = [s.refinenet1.resConfUnit2(out + s.refinenet1.resConfUnit1(l1))]
    del out, l1
    return refineOutput(s, held, interpolate)


def refineOutput(s, held, interpolate=F.interpolate):
    """The rest of ``output_conv1(refinenet1(...))`` once its residual units
    have run: refinenet1's upsample, its 1x1 out_conv and output_conv1,
    striped. ``held`` is a one-element list handing over the map."""
    r1 = s.refinenet1
    h, w = held[0].shape[2:]
    size = r1.size if r1.size is not None else (h * 2, w * 2)

    def head(t):
        t = r1.out_conv(t)  # rebinding frees the upsampled map, as r1 did
        return s.output_conv1(t)

    (out,) = stripedUpsampleConv(held, size, [head], interpolate)
    return out


def _posRows(head, W, H, hOut, wOut):
    """addRows hook adding DA3's UV embedding, rows r0:r1 of the full grid.
    The embedding is per pixel, so slicing the grid first gives the same
    values as _add_pos_embed on the whole map."""
    from depth_anything_3.model.utils.head_utils import (
        create_uv_grid,
        position_grid_to_embed,
    )

    def add(band, r0, r1):
        grid = create_uv_grid(
            wOut, hOut, aspect_ratio=W / H, dtype=band.dtype, device=band.device
        )[r0:r1]
        pe = position_grid_to_embed(grid, band.shape[1]) * 0.1
        return band + pe.permute(2, 0, 1)[None].expand(band.shape[0], -1, -1, -1)

    return add


def _leanDPTForward(self, feats, H, W, patch_start_idx):
    from depth_anything_3.model.utils.head_utils import custom_interpolate

    if self.has_conf or self.pos_embed:
        return self._originalForwardImpl(feats, H, W, patch_start_idx)
    resized, ph, pw = _resizedFeats(self, feats, H, W, patch_start_idx, True)
    fused = [mainFusion(self.scratch, resized, custom_interpolate)]
    hOut = int(ph * self.patch_size / self.down_ratio)
    wOut = int(pw * self.patch_size / self.down_ratio)
    heads = [self.scratch.output_conv2]
    if self.use_sky_head:
        heads.append(self.scratch.sky_output_conv2)
    ys = stripedUpsampleConv(fused, (hOut, wOut), heads, custom_interpolate)
    outs = {
        self.head_main: self._apply_activation_single(ys[0], self.activation).squeeze(1)
    }
    if self.use_sky_head:
        outs[self.sky_name] = self._apply_sky_activation(ys[1]).squeeze(1)
    return outs


def _leanDualDPTForward(self, feats, H, W, patch_start_idx):
    """DualDPT's depth branch alone. The aux (ray) branch is a second full
    fusion chain whose output no TAS path reads -- OutputProcessor takes only
    depth/depth_conf -- so skipping it leaves depth bit-identical."""
    from depth_anything_3.model.utils.head_utils import custom_interpolate

    resized, ph, pw = _resizedFeats(self, feats, H, W, patch_start_idx, False)
    fused = [mainFusion(self.scratch, resized, custom_interpolate)]
    hOut = int(ph * self.patch_size / self.down_ratio)
    wOut = int(pw * self.patch_size / self.down_ratio)
    addRows = _posRows(self, W, H, hOut, wOut) if self.pos_embed else None
    (logits,) = stripedUpsampleConv(
        fused, (hOut, wOut), [self.scratch.output_conv2], custom_interpolate, addRows
    )
    fmap = logits.permute(0, 2, 3, 1)
    return {
        self.head_main: self._apply_activation_single(
            fmap[..., :-1], self.activation
        ).squeeze(-1),
        f"{self.head_main}_conf": self._apply_activation_single(
            fmap[..., -1], self.conf_activation
        ),
    }


def installLeanHead(model):
    """Rebind the DA3 head of a loaded MonocularDepthAnything3 for inference.

    Weights are untouched; ONNX export binds its own depth-only head
    (streaming_export) and tracing falls back to the single-pass ops.
    """
    import types

    from depth_anything_3.model.dpt import DPT
    from depth_anything_3.model.dualdpt import DualDPT

    head = model.model.head
    if isinstance(head, DualDPT):
        forward = _leanDualDPTForward
    elif isinstance(head, DPT):
        forward = _leanDPTForward
    else:
        return model
    head._originalForwardImpl = head._forward_impl
    head._forward_impl = types.MethodType(forward, head)
    return model
