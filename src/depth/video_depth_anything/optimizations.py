# Copyright (2025) Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Inference optimizations for the streaming VideoDepthAnything arch, ported from
the proven image DepthAnythingV2 fp16 fast path (see src/depth/dinov2.py,
src/depth/blocks_v2.py, src/depth/fold_layerscale.py) plus a configurable
temporal-attention window.

All changes are forward-only / weight-identical (SDPA reuses the same qkv/proj
Linears; skip-add has no params; LayerScale fold is algebraically exact in fp32).
Applied by rebinding instance forwards on an og VideoDepthAnything model, so the
original module (video_depth_stream.py) is left untouched for the og_video_* path.

Measured on RTX 3090 (ViT-S, 360p streaming, fp16): +~50% forward vs plain-fp16
baseline; quality within the fp16 noise floor. NOT applied: temporal-attention
SDPA (measured regression at seq=32/head_dim=24).
"""

import types

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms import Compose

from .blocks import FeatureFusionBlock, ResidualConvUnit
from .dinov2_layers.attention import Attention
from .dinov2_layers.layer_scale import LayerScale
from .dinov2_layers.mlp import Mlp
from .transform import NormalizeImage, PrepareForNet, Resize

INFER_LEN = 32


# ---- DINOv2 ViT attention: SDPA (fp16/bf16) with fp32 manual fallback ----
def _sdpa_attn_forward(self, x, attn_bias=None):
    B, N, C = x.shape
    qkv = (
        self.qkv(x)
        .reshape(B, N, 3, self.num_heads, C // self.num_heads)
        .permute(2, 0, 3, 1, 4)
    )
    if x.dtype in (torch.float16, torch.bfloat16):
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        x = x.transpose(1, 2).reshape(B, N, C)
    else:
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


# ---- DPT blocks: FloatFunctional.add -> '+', drop redundant .contiguous() ----
def _rcu_forward(self, x):
    out = self.activation(x)
    out = self.conv1(out)
    if self.bn:
        out = self.bn1(out)
    out = self.activation(out)
    out = self.conv2(out)
    if self.bn:
        out = self.bn2(out)
    if self.groups > 1:
        out = self.conv_merge(out)
    return out + x


def _ffb_forward(self, *xs, size=None):
    output = xs[0]
    if len(xs) == 2:
        res = self.resConfUnit1(xs[1])
        output = output + res
    output = self.resConfUnit2(output)
    if size is None and self.size is None:
        modifier = {"scale_factor": 2}
    elif size is None:
        modifier = {"size": self.size}
    else:
        modifier = {"size": size}
    output = F.interpolate(
        output, **modifier, mode="bilinear", align_corners=self.align_corners
    )
    output = self.out_conv(output)
    return output


# ---- pos-embed interpolation cache (constant input size in streaming) ----
def _install_pos_cache(model):
    dino = model.pretrained
    orig = dino.interpolate_pos_encoding
    cache = {}

    def cached(x, w, h):
        key = (x.shape[1], w, h, x.dtype)
        r = cache.get(key)
        if r is None:
            r = orig(x, w, h)
            cache[key] = r
        return r

    dino.interpolate_pos_encoding = cached


# ---- LayerScale fold into preceding Linear (attn.proj / mlp.fc2) ----
@torch.no_grad()
def _fold_into_linear(linear, gamma):
    linear.weight.mul_(gamma.unsqueeze(1).to(linear.weight.dtype))
    if linear.bias is not None:
        linear.bias.mul_(gamma.to(linear.bias.dtype))


@torch.no_grad()
def _fold_layerscale(model):
    folded = 0
    for m in model.modules():
        if not (hasattr(m, "attn") and hasattr(m, "mlp")):
            continue
        ls1 = getattr(m, "ls1", None)
        if isinstance(ls1, LayerScale) and hasattr(m.attn, "proj"):
            _fold_into_linear(m.attn.proj, ls1.gamma.data)
            m.ls1 = nn.Identity()
            folded += 1
        ls2 = getattr(m, "ls2", None)
        if isinstance(ls2, LayerScale):
            last = getattr(m.mlp, "fc2", None) if isinstance(m.mlp, Mlp) else None
            if isinstance(last, nn.Linear):
                _fold_into_linear(last, ls2.gamma.data)
                m.ls2 = nn.Identity()
                folded += 1
    return folded


def keep_fp32_islands(model):
    """Keep the arch's intended fp32 island (final output_conv2) in fp32 after
    .half(). dpt_temporal casts its input to .float(); the weights must match.
    Call AFTER model.half()."""
    model.head.scratch.output_conv2.float()


# ---- windowed + dtype-aware streaming inference ----
def _optimized_infer_one(self, frame, input_size=518, device="cuda", fp32=False):
    """Faithful port of VideoDepthAnything.infer_video_depth_one with two changes:
    (1) cast the internally-built input to the model dtype (so a .half() model
    works end-to-end); (2) attend a configurable window (self._n_recent recent
    frames + 2 anchors) instead of the fixed 31. Storage/slide unchanged."""
    dtype = self.pretrained.patch_embed.proj.weight.dtype
    n_recent = getattr(self, "_n_recent", INFER_LEN - 3)
    self.id += 1

    if self.transform is None:  # first frame
        frame_height, frame_width = frame.shape[:2]
        self.frame_height = frame_height
        self.frame_width = frame_width
        ratio = max(frame_height, frame_width) / min(frame_height, frame_width)
        if ratio > 1.78:
            input_size = int(input_size * 1.777 / ratio)
            input_size = round(input_size / 14) * 14
        self.transform = Compose(
            [
                Resize(
                    width=input_size,
                    height=input_size,
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method="lower_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
        )
        cur_input = (
            torch.from_numpy(
                self.transform({"image": frame.astype(np.float32) / 255.0})["image"]
            )
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device=device, dtype=dtype)
        )
        with torch.inference_mode():
            cur_feature = self.forward_features(cur_input)
            depth, cached_hidden_state_list = self.forward_depth(
                cur_feature, cur_input.shape
            )
        depth = depth.to(cur_input.dtype)
        depth = F.interpolate(
            depth.flatten(0, 1).unsqueeze(1),
            size=(frame_height, frame_width),
            mode="bilinear",
            align_corners=True,
        )
        self.frame_cache_list = [cached_hidden_state_list] * INFER_LEN
        self.frame_id_list.extend([0] * (INFER_LEN - 1))
        new_depth = depth[0][0].float().cpu().numpy()
    else:
        frame_height, frame_width = frame.shape[:2]
        cur_input = (
            torch.from_numpy(
                self.transform({"image": frame.astype(np.float32) / 255.0})["image"]
            )
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device=device, dtype=dtype)
        )
        with torch.inference_mode():
            cur_feature = self.forward_features(cur_input)
            cache_list = self.frame_cache_list
            cur_list = cache_list[0:2] + cache_list[-n_recent:]
            cur_cache = [
                torch.cat([h[i] for h in cur_list], dim=1)
                for i in range(len(cur_list[0]))
            ]
            depth, new_cache = self.forward_depth(
                cur_feature, cur_input.shape, cached_hidden_state_list=cur_cache
            )
        depth = depth.to(cur_input.dtype)
        depth = F.interpolate(
            depth.flatten(0, 1).unsqueeze(1),
            size=(frame_height, frame_width),
            mode="bilinear",
            align_corners=True,
        )
        new_depth = depth[-1][0].float().cpu().numpy()
        self.frame_cache_list.append(new_cache)

    self.frame_id_list.append(self.id)
    if self.id + INFER_LEN > self.gap + 1:
        del self.frame_id_list[1]
        del self.frame_cache_list[1]
    return new_depth


def apply_optimizations(model, window=32):
    """Rebind optimized forwards on `model` (an og VideoDepthAnything instance).
    `window` = temporal attention window (frames attended, incl. current);
    default 32 reproduces the original behavior exactly. Call BEFORE .half().
    Returns a dict of applied counts."""
    if window < 4 or window > INFER_LEN:
        raise ValueError(f"window must be in [4, {INFER_LEN}], got {window}")
    counts = dict(sdpa=0, fast_add=0, ls_fold=0)
    for m in model.modules():
        if isinstance(m, Attention):
            m.forward = types.MethodType(_sdpa_attn_forward, m)
            counts["sdpa"] += 1
        elif isinstance(m, ResidualConvUnit):
            m.forward = types.MethodType(_rcu_forward, m)
            counts["fast_add"] += 1
        elif isinstance(m, FeatureFusionBlock):
            m.forward = types.MethodType(_ffb_forward, m)
            counts["fast_add"] += 1
    counts["ls_fold"] = _fold_layerscale(model)
    _install_pos_cache(model)
    model._n_recent = window - 3
    model.infer_video_depth_one = types.MethodType(_optimized_infer_one, model)
    return counts
