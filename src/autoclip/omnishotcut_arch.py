"""
OmniShotCut (UVA Computer Vision Lab, arXiv 2604.24762), inference-only port.

Upstream: https://github.com/UVA-Computer-Vision-Lab/OmniShotCut at 4f8033c
(``omnishotcut/architecture``). License: MIT, code and weights.

A DETR-style shot-query transformer: a ResNet-18 trunk encodes every frame of
a 100-frame window, a transformer encoder attends over all frames x 3x4 cells
with a 3-D sine position code, and 24 learned shot queries decode, in order,
the end frame of each shot (a classification over frame indices) plus the
shot's intra label (general / dissolve / wipe / ...) and inter label (how it
starts: hard cut, transition, sudden jump).

Changes from upstream, none of which change the math:

* The config is the v1.5 checkpoint's, fixed here; the weight file holds only
  the state dict (upstream pickles an ``argparse.Namespace`` beside it).
* Every frame of a window has the same size, so the DETR padding mask is all
  False: it is dropped, and the position code (a function of that mask) is
  computed once at construction.
* Attention runs batch-first through ``scaled_dot_product_attention`` on the
  ``nn.MultiheadAttention`` parameters (same names, so the checkpoint loads
  strict). Dropout (0.0 in the checkpoint) and the auxiliary decoder outputs
  are gone; the decoder returns only its last, normed layer.
* torchvision's ResNet-18 is built without its ImageNet download; the weights
  come from the checkpoint.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

WINDOW = 100
HEIGHT = 96
WIDTH = 128
HIDDEN = 576
HEADS = 8
ENCODER_LAYERS = 3
DECODER_LAYERS = 6
FEEDFORWARD = 2048
QUERIES = 24
INTRA_CLASSES = 10  # 9 labels + no-object
INTER_CLASSES = 7  # 6 labels + no-object

# Label names by index (upstream label_correspondence.*_int2string).
INTRA_NAMES = (
    "General",
    "Dissolve",
    "Wipes",
    "Push",
    "Slide",
    "Zoom",
    "Fade",
    "Doorway",
    "Padding",
)
INTER_NAMES = (
    "New_Start",
    "Hard_Cut",
    "Transition_Source",
    "Transition",
    "Sudden_Jump",
    "Padding",
)

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def _positionCode(frames, height, width, steps):
    """Upstream ``PositionEmbeddingSine`` (normalize=True) on an all-False
    mask, as a ``[frames * height * width, 3 * steps]`` table in the
    transformer's token order (frame-major, then row, then column)."""
    ones = torch.ones(1, frames, height, width)
    eps, scale = 1e-6, 2 * math.pi
    zEmbed = ones.cumsum(1)
    yEmbed = ones.cumsum(2)
    xEmbed = ones.cumsum(3)
    zEmbed = zEmbed / (zEmbed[:, -1:, :, :] + eps) * scale
    yEmbed = yEmbed / (yEmbed[:, :, -1:, :] + eps) * scale
    xEmbed = xEmbed / (xEmbed[:, :, :, -1:] + eps) * scale

    dimT = torch.arange(steps, dtype=torch.float32)
    dimT = 10000 ** (2 * (dimT // 2) / steps)

    def code(embed):
        pos = embed[..., None] / dimT
        return torch.stack((pos[..., 0::2].sin(), pos[..., 1::2].cos()), dim=5).flatten(
            4
        )

    pos = torch.cat((code(zEmbed), code(yEmbed), code(xEmbed)), dim=4)
    return pos[0].reshape(frames * height * width, 3 * steps)


class _Attention(nn.Module):
    """``nn.MultiheadAttention``'s parameters, batch-first, no masks."""

    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.in_proj_weight = nn.Parameter(torch.empty(3 * dim, dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * dim))
        self.out_proj = nn.Linear(dim, dim)

    def _split(self, x):
        b, n, c = x.shape
        return x.view(b, n, self.heads, c // self.heads).transpose(1, 2)

    def forward(self, query, key, value):
        wq, wk, wv = self.in_proj_weight.chunk(3)
        bq, bk, bv = self.in_proj_bias.chunk(3)
        q = self._split(F.linear(query, wq, bq))
        k = self._split(F.linear(key, wk, bk))
        v = self._split(F.linear(value, wv, bv))
        out = F.scaled_dot_product_attention(q, k, v)
        b, _, n, _ = out.shape
        return self.out_proj(out.transpose(1, 2).reshape(b, n, -1))


class _EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attention(HIDDEN, HEADS)
        self.linear1 = nn.Linear(HIDDEN, FEEDFORWARD)
        self.linear2 = nn.Linear(FEEDFORWARD, HIDDEN)
        self.norm1 = nn.LayerNorm(HIDDEN)
        self.norm2 = nn.LayerNorm(HIDDEN)

    def forward(self, src, pos):
        qk = src + pos
        src = self.norm1(src + self.self_attn(qk, qk, src))
        return self.norm2(src + self.linear2(F.relu(self.linear1(src))))


class _DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attention(HIDDEN, HEADS)
        self.multihead_attn = _Attention(HIDDEN, HEADS)
        self.linear1 = nn.Linear(HIDDEN, FEEDFORWARD)
        self.linear2 = nn.Linear(FEEDFORWARD, HIDDEN)
        self.norm1 = nn.LayerNorm(HIDDEN)
        self.norm2 = nn.LayerNorm(HIDDEN)
        self.norm3 = nn.LayerNorm(HIDDEN)

    def forward(self, tgt, memoryKey, memory, queryPos):
        qk = tgt + queryPos
        tgt = self.norm1(tgt + self.self_attn(qk, qk, tgt))
        tgt = self.norm2(tgt + self.multihead_attn(tgt + queryPos, memoryKey, memory))
        return self.norm3(tgt + self.linear2(F.relu(self.linear1(tgt))))


class _Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(_EncoderLayer() for _ in range(ENCODER_LAYERS))


class _Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(_DecoderLayer() for _ in range(DECODER_LAYERS))
        self.norm = nn.LayerNorm(HIDDEN)


class _Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _Encoder()
        self.decoder = _Decoder()


class _Trunk(nn.Module):
    """ResNet-18 up to layer4, under upstream's ``backbone.0.body`` keys."""

    def __init__(self):
        super().__init__()
        from torchvision.models import resnet18
        from torchvision.ops.misc import FrozenBatchNorm2d

        body = resnet18(weights=None, norm_layer=FrozenBatchNorm2d)
        del body.avgpool, body.fc
        self.body = body

    def forward(self, x):
        b = self.body
        x = b.maxpool(b.relu(b.bn1(b.conv1(x))))
        return b.layer4(b.layer3(b.layer2(b.layer1(x))))


class OmniShotCut(nn.Module):
    """``forward(frames)`` takes one or more windows of ``WINDOW`` RGB uint8
    frames, ``[B, WINDOW, HEIGHT, WIDTH, 3]``, and returns, per window, the 24
    queries' argmax labels as ``(intra, inter, end)`` long tensors of shape
    ``[B, QUERIES]`` (``end`` is the exclusive local end frame of each query's
    shot). Upstream's softmax-then-drop-no-object-then-argmax order is kept."""

    def __init__(self):
        super().__init__()
        self.backbone = nn.ModuleList([_Trunk()])
        self.transformer = _Transformer()
        self.input_proj = nn.Conv2d(512, HIDDEN, kernel_size=1)
        self.query_embed = nn.Embedding(QUERIES, HIDDEN)
        self.intra_relation_class_embed = nn.Linear(HIDDEN, INTRA_CLASSES)
        self.inter_relation_class_embed = nn.Linear(HIDDEN, INTER_CLASSES)
        self.range_class_embed = nn.Linear(HIDDEN, WINDOW + 2)

        cellsH, cellsW = HEIGHT // 32, WIDTH // 32
        self.register_buffer(
            "pos", _positionCode(WINDOW, cellsH, cellsW, HIDDEN // 3), persistent=False
        )
        self.register_buffer(
            "mean", torch.tensor(_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "std", torch.tensor(_STD).view(1, 3, 1, 1), persistent=False
        )

    def logits(self, frames):
        """The three heads' raw logits: intra, inter, range."""
        batch = frames.shape[0]
        x = frames.reshape(batch * WINDOW, HEIGHT, WIDTH, 3).permute(0, 3, 1, 2)
        x = x.contiguous()
        x = (x.float() / 255.0 - self.mean) / self.std

        feat = self.input_proj(self.backbone[0](x))  # [B*F, C, h, w]
        _, c, h, w = feat.shape
        src = (
            feat.reshape(batch, WINDOW, c, h * w)
            .permute(0, 1, 3, 2)
            .reshape(batch, WINDOW * h * w, c)
        )
        pos = self.pos.unsqueeze(0)

        for layer in self.transformer.encoder.layers:
            src = layer(src, pos)

        memoryKey = src + pos
        queryPos = self.query_embed.weight.unsqueeze(0).expand(batch, -1, -1)
        tgt = torch.zeros_like(queryPos)
        for layer in self.transformer.decoder.layers:
            tgt = layer(tgt, memoryKey, src, queryPos)
        hs = self.transformer.decoder.norm(tgt)
        return (
            self.intra_relation_class_embed(hs),
            self.inter_relation_class_embed(hs),
            self.range_class_embed(hs),
        )

    def forward(self, frames):
        return tuple(
            logits.softmax(-1)[..., :-1].argmax(-1) for logits in self.logits(frames)
        )
