"""SMoVSR (2x_umzi_anime_smovsr): a bidirectional recurrent video upscaler.

Ported from https://github.com/enhancr/SMoVSR (``smovsr_arch.py``, MIT License,
Copyright (c) 2026 enhancr). Only the inference configuration TAS ships is
kept (re-parameterised ``ConvNXC`` convolutions, a ``pixelshuffledirect``
upsampler), and parameter names are unchanged so the release ``.pth`` loads
strictly. ``.eval()`` folds every ``ConvNXC``/``DOConv2d`` into one ``Conv2d``.

``SMoVSRWindow`` is TAS's streaming form of ``SMoVSR.forward``. The upstream
forward reads a whole clip twice (first to last, then last to first), so every
output frame depends on every input frame. TAS cannot hold a whole video, so it
runs windows of ``T`` frames and keeps the first ``S``:

- the forward recurrence is causal, so its state is carried from one window to
  the next (``state`` in, ``stateOut`` out) and never restarts mid-shot;
- the backward recurrence needs future frames, so each window reads ``T - S``
  frames past its last output;
- ``gates`` restart either recurrence exactly where a shot (or the stream)
  starts or ends, so a window that is padded past a cut or the last frame
  produces the same frames as the upstream forward run on the real frames.

With every gate open and ``S == T`` it is ``SMoVSR.forward`` on one clip.
"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter


def _pair(x):
    if isinstance(x, tuple | list):
        if len(x) < 2:
            return x[0], x[0]
        if len(x) > 2:
            return x[0], x[1]
        return x
    return x, x


def _samePadding(kernelSize, dilation):
    kh, kw = kernelSize
    dh, dw = dilation
    return ((kh - 1) * dh) // 2, ((kw - 1) * dw) // 2


class DOConv2d(nn.Module):
    """Depthwise over-parameterised convolution; ``.eval()`` folds it into ``eval_conv``."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=None,
        groups=1,
        bias=True,
        dilation=1,
        mul=1.0,
    ):
        super().__init__()
        kernel_size = _pair(kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.mul = nn.Parameter(torch.ones(1) * mul, requires_grad=True)
        self.dilation = _pair(dilation)
        self.padding = (
            _samePadding(self.kernel_size, self.dilation)
            if padding is None
            else _pair(padding)
        )
        self.groups = groups
        m, n = self.kernel_size
        self.D_mul = m * n
        self.W = Parameter(torch.empty(out_channels, in_channels // groups, self.D_mul))
        self.dow_func = self.dow
        if m * n > 1:
            self.D = Parameter(torch.zeros(in_channels, m * n, self.D_mul))
            eye = torch.eye(m * n, dtype=torch.float32).reshape(1, m * n, m * n)
            self.d_diag = Parameter(
                eye.repeat((in_channels, 1, self.D_mul // (m * n))),
                requires_grad=False,
            )
            self.dow_func = self.dow_mult
        self.DoW_shape = (out_channels, in_channels // groups, m, n)
        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        self.eval_conv = nn.Conv2d(
            in_channels,
            out_channels,
            self.kernel_size,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        self.forward_func = self.train_forward

    def dow_mult(self):
        d = self.D + self.d_diag
        w = torch.reshape(
            self.W, (self.out_channels // self.groups, self.in_channels, self.D_mul)
        )
        return torch.reshape(torch.einsum("ims,ois->oim", d, w), self.DoW_shape)

    def dow(self):
        return torch.reshape(self.W, self.DoW_shape)

    def update_eval(self):
        self.eval_conv.weight.data = (
            (self.dow_func() * self.mul).clone().detach().contiguous()
        )
        self.eval_conv.bias.data = (self.bias * self.mul).clone().detach().contiguous()

    def train(self, mode: bool = True):
        if not mode:
            self.update_eval()
            self.forward_func = self.eval_forward
        else:
            self.forward_func = self.train_forward
        return super().train(mode)

    def train_forward(self, x):
        return F.conv2d(
            x,
            self.dow_func() * self.mul,
            self.bias * self.mul,
            1,
            self.padding,
            self.dilation,
            self.groups,
        )

    def eval_forward(self, x):
        return self.eval_conv(x)

    def forward(self, x):
        return self.forward_func(x)


class ConvNXC(nn.Module):
    """1x1 -> kxk -> 1x1 plus a 1x1 skip, folded into one kxk conv by ``.eval()``."""

    def __init__(self, c_in, c_out, kernel_size=(3, 3), gain1=2, mul=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.sk = DOConv2d(c_in, c_out, kernel_size=1, padding=0, mul=0.02)
        self.padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)
        self.conv = nn.Sequential(
            DOConv2d(c_in, c_in * gain1, kernel_size=1, padding=0),
            DOConv2d(c_in * gain1, c_out * gain1, kernel_size=kernel_size, padding=0),
            DOConv2d(c_out * gain1, c_out, kernel_size=1, padding=0, mul=mul),
        )
        self.eval_conv = nn.Conv2d(c_in, c_out, kernel_size, 1, self.padding)
        self.forward_func = self.forward_train

    def update_params(self) -> None:
        for conv in self.conv:
            conv.update_eval()
        w1 = self.conv[0].eval_conv.weight.data.clone().detach()
        w2 = self.conv[1].eval_conv.weight.data.clone().detach()
        w3 = self.conv[2].eval_conv.weight.data.clone().detach()
        kh, kw = self.kernel_size
        w = (
            F.conv2d(w1.flip(2, 3).permute(1, 0, 2, 3), w2, padding=(kh - 1, kw - 1))
            .flip(2, 3)
            .permute(1, 0, 2, 3)
        )
        weight = (
            F.conv2d(w.flip(2, 3).permute(1, 0, 2, 3), w3)
            .flip(2, 3)
            .permute(1, 0, 2, 3)
        )
        self.sk.update_eval()
        skW = self.sk.eval_conv.weight.data.clone().detach()
        b1 = self.conv[0].eval_conv.bias.data.clone().detach()
        b2 = self.conv[1].eval_conv.bias.data.clone().detach()
        b3 = self.conv[2].eval_conv.bias.data.clone().detach()
        b = (w2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b2
        bias = (w3 * b.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b3
        skB = self.sk.eval_conv.bias.data.clone().detach()
        padH, padW = (kh - 1) // 2, (kw - 1) // 2
        skW = F.pad(skW, [padW, padW, padH, padH])
        self.eval_conv.weight.data = (weight + skW).contiguous()
        self.eval_conv.bias.data = (bias + skB).contiguous()

    def train(self, mode: bool = True):
        super().train(mode)
        if not mode:
            self.update_params()
            self.forward_func = self.forward_eval
        else:
            self.forward_func = self.forward_train
        return self

    def forward_train(self, x):
        padH = (self.kernel_size[0] - 1) // 2
        padW = (self.kernel_size[1] - 1) // 2
        xPad = F.pad(x, (padW, padW, padH, padH), "constant", 0)
        return self.conv(xPad) + self.sk(x)

    def forward_eval(self, x):
        return self.eval_conv(x)

    def forward(self, x):
        return self.forward_func(x)


class UniUpsampleV4_light(nn.Sequential):
    """The ``pixelshuffledirect`` upsampler; ``MetaUpsample`` is kept for strict loading."""

    def __init__(self, scale=2, in_dim=32, out_dim=3, mid_dim=32, group=4):
        super().__init__(
            ConvNXC(in_dim, out_dim * scale**2, (3, 3)), nn.PixelShuffle(scale)
        )
        # 254, index of "pixelshuffledirect" in upstream SampleMods, scale, dims, rep
        self.register_buffer(
            "MetaUpsample",
            torch.tensor(
                [254, 1, scale, in_dim, out_dim, mid_dim, group, 1], dtype=torch.uint8
            ),
        )


class SMB(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.short = in_channels == out_channels
        self.body = nn.Sequential(
            ConvNXC(in_channels, out_channels, (3, 3)),
            nn.SiLU(True),
            ConvNXC(out_channels, out_channels, (3, 3)),
            nn.SiLU(True),
            ConvNXC(out_channels, out_channels, (3, 3)),
        )
        if self.short:
            self.gate_conv = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 1), nn.Tanh()
            )

    def forward(self, x):
        out = self.body(x)
        if self.short:
            return (out + x).mul_(self.gate_conv(out))
        return F.silu(out, True)


class FSMoSR(nn.Module):
    def __init__(self, dim=32, h_clamp: float | None = None, align_kernel: int = 7):
        super().__init__()
        self.blocks_2 = nn.Sequential(*[SMB(dim) for _ in range(3)])
        self.end_block = nn.Sequential(SMB(dim), ConvNXC(dim, dim, (3, 3)))
        self.h_clamp = h_clamp
        self.align = DOConv2d(dim, dim, align_kernel, groups=dim)
        self.z = nn.Sequential(nn.Conv2d(2 * dim, dim, 1), nn.Sigmoid())

    def clamp(self, t):
        return t if self.h_clamp is None else t.clamp(-self.h_clamp, self.h_clamp)

    def step_wo_hystory(self, x):
        feat = self.blocks_2(x)
        return self.end_block(feat + x), feat

    def forward(self, x, h):
        h = self.clamp(self.align(h))
        feat = self.blocks_2(x + h)
        out = self.end_block(feat + x)
        z = self.z(torch.cat([x, h], 1))
        return out, h + z * (self.clamp(feat) - h)

    def gatedStep(self, x, h, gate):
        """``forward(x, h)`` where ``gate`` is 1, ``step_wo_hystory`` (clamped) where 0.

        Exact for a 0/1 gate in any precision: the closed branch is multiplied
        by 0 and the open one by 1.
        """
        hA = self.clamp(self.align(h)) * gate
        feat = self.blocks_2(x + hA)
        out = self.end_block(feat + x)
        z = self.z(torch.cat([x, hA], 1))
        clamped = self.clamp(feat)
        return out, gate * (hA + z * (clamped - hA)) + (1 - gate) * clamped


class SMoVSR(nn.Module):
    def __init__(self, in_ch=3, dim=32, scale=2, h_clamp=30.0, align_kernel=7):
        super().__init__()
        self.scale = scale
        self.dim = dim
        self.fe = nn.Sequential(SMB(in_ch, dim), SMB(dim, dim))
        self.f_forward = FSMoSR(dim, h_clamp, align_kernel)
        self.f_backward = FSMoSR(dim, h_clamp, align_kernel)
        self.backward_cat = nn.Conv2d(dim * 2, dim, 1)
        self.upsampler = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.PReLU(dim),
            UniUpsampleV4_light(scale, dim, 3, 32, 4),
        )

    def forward(self, x):
        """The upstream whole-clip forward: ``(B, T, 3, H, W)`` -> ``(B, T, 3, sH, sW)``."""
        b, t, c, h, w = x.size()
        flat = x.view(-1, c, h, w)
        short = F.interpolate(
            flat, scale_factor=self.scale, mode="bilinear", align_corners=False
        )
        feats = self.fe(flat).view(b, t, -1, h, w)
        short = short.view(b, t, -1, h * self.scale, w * self.scale)
        forwardFeats = []
        out, prop = self.f_forward.step_wo_hystory(feats[:, 0])
        prop = self.f_forward.clamp(prop)
        forwardFeats.append(out)
        for i in range(1, t):
            out, prop = self.f_forward(feats[:, i], prop)
            forwardFeats.append(out)
        backwardFeats = [None] * t
        last = self.backward_cat(torch.cat([feats[:, -1], forwardFeats[-1]], 1))
        out, prop = self.f_backward.step_wo_hystory(last)
        prop = self.f_forward.clamp(prop)
        backwardFeats[t - 1] = out
        for i in range(t - 2, -1, -1):
            out, prop = self.f_backward(
                self.backward_cat(torch.cat([feats[:, i], forwardFeats[i]], 1)), prop
            )
            backwardFeats[i] = out
        fused = torch.cat(
            [torch.stack(forwardFeats, 1), torch.stack(backwardFeats, 1)], dim=2
        )
        fused = self.upsampler(fused.flatten(0, 1))
        return fused.view(b, t, -1, h * self.scale, w * self.scale) + short


class SMoVSRWindow(nn.Module):
    """One streaming window of ``SMoVSR`` (see the module docstring).

    Inputs: ``clip`` ``(1, T, 3, H, W)``; ``state`` ``(1, dim, H, W)``, the
    forward state after the previous window's last output; ``gates``
    ``(1, T + 1)`` of 0/1 in the clip's dtype. ``gates[:, 0]`` is 1 when frame 0
    continues ``state``'s shot. ``gates[:, 1 + i]`` is 1 when frame ``i + 1``
    continues frame ``i``'s shot, so the backward recurrence carries into ``i``;
    a padded frame past a cut or the stream end has its gate closed.

    Outputs: ``frames`` ``(1, S, 3, sH, sW)`` and ``stateOut``, the forward
    state after frame ``S - 1`` (the next window's ``state``).
    """

    def __init__(self, net: SMoVSR, windowFrames: int, outputFrames: int):
        super().__init__()
        if not 0 < outputFrames <= windowFrames:
            raise ValueError(
                f"outputFrames must be in 1..{windowFrames}, got {outputFrames}"
            )
        self.net = net
        self.windowFrames = windowFrames
        self.outputFrames = outputFrames

    def forward(self, clip: Tensor, state: Tensor, gates: Tensor):
        net = self.net
        fwd, bwd = net.f_forward, net.f_backward
        T, S = self.windowFrames, self.outputFrames
        gates = gates.to(clip.dtype)

        def gate(i):
            return gates[:, i : i + 1].view(1, 1, 1, 1)

        forwardOut = []
        backwardIn = []
        stateOut = state
        h = state
        for i in range(T):
            x = net.fe(clip[:, i])
            if i == 0:
                out, h = fwd.gatedStep(x, h, gate(0))
            else:
                out, h = fwd(x, h)
            if i < S:
                forwardOut.append(out)
            if i == S - 1:
                stateOut = h
            backwardIn.append(net.backward_cat(torch.cat([x, out], 1)))

        backwardOut: list = [None] * S
        out, hb = bwd.step_wo_hystory(backwardIn[T - 1])
        hb = bwd.clamp(hb)
        if T - 1 < S:
            backwardOut[T - 1] = out
        for i in range(T - 2, -1, -1):
            out, hb = bwd.gatedStep(backwardIn[i], hb, gate(1 + i))
            if i < S:
                backwardOut[i] = out

        frames = []
        for i in range(S):
            residual = net.upsampler(torch.cat([forwardOut[i], backwardOut[i]], 1))
            short = F.interpolate(
                clip[:, i], scale_factor=net.scale, mode="bilinear", align_corners=False
            )
            frames.append(residual + short)
        return torch.stack(frames, 1), stateOut


def loadSMoVSR(path: str) -> SMoVSR:
    """The release ``.pth`` (a neosr checkpoint: weights under ``params``), fused, fp32, CPU."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    stateDict = checkpoint.get("params", checkpoint)
    model = SMoVSR(h_clamp=30.0, align_kernel=7)
    model.load_state_dict(stateDict, strict=True)
    return model.eval()
