import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import trunc_normal_

class CSELayer(nn.Module):
    def __init__(self, num_channels: int = 48, reduction_ratio: int = 2) -> None:
        super().__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.squeezing = nn.Sequential(
            nn.Conv2d(num_channels, num_channels_reduced, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(num_channels_reduced, num_channels, 1, 1),
            nn.Hardsigmoid(True),
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        squeeze_tensor = torch.mean(input_tensor, dim=(2, 3), keepdim=True)
        return input_tensor * self.squeezing(squeeze_tensor)

class RMSNorm(nn.Module):
    """RMSNorm for NCHW tensors (channels_first)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim, 1, 1))
        self.offset = nn.Parameter(torch.zeros(dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = x.norm(2, dim=1, keepdim=True)
        d_x = x.size(1)
        rms_x = norm_x * (d_x ** (-0.5))
        x_normed = x / (rms_x + self.eps)
        return self.scale * x_normed + self.offset

class Conv3XC(nn.Module):
    def __init__(
        self, c_in: int, c_out: int, gain: int = 2, s: int = 1, bias: bool = True
    ) -> None:
        super().__init__()
        self.weight_concat: torch.Tensor | None = None
        self.bias_concat: torch.Tensor | None = None
        self.stride = s

        self.sk = nn.Conv2d(c_in, c_out, 1, padding=0, stride=s, bias=bias)
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_in * gain, 1, padding=0, bias=bias),
            nn.Conv2d(c_in * gain, c_out * gain, 3, stride=s, padding=0, bias=bias),
            nn.Conv2d(c_out * gain, c_out, 1, padding=0, bias=bias),
        )
        self.eval_conv = nn.Conv2d(c_in, c_out, 3, padding=1, stride=s, bias=bias)

    def update_params(self) -> None:
        w1, b1 = self.conv[0].weight.data, self.conv[0].bias.data
        w2, b2 = self.conv[1].weight.data, self.conv[1].bias.data
        w3, b3 = self.conv[2].weight.data, self.conv[2].bias.data

        w = F.conv2d(w1.flip(2, 3).permute(1, 0, 2, 3), w2, padding=2).flip(2, 3).permute(1, 0, 2, 3)
        b = (w2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b2

        self.weight_concat = F.conv2d(w.flip(2, 3).permute(1, 0, 2, 3), w3, padding=0).flip(2, 3).permute(1, 0, 2, 3)
        self.bias_concat = (w3 * b.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b3

        sk_w, sk_b = self.sk.weight.data, self.sk.bias.data
        sk_w = F.pad(sk_w, (1, 1, 1, 1))

        self.eval_conv.weight.data = self.weight_concat + sk_w
        self.eval_conv.bias.data = self.bias_concat + sk_b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_pad = F.pad(x, (1, 1, 1, 1), "constant", 0)
        return self.conv(x_pad) + self.sk(x)

class SeqConv3x3(nn.Module):
    def __init__(self, inp_planes: int, out_planes: int, depth_multiplier: float) -> None:
        super().__init__()
        self.mid_planes = int(out_planes * depth_multiplier)
        conv0 = nn.Conv2d(inp_planes, self.mid_planes, 1, padding=0)
        self.k0, self.b0 = conv0.weight, conv0.bias
        conv1 = nn.Conv2d(self.mid_planes, out_planes, 3)
        self.k1, self.b1 = conv1.weight, conv1.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y0 = F.conv2d(x, self.k0, self.b0)
        y0 = F.pad(y0, (1, 1, 1, 1), "constant", 0)
        b0_pad = self.b0.view(1, -1, 1, 1)
        y0[:, :, 0:1, :], y0[:, :, -1:, :], y0[:, :, :, 0:1], y0[:, :, :, -1:] = b0_pad, b0_pad, b0_pad, b0_pad
        return F.conv2d(y0, self.k1, self.b1)

    def rep_params(self) -> tuple[torch.Tensor, torch.Tensor]:
        rk = F.conv2d(self.k1, self.k0.permute(1, 0, 2, 3))
        rb = torch.ones(1, self.mid_planes, 3, 3, device=self.k0.device, dtype=self.k0.dtype) * self.b0.view(1, -1, 1, 1)
        rb = F.conv2d(rb, self.k1).view(-1) + self.b1
        return rk, rb

class RepConv(nn.Module):
    def __init__(self, in_dim: int = 3, out_dim: int = 32) -> None:
        super().__init__()
        self.conv1 = SeqConv3x3(in_dim, out_dim, 2)
        self.conv2 = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        self.conv3 = Conv3XC(in_dim, out_dim)
        self.conv_3x3_rep = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        self.alpha = nn.Parameter(torch.ones(3))

    def fuse(self) -> None:
        w1, b1 = self.conv1.rep_params()
        w2, b2 = self.conv2.weight.data, self.conv2.bias.data
        self.conv3.update_params()
        w3, b3 = self.conv3.eval_conv.weight.data, self.conv3.eval_conv.bias.data
        self.conv_3x3_rep.weight.data = self.alpha[0] * w1 + self.alpha[1] * w2 + self.alpha[2] * w3
        self.conv_3x3_rep.bias.data = self.alpha[0] * b1 + self.alpha[1] * b2 + self.alpha[2] * b3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self.alpha[0] * self.conv1(x) + self.alpha[1] * self.conv2(x) + self.alpha[2] * self.conv3(x)
        return self.conv_3x3_rep(x)

class OmniShift(nn.Module):
    def __init__(self, dim: int = 48) -> None:
        super().__init__()
        self.conv1x1 = nn.Conv2d(dim, dim, 1, groups=dim)
        self.conv3x3 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.conv5x5 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.alpha1 = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.alpha2 = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.alpha3 = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.alpha4 = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.conv5x5_reparam = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)

    def reparam_5x5(self) -> None:
        w1, w3, w5 = self.conv1x1.weight.data, self.conv3x3.weight.data, self.conv5x5.weight.data
        id_w = torch.ones_like(w1)
        combined_w = (self.alpha1.transpose(0, 1) * F.pad(id_w, (2, 2, 2, 2)) +
                      self.alpha2.transpose(0, 1) * F.pad(w1, (2, 2, 2, 2)) +
                      self.alpha3.transpose(0, 1) * F.pad(w3, (1, 1, 1, 1)) +
                      self.alpha4.transpose(0, 1) * w5)
        combined_b = (self.alpha2.squeeze() * self.conv1x1.bias.data +
                      self.alpha3.squeeze() * self.conv3x3.bias.data +
                      self.alpha4.squeeze() * self.conv5x5.bias.data)
        self.conv5x5_reparam.weight.data, self.conv5x5_reparam.bias.data = combined_w, combined_b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self.alpha1 * x + self.alpha2 * self.conv1x1(x) + self.alpha3 * self.conv3x3(x) + self.alpha4 * self.conv5x5(x)
        return self.conv5x5_reparam(x)

class ParPixelUnshuffle(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, down: int) -> None:
        super().__init__()
        self.pu = nn.PixelUnshuffle(down)
        self.poll = nn.Sequential(nn.MaxPool2d(down, down), RepConv(in_dim, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pu(x) + self.poll(x)

class GatedCNNBlock(nn.Module):
    def __init__(self, dim: int = 64, expansion_ratio: float = 2.0, dccm: bool = True, se: bool = False) -> None:
        super().__init__()
        self.norm = RMSNorm(dim)
        hidden = int(expansion_ratio * dim)
        self.fc1 = RepConv(dim, hidden * 2)
        self.act = nn.Mish()
        self.split_indices = [hidden, hidden - dim, dim]
        self.conv = nn.Sequential(ParPixelUnshuffle(dim, dim * 4, 2), OmniShift(dim * 4),
                                  CSELayer(dim * 4) if se else nn.Identity(), nn.PixelShuffle(2))
        self.fc2 = RepConv(hidden, dim) if dccm else nn.Conv2d(hidden, dim, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm(x)
        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=1)
        x = self.act(self.fc2(self.act(g) * torch.cat((i, self.conv(c)), dim=1)))
        return x + shortcut

class RTMoSR(nn.Module):
    def __init__(self, scale: int = 2, dim: int = 32, ffn_expansion: float = 2.0, n_blocks: int = 2,
                 unshuffle_mod: bool = False, dccm: bool = True, se: bool = True) -> None:
        super().__init__()
        self.scale = scale
        unshuffle, inner_scale = 0, scale
        if inner_scale < 4 and unshuffle_mod:
            unshuffle, inner_scale = 4 // inner_scale, 4
        self.pad = (unshuffle if unshuffle > 0 else 1) * 2

        if unshuffle == 0:
            self.to_feat = RepConv(3, dim)
        else:
            self.to_feat = nn.Sequential(nn.PixelUnshuffle(unshuffle), RepConv(3 * unshuffle**2, dim))

        self.body = nn.Sequential(*[GatedCNNBlock(dim, ffn_expansion, dccm, se) for _ in range(n_blocks)])
        self.to_img = nn.Sequential(RepConv(dim, 3 * inner_scale**2), nn.PixelShuffle(inner_scale))
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m) -> None:
        if isinstance(m, nn.Conv2d | nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2:]
        ph, pw = (self.pad - h % self.pad) % self.pad, (self.pad - w % self.pad) % self.pad
        out = self.to_feat(F.pad(x, (0, pw, 0, ph), "reflect"))
        out = self.to_img(self.body(out))[:, :, :h * self.scale, :w * self.scale]
        return out + F.interpolate(x, scale_factor=self.scale, mode="nearest")

def RTMoSR_L(**kwargs): return RTMoSR(unshuffle_mod=True, **kwargs)
def RTMoSR_UL(**kwargs): return RTMoSR(ffn_expansion=1.5, dccm=False, se=False, unshuffle_mod=True, **kwargs)