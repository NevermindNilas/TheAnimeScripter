from typing import Self, Literal
from torch import nn
import math
import torch
import numpy as np

from torch.nn import functional as F

from torch.nn.parameter import Parameter

SampleMods = Literal[
    "conv",
    "pixelshuffledirect",
    "pixelshuffle",
    "nearest+conv",
    "dysample",
    "pa_up",
]


class DySample(nn.Module):
    """Adapted from 'Learning to Upsample by Learning to Sample':
    https://arxiv.org/abs/2308.15085
    https://github.com/tiny-smart/dysample
    """

    def __init__(
        self,
        in_channels: int = 64,
        out_ch: int = 3,
        scale: int = 2,
        groups: int = 4,
        end_convolution: bool = True,
        end_kernel=3,
    ) -> None:
        super().__init__()

        if in_channels <= groups or in_channels % groups != 0:
            msg = "Incorrect in_channels and groups values."
            raise ValueError(msg)

        out_channels = 2 * groups * scale**2
        self.scale = scale
        self.groups = groups
        self.end_convolution = end_convolution
        if end_convolution:
            self.end_conv = nn.Conv2d(
                in_channels, out_ch, end_kernel, 1, end_kernel // 2
            )
        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        if self.training:
            nn.init.trunc_normal_(self.offset.weight, std=0.02)
            nn.init.constant_(self.scope.weight, val=0)

        self.register_buffer("init_pos", self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return (
            torch.stack(torch.meshgrid([h, h], indexing="ij"))
            .transpose(1, 2)
            .repeat(1, self.groups, 1)
            .reshape(1, -1, 1, 1)
        )

    def forward(self, x):
        offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5

        coords = (
            torch.stack(torch.meshgrid([coords_w, coords_h], indexing="ij"))
            .transpose(1, 2)
            .unsqueeze(1)
            .unsqueeze(0)
            .type(x.dtype)
            .to(x.device, non_blocking=True)
        )
        normalizer = torch.tensor(
            [W, H], dtype=x.dtype, device=x.device, pin_memory=True
        ).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1

        coords = (
            F.pixel_shuffle(coords.reshape(B, -1, H, W), self.scale)
            .view(B, 2, -1, self.scale * H, self.scale * W)
            .permute(0, 2, 3, 4, 1)
            .contiguous()
            .flatten(0, 1)
        )
        output = F.grid_sample(
            x.reshape(B * self.groups, -1, H, W),
            coords,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        ).view(B, -1, self.scale * H, self.scale * W)

        if self.end_convolution:
            output = self.end_conv(output)

        return output


class PA(nn.Module):
    def __init__(self, dim, rep) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            ConvNXC(dim, dim, (1, 1)) if rep else DOConv2d(dim, dim, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return x.mul(self.conv(x))


class UniUpsampleV4_light(nn.Sequential):
    def __init__(
        self,
        upsample: SampleMods = "pa_up",
        scale: int = 2,
        in_dim: int = 48,
        out_dim: int = 3,
        mid_dim: int = 48,
        group: int = 4,
        dysample_end_kernel=1,
        rep: bool = False,
    ) -> None:
        m = []

        if scale == 1 or upsample == "conv":
            m.append(
                ConvNXC(in_dim, out_dim, (3, 3))
                if rep
                else DOConv2d(in_dim, out_dim, 3)
            )
        elif upsample == "pixelshuffledirect":
            m.extend(
                [
                    ConvNXC(in_dim, out_dim * scale**2, (3, 3))
                    if rep
                    else DOConv2d(in_dim, out_dim * scale**2, 3),
                    nn.PixelShuffle(scale),
                ]
            )
        elif upsample == "pixelshuffle":
            m.extend(
                [
                    ConvNXC(in_dim, mid_dim, (3, 3))
                    if rep
                    else DOConv2d(in_dim, mid_dim, 3),
                    nn.LeakyReLU(inplace=True),
                ]
            )
            if (scale & (scale - 1)) == 0:  # scale = 2^n
                for _ in range(int(math.log2(scale))):
                    m.extend(
                        [
                            ConvNXC(mid_dim, mid_dim * 4, (3, 3))
                            if rep
                            else DOConv2d(mid_dim, 4 * mid_dim, 3),
                            nn.PixelShuffle(2),
                        ]
                    )
            elif scale == 3:
                m.extend(
                    [
                        ConvNXC(mid_dim, mid_dim * 9, (3, 3))
                        if rep
                        else DOConv2d(mid_dim, 9 * mid_dim, 3),
                        nn.PixelShuffle(3),
                    ]
                )
            else:
                raise ValueError(
                    f"scale {scale} is not supported. Supported scales: 2^n and 3."
                )
            m.append(
                ConvNXC(mid_dim, out_dim, (3, 3))
                if rep
                else DOConv2d(mid_dim, out_dim, 3)
            )
        elif upsample == "nearest+conv":
            if (scale & (scale - 1)) == 0:
                for _ in range(int(math.log2(scale))):
                    m.extend(
                        (
                            ConvNXC(in_dim, in_dim, (3, 3))
                            if rep
                            else DOConv2d(in_dim, in_dim, 3),
                            nn.Upsample(scale_factor=2),
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        )
                    )
                m.extend(
                    (
                        ConvNXC(in_dim, in_dim, (3, 3))
                        if rep
                        else DOConv2d(in_dim, in_dim, 3),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    )
                )
            elif scale == 3:
                m.extend(
                    (
                        ConvNXC(in_dim, in_dim, (3, 3))
                        if rep
                        else DOConv2d(in_dim, in_dim, 3),
                        nn.Upsample(scale_factor=scale),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        ConvNXC(in_dim, in_dim, (3, 3))
                        if rep
                        else DOConv2d(in_dim, in_dim, 3),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    )
                )
            else:
                raise ValueError(
                    f"scale {scale} is not supported. Supported scales: 2^n and 3."
                )
            m.append(
                ConvNXC(in_dim, out_dim, (3, 3))
                if rep
                else DOConv2d(in_dim, out_dim, 3)
            )
        elif upsample == "dysample":
            if mid_dim != in_dim:
                m.extend(
                    [
                        ConvNXC(in_dim, out_dim, (3, 3))
                        if rep
                        else DOConv2d(in_dim, mid_dim, 3),
                        nn.LeakyReLU(inplace=True),
                    ]
                )
            m.append(
                DySample(mid_dim, out_dim, scale, group, end_kernel=dysample_end_kernel)
            )

        elif upsample == "pa_up":
            if (scale & (scale - 1)) == 0:
                for _ in range(int(math.log2(scale))):
                    m.extend(
                        [
                            nn.Upsample(scale_factor=2),
                            ConvNXC(in_dim, mid_dim, (3, 3))
                            if rep
                            else DOConv2d(in_dim, mid_dim, 3),
                            PA(mid_dim, rep),
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                            ConvNXC(mid_dim, mid_dim, (3, 3))
                            if rep
                            else DOConv2d(mid_dim, mid_dim, 3),
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        ]
                    )
                    in_dim = mid_dim
            elif scale == 3:
                m.extend(
                    [
                        nn.Upsample(scale_factor=3),
                        ConvNXC(in_dim, mid_dim, (3, 3))
                        if rep
                        else DOConv2d(in_dim, mid_dim, 3),
                        PA(mid_dim, rep),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        ConvNXC(mid_dim, mid_dim, (3, 3))
                        if rep
                        else DOConv2d(mid_dim, mid_dim, 3),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    ]
                )
            else:
                raise ValueError(
                    f"scale {scale} is not supported. Supported scales: 2^n and 3."
                )
            m.append(
                ConvNXC(mid_dim, out_dim, (3, 3))
                if rep
                else DOConv2d(mid_dim, out_dim, 3)
            )
        else:
            raise ValueError(
                f"An invalid Upsample was selected. Please choose one of {SampleMods}"
            )
        super().__init__(*m)

        self.register_buffer(
            "MetaUpsample",
            torch.tensor(
                [
                    254,  # Block version, if you change something, please number from the end so that you can distinguish between authorized changes and third parties
                    list(SampleMods.__args__).index(upsample),  # UpSample method index
                    scale,
                    in_dim,
                    out_dim,
                    mid_dim,
                    group,
                    int(rep),
                ],
                dtype=torch.uint8,
            ),
        )


def get_same_padding(kernel_size, dilation):
    """
    kernel_size: (kh, kw)
    dilation: (dh, dw)
    """
    kh, kw = kernel_size
    dh, dw = dilation

    pad_h = ((kh - 1) * dh) // 2
    pad_w = ((kw - 1) * dw) // 2

    return pad_h, pad_w


def _pair(x):
    if isinstance(x, (tuple, list)):
        if len(x) < 2:
            return x[0], x[0]
        elif len(x) > 2:
            return x[0], x[1]
        return x
    else:
        return x, x


class DOConv2d(nn.Module):
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
        super(DOConv2d, self).__init__()

        kernel_size = _pair(kernel_size)

        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = 1
        self.mul = nn.Parameter(torch.ones(1) * mul, requires_grad=True)
        self.dilation = _pair(dilation)
        self.padding = (
            get_same_padding(self.kernel_size, self.dilation)
            if padding is None
            else _pair(padding)
        )
        self.groups = groups
        self._padding_repeated_twice = tuple(x for x in self.padding for _ in range(2))
        M = self.kernel_size[0]
        N = self.kernel_size[1]
        self.D_mul = M * N
        self.W = Parameter(
            torch.Tensor(out_channels, in_channels // groups, self.D_mul)
        )
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        self.dow_func = self.dow
        if M * N > 1:
            self.D = Parameter(torch.Tensor(in_channels, M * N, self.D_mul))
            init_zero = np.zeros([in_channels, M * N, self.D_mul], dtype=np.float32)
            self.D.data = torch.from_numpy(init_zero)
            eye = torch.reshape(
                torch.eye(M * N, dtype=torch.float32), (1, M * N, M * N)
            )
            d_diag = eye.repeat((in_channels, 1, self.D_mul // (M * N)))
            if self.D_mul % (M * N) != 0:
                zeros = torch.zeros([in_channels, M * N, self.D_mul % (M * N)])
                self.d_diag = Parameter(
                    torch.cat([d_diag, zeros], dim=2), requires_grad=False
                )
            else:
                self.d_diag = Parameter(d_diag, requires_grad=False)
            self.dow_func = self.dow_mult
        self.DoW_shape = (self.out_channels, self.in_channels // self.groups, M, N)
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)
        self.eval_conv = nn.Conv2d(
            in_channels,
            out_channels,
            self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        self.forward_func = self.train_forward

    def dow_mult(self):
        D = self.D + self.d_diag
        W = torch.reshape(
            self.W, (self.out_channels // self.groups, self.in_channels, self.D_mul)
        )
        DoW = torch.reshape(torch.einsum("ims,ois->oim", D, W), self.DoW_shape)
        return DoW

    def update_eval(self):
        self.eval_conv.weight.data = (
            (self.dow_func() * self.mul).clone().detach().contiguous()
        )
        self.eval_conv.bias.data = (self.bias * self.mul).clone().detach().contiguous()

    def train(self, mode: bool = True) -> Self:
        if not mode:
            self.update_eval()
            self.forward_func = self.eval_forward
        else:
            self.forward_func = self.train_forward
        return super().train(mode)

    def dow(self):
        return torch.reshape(self.W, self.DoW_shape)

    def fuse_wb(self):
        return self.dow_func(), self.bias

    def train_forward(self, x):
        return F.conv2d(
            x,
            self.dow_func() * self.mul,
            self.bias * self.mul,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def eval_forward(self, x):
        return self.eval_conv(x)

    def forward(self, x):
        return self.forward_func(x)


class ConvNXC(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: int,
        kernel_size: tuple[int, int] = (3, 3),
        gain1: int = 2,
        mul: int = 1,
    ) -> None:
        super().__init__()
        self.weight_concat = None
        self.bias_concat = None
        self.update_params_flag = False
        self.kernel_size = kernel_size
        self.c_in = c_in
        self.c_out = c_out
        gain = gain1

        self.sk = DOConv2d(
            in_channels=c_in, out_channels=c_out, kernel_size=1, padding=0, mul=0.02
        )

        pad_h = (kernel_size[0] - 1) // 2
        pad_w = (kernel_size[1] - 1) // 2
        self.padding = (pad_h, pad_w)

        self.conv = nn.Sequential(
            DOConv2d(
                in_channels=c_in,
                out_channels=c_in * gain,
                kernel_size=1,
                padding=0,
            ),
            DOConv2d(
                in_channels=c_in * gain,
                out_channels=c_out * gain,
                kernel_size=kernel_size,
                padding=0,
            ),
            DOConv2d(
                in_channels=c_out * gain,
                out_channels=c_out,
                kernel_size=1,
                padding=0,
                mul=mul,
            ),
        )
        self.eval_conv = nn.Conv2d(c_in, c_out, kernel_size, 1, self.padding)
        self.forward_func = self.forward_train

    def update_params(self) -> None:
        self.conv[0].update_eval()
        self.conv[1].update_eval()
        self.conv[2].update_eval()
        w1 = self.conv[0].eval_conv.weight.data.clone().detach()
        w2 = self.conv[1].eval_conv.weight.data.clone().detach()
        w3 = self.conv[2].eval_conv.weight.data.clone().detach()

        kh, kw = self.kernel_size
        pad_h = kh - 1
        pad_w = kw - 1

        w = (
            F.conv2d(
                w1.flip(2, 3).permute(1, 0, 2, 3),
                w2,
                padding=(pad_h, pad_w),
                stride=1,
            )
            .flip(2, 3)
            .permute(1, 0, 2, 3)
        )

        self.weight_concat = (
            F.conv2d(
                w.flip(2, 3).permute(1, 0, 2, 3),
                w3,
                padding=0,
                stride=1,
            )
            .flip(2, 3)
            .permute(1, 0, 2, 3)
        )
        self.sk.update_eval()
        sk_w = self.sk.eval_conv.weight.data.clone().detach()
        b1 = self.conv[0].eval_conv.bias.data.clone().detach()
        b2 = self.conv[1].eval_conv.bias.data.clone().detach()
        b3 = self.conv[2].eval_conv.bias.data.clone().detach()
        b = (w2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b2
        self.bias_concat = (w3 * b.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b3
        sk_b = self.sk.eval_conv.bias.data.clone().detach()

        kh, kw = self.kernel_size
        h_pixels_to_pad = (kh - 1) // 2
        w_pixels_to_pad = (kw - 1) // 2

        sk_w = F.pad(
            sk_w,
            [w_pixels_to_pad, w_pixels_to_pad, h_pixels_to_pad, h_pixels_to_pad],
        )

        self.weight_concat = self.weight_concat + sk_w
        self.eval_conv.weight.data = self.weight_concat

        self.bias_concat = self.bias_concat + sk_b
        self.eval_conv.bias.data = self.bias_concat

    def train(self, mode: bool = True) -> Self:
        super().train(mode)
        if not mode:
            self.update_params()
            self.forward_func = self.forward_eval
        else:
            self.forward_func = self.forward_train
        return self

    def forward_train(self, x: torch.Tensor) -> torch.Tensor:
        pad_h = (self.kernel_size[0] - 1) // 2
        pad_w = (self.kernel_size[1] - 1) // 2
        x_pad = F.pad(x, (pad_w, pad_w, pad_h, pad_h), "constant", 0)
        out = self.conv(x_pad) + self.sk(x)
        return out

    def forward_eval(self, x: torch.Tensor) -> torch.Tensor:
        return self.eval_conv(x)

    def forward(self, x):
        return self.forward_func(x)


class SMB(nn.Module):
    def __init__(self, in_channels, out_channels=None, rep=False):
        super(SMB, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        self.in_channels = in_channels
        if in_channels != out_channels:
            self.short = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

            with torch.no_grad():
                w = torch.zeros_like(self.short.weight)

                k = out_channels // in_channels
                remainder = out_channels % in_channels

                out_c = 0
                for c in range(in_channels):
                    copies = k + (1 if c < remainder else 0)

                    for _ in range(copies):
                        w[out_c, c, 0, 0] = 1.0
                        out_c += 1

                self.short.weight.copy_(w)
        else:
            self.short = nn.Identity()
        self.body = nn.Sequential(
            ConvNXC(in_channels, out_channels, (3, 3), mul=3)
            if rep
            else DOConv2d(in_channels, out_channels, 3, mul=3),
            nn.SiLU(True),
            ConvNXC(out_channels, out_channels, (3, 3), mul=3)
            if rep
            else DOConv2d(out_channels, out_channels, 3, mul=3),
            nn.SiLU(True),
            ConvNXC(out_channels, out_channels * 2, (3, 3), mul=3)
            if rep
            else DOConv2d(out_channels, out_channels * 2, 3, mul=3),
        )

    def forward(self, x):
        out, sim_mo = self.body(x).chunk(2, 1)
        return (out + self.short(x)).mul_(F.tanh(sim_mo))


class SMoSR(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        out_ch: int = 3,
        dim: int = 48,
        scale: int = 2,
        rep: bool = False,
        n_mb: int = 3,
        upsampler: SampleMods = "pixelshuffledirect",
        upsampler_mid_dim: int = 32,
        d_kernel: int = 3,
    ):
        super(SMoSR, self).__init__()
        self.short = nn.Conv2d(in_ch, in_ch * scale * scale, 1, 1, 0)

        with torch.no_grad():
            w = torch.zeros_like(self.short.weight)
            for c in range(in_ch):
                start_idx = c * (scale**2)
                end_idx = (c + 1) * (scale**2)

                for out_c in range(start_idx, end_idx):
                    w[out_c, c, 0, 0] = 1.0

            self.short.weight.copy_(w)
        self.blocks_1 = nn.Sequential(SMB(in_ch, dim, rep=rep), SMB(dim, rep=rep))
        self.blocks_2 = nn.Sequential(*[SMB(dim, rep=rep) for _ in range(n_mb)])
        self.end_block = nn.Sequential(
            SMB(dim, rep=rep),
            ConvNXC(dim, dim, (3, 3), mul=1) if rep else DOConv2d(dim, dim, 3),
        )
        self.upsampler = UniUpsampleV4_light(
            upsampler,
            scale,
            dim + in_ch * scale * scale,
            out_ch,
            upsampler_mid_dim,
            4,
            d_kernel,
            rep,
        )
        self.scale = scale * 2

    def forward(self, x):
        x = F.pad(x, [2, 2, 2, 2], mode="reflect")
        short = self.short(x)
        x = self.blocks_1(x)
        x = self.blocks_2(x).add_(x)
        x = self.upsampler(torch.cat([short, self.end_block(x)], 1))
        _, _, h, w = x.shape
        x = x[:, :, self.scale : h - self.scale, self.scale : w - self.scale]
        return x
