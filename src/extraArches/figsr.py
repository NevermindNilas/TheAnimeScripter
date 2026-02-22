import math
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

SampleMods = Literal[
    "conv",
    "pixelshuffledirect",
    "pixelshuffle",
    "nearest+conv",
    "dysample",
    "transpose+conv",
    "lda",
    "pa_up",
]


def ICNR(tensor, initializer, upscale_factor=2, *args, **kwargs):
    upscale_factor_squared = upscale_factor * upscale_factor
    assert tensor.shape[0] % upscale_factor_squared == 0, (
        "The size of the first dimension: "
        f"tensor.shape[0] = {tensor.shape[0]}"
        " is not divisible by square of upscale_factor: "
        f"upscale_factor = {upscale_factor}"
    )
    sub_kernel = torch.empty(
        tensor.shape[0] // upscale_factor_squared, *tensor.shape[1:]
    )
    sub_kernel = initializer(sub_kernel, *args, **kwargs)
    return sub_kernel.repeat_interleave(upscale_factor_squared, dim=0)


class DySample(nn.Module):
    """Adapted from 'Learning to Upsample by Learning to Sample'."""

    def __init__(
        self,
        in_channels: int = 64,
        out_ch: int = 3,
        scale: int = 2,
        groups: int = 4,
        end_convolution: bool = True,
        end_kernel=1,
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

    def _init_pos(self) -> Tensor:
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return (
            torch.stack(torch.meshgrid([h, h], indexing="ij"))
            .transpose(1, 2)
            .repeat(1, self.groups, 1)
            .reshape(1, -1, 1, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
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
            [W, H], dtype=x.dtype, device=x.device
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


class LayerNorm(nn.Module):
    def __init__(self, dim: int = 64, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps
        self.dim = (dim,)

    def forward(self, x):
        if x.is_contiguous(memory_format=torch.channels_last):
            return F.layer_norm(
                x.permute(0, 2, 3, 1), self.dim, self.weight, self.bias, self.eps
            ).permute(0, 3, 1, 2)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class LDA_AQU(nn.Module):
    def __init__(
        self,
        in_channels=48,
        reduction_factor=4,
        nh=1,
        scale_factor=2.0,
        k_e=3,
        k_u=3,
        n_groups=2,
        range_factor=11,
        rpb=True,
    ) -> None:
        super().__init__()
        self.k_u = k_u
        self.num_head = nh
        self.scale_factor = scale_factor
        self.n_groups = n_groups
        self.offset_range_factor = range_factor

        self.attn_dim = in_channels // (reduction_factor * self.num_head)
        self.scale = self.attn_dim**-0.5
        self.rpb = rpb
        self.hidden_dim = in_channels // reduction_factor
        self.proj_q = nn.Conv2d(
            in_channels, self.hidden_dim, kernel_size=1, stride=1, padding=0, bias=False
        )

        self.proj_k = nn.Conv2d(
            in_channels, self.hidden_dim, kernel_size=1, stride=1, padding=0, bias=False
        )

        self.group_channel = in_channels // (reduction_factor * self.n_groups)
        self.conv_offset = nn.Sequential(
            nn.Conv2d(
                self.group_channel,
                self.group_channel,
                3,
                1,
                1,
                groups=self.group_channel,
                bias=False,
            ),
            LayerNorm(self.group_channel),
            nn.SiLU(),
            nn.Conv2d(self.group_channel, 2 * k_u**2, k_e, 1, k_e // 2),
        )
        self.layer_norm = LayerNorm(in_channels)

        self.pad = int((self.k_u - 1) / 2)
        base = np.arange(-self.pad, self.pad + 1).astype(np.float32)
        base_y = np.repeat(base, self.k_u)
        base_x = np.tile(base, self.k_u)
        base_offset = np.stack([base_y, base_x], axis=1).flatten()
        base_offset = torch.tensor(base_offset).view(1, -1, 1, 1)
        self.register_buffer("base_offset", base_offset, persistent=False)

        if self.rpb:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(
                    1, self.num_head, 1, self.k_u**2, self.hidden_dim // self.num_head
                )
            )
            nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(self.conv_offset[-1].weight, 0)
        nn.init.constant_(self.conv_offset[-1].bias, 0)

    def get_offset(self, offset, Hout, Wout):
        B, _, _, _ = offset.shape
        device = offset.device
        row_indices = torch.arange(Hout, device=device)
        col_indices = torch.arange(Wout, device=device)
        row_indices, col_indices = torch.meshgrid(row_indices, col_indices, indexing="ij")
        index_tensor = torch.stack((row_indices, col_indices), dim=-1).view(
            1, Hout, Wout, 2
        )
        offset = rearrange(
            offset, "b (kh kw d) h w -> b kh h kw w d", kh=self.k_u, kw=self.k_u
        )
        offset = offset + index_tensor.view(1, 1, Hout, 1, Wout, 2)
        offset = offset.contiguous().view(B, self.k_u * Hout, self.k_u * Wout, 2)

        offset[..., 0] = 2 * offset[..., 0] / (Hout - 1) - 1
        offset[..., 1] = 2 * offset[..., 1] / (Wout - 1) - 1
        offset = offset.flip(-1)
        return offset

    def extract_feats(self, x, offset, ks=3):
        out = nn.functional.grid_sample(
            x, offset, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        out = rearrange(out, "b c (ksh h) (ksw w) -> b (ksh ksw) c h w", ksh=ks, ksw=ks)
        return out

    def forward(self, x):
        B, C, H, W = x.shape
        out_H, out_W = int(H * self.scale_factor), int(W * self.scale_factor)
        v = x
        x = self.layer_norm(x)
        q = self.proj_q(x)
        k = self.proj_k(x)

        q = torch.nn.functional.interpolate(
            q, (out_H, out_W), mode="bilinear", align_corners=True
        )
        q_off = q.view(B * self.n_groups, -1, out_H, out_W)
        pred_offset = self.conv_offset(q_off)
        offset = pred_offset.tanh().mul(self.offset_range_factor) + self.base_offset.to(
            x.dtype
        )

        k = k.view(B * self.n_groups, self.hidden_dim // self.n_groups, H, W)
        v = v.view(B * self.n_groups, C // self.n_groups, H, W)
        offset = self.get_offset(offset, out_H, out_W)
        k = self.extract_feats(k, offset=offset)
        v = self.extract_feats(v, offset=offset)

        q = rearrange(q, "b (nh c) h w -> b nh (h w) () c", nh=self.num_head)
        k = rearrange(k, "(b g) n c h w -> b (h w) n (g c)", g=self.n_groups)
        v = rearrange(v, "(b g) n c h w -> b (h w) n (g c)", g=self.n_groups)
        k = rearrange(k, "b n1 n (nh c) -> b nh n1 n c", nh=self.num_head)
        v = rearrange(v, "b n1 n (nh c) -> b nh n1 n c", nh=self.num_head)

        if self.rpb:
            k = k + self.relative_position_bias_table

        q = q * self.scale
        attn = q @ k.transpose(-1, -2)
        attn = attn.softmax(dim=-1)
        out = attn @ v

        out = rearrange(out, "b nh (h w) t c -> b (nh c) (t h) w", h=out_H)
        return out


class PA(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.Sigmoid())

    def forward(self, x):
        return x.mul(self.conv(x))


class UniUpsampleV3(nn.Sequential):
    def __init__(
        self,
        upsample: SampleMods = "pa_up",
        scale: int = 2,
        in_dim: int = 48,
        out_dim: int = 3,
        mid_dim: int = 48,
        group: int = 4,
        dysample_end_kernel=1,
    ) -> None:
        m = []

        if scale == 1 or upsample == "conv":
            m.append(nn.Conv2d(in_dim, out_dim, 3, 1, 1))
        elif upsample == "pixelshuffledirect":
            m.extend(
                [nn.Conv2d(in_dim, out_dim * scale**2, 3, 1, 1), nn.PixelShuffle(scale)]
            )
        elif upsample == "pixelshuffle":
            m.extend([nn.Conv2d(in_dim, mid_dim, 3, 1, 1), nn.LeakyReLU(inplace=True)])
            if (scale & (scale - 1)) == 0:
                for _ in range(int(math.log2(scale))):
                    m.extend(
                        [nn.Conv2d(mid_dim, 4 * mid_dim, 3, 1, 1), nn.PixelShuffle(2)]
                    )
            elif scale == 3:
                m.extend([nn.Conv2d(mid_dim, 9 * mid_dim, 3, 1, 1), nn.PixelShuffle(3)])
            else:
                raise ValueError(
                    f"scale {scale} is not supported. Supported scales: 2^n and 3."
                )
            m.append(nn.Conv2d(mid_dim, out_dim, 3, 1, 1))
        elif upsample == "nearest+conv":
            if (scale & (scale - 1)) == 0:
                for _ in range(int(math.log2(scale))):
                    m.extend(
                        (
                            nn.Conv2d(in_dim, in_dim, 3, 1, 1),
                            nn.Upsample(scale_factor=2),
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        )
                    )
                m.extend(
                    (
                        nn.Conv2d(in_dim, in_dim, 3, 1, 1),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    )
                )
            elif scale == 3:
                m.extend(
                    (
                        nn.Conv2d(in_dim, in_dim, 3, 1, 1),
                        nn.Upsample(scale_factor=scale),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        nn.Conv2d(in_dim, in_dim, 3, 1, 1),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    )
                )
            else:
                raise ValueError(
                    f"scale {scale} is not supported. Supported scales: 2^n and 3."
                )
            m.append(nn.Conv2d(in_dim, out_dim, 3, 1, 1))
        elif upsample == "dysample":
            if mid_dim != in_dim:
                m.extend(
                    [nn.Conv2d(in_dim, mid_dim, 3, 1, 1), nn.LeakyReLU(inplace=True)]
                )
            m.append(
                DySample(mid_dim, out_dim, scale, group, end_kernel=dysample_end_kernel)
            )
        elif upsample == "transpose+conv":
            if scale == 2:
                m.append(nn.ConvTranspose2d(in_dim, out_dim, 4, 2, 1))
            elif scale == 3:
                m.append(nn.ConvTranspose2d(in_dim, out_dim, 3, 3, 0))
            elif scale == 4:
                m.extend(
                    [
                        nn.ConvTranspose2d(in_dim, in_dim, 4, 2, 1),
                        nn.GELU(),
                        nn.ConvTranspose2d(in_dim, out_dim, 4, 2, 1),
                    ]
                )
            else:
                raise ValueError(
                    f"scale {scale} is not supported. Supported scales: 2, 3, 4"
                )
            m.append(nn.Conv2d(out_dim, out_dim, 3, 1, 1))
        elif upsample == "lda":
            if mid_dim != in_dim:
                m.extend(
                    [nn.Conv2d(in_dim, mid_dim, 3, 1, 1), nn.LeakyReLU(inplace=True)]
                )
            m.append(LDA_AQU(mid_dim, scale_factor=scale))
            m.append(nn.Conv2d(mid_dim, out_dim, 3, 1, 1))
        elif upsample == "pa_up":
            if (scale & (scale - 1)) == 0:
                for _ in range(int(math.log2(scale))):
                    m.extend(
                        [
                            nn.Upsample(scale_factor=2),
                            nn.Conv2d(in_dim, mid_dim, 3, 1, 1),
                            PA(mid_dim),
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                            nn.Conv2d(mid_dim, mid_dim, 3, 1, 1),
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        ]
                    )
                    in_dim = mid_dim
            elif scale == 3:
                m.extend(
                    [
                        nn.Upsample(scale_factor=3),
                        nn.Conv2d(in_dim, mid_dim, 3, 1, 1),
                        PA(mid_dim),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        nn.Conv2d(mid_dim, mid_dim, 3, 1, 1),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    ]
                )
            else:
                raise ValueError(
                    f"scale {scale} is not supported. Supported scales: 2^n and 3."
                )
            m.append(nn.Conv2d(mid_dim, out_dim, 3, 1, 1))
        else:
            raise ValueError(
                f"An invalid Upsample was selected. Please choose one of {SampleMods}"
            )
        super().__init__(*m)

        self.register_buffer(
            "MetaUpsample",
            torch.tensor(
                [
                    3,
                    list(SampleMods.__args__).index(upsample),
                    scale,
                    in_dim,
                    out_dim,
                    mid_dim,
                    group,
                ],
                dtype=torch.uint8,
            ),
        )


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.offset = nn.Parameter(torch.zeros(dim))
        self.eps = nn.Parameter(torch.Tensor(torch.ones(1) * eps), requires_grad=False)
        self.rms = nn.Parameter(
            torch.Tensor(torch.ones(1) * (dim**-0.5)), requires_grad=False
        )

    def forward(self, x: Tensor) -> Tensor:
        norm_x = torch.addcmul(self.eps, x.norm(2, dim=1, keepdim=True), self.rms)
        return torch.addcmul(
            self.offset[:, None, None], x.div(norm_x), self.scale[:, None, None]
        )


class CustomRFFT2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        y = torch.fft.rfft2(x, dim=(2, 3), norm="ortho")
        return torch.view_as_real(y)

    @staticmethod
    def symbolic(g, x: torch.Value):
        shp = g.op("Shape", x)
        iH = g.op("Constant", value_t=torch.tensor([2], dtype=torch.int64))
        iW = g.op("Constant", value_t=torch.tensor([3], dtype=torch.int64))
        nH = g.op("Gather", shp, iH, axis_i=0)
        nW = g.op("Gather", shp, iW, axis_i=0)

        axes_last = g.op("Constant", value_t=torch.tensor([4], dtype=torch.int64))
        x_u = g.op("Unsqueeze", x, axes_last)
        zero = g.op("Sub", x_u, x_u)
        x_c = g.op("Concat", x_u, zero, axis_i=4)

        Hf = g.op("Cast", nH, to_i=torch.onnx.TensorProtoDataType.FLOAT)
        Wf = g.op("Cast", nW, to_i=torch.onnx.TensorProtoDataType.FLOAT)

        y = g.op("DFT", x_c, nW, axis_i=3, onesided_i=1)
        y = g.op("Div", y, g.op("Sqrt", Wf))

        y = g.op("DFT", y, nH, axis_i=2, onesided_i=0)
        y = g.op("Div", y, g.op("Sqrt", Hf))

        return y


class CustomIRFFT2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_ri: torch.Tensor):
        x_c = torch.view_as_complex(x_ri)
        return torch.fft.irfft2(x_c, dim=(2, 3), norm="ortho")

    @staticmethod
    def symbolic(g, x: torch.Value):
        shp = g.op("Shape", x)
        iH = g.op("Constant", value_t=torch.tensor([2], dtype=torch.int64))
        iWr = g.op("Constant", value_t=torch.tensor([3], dtype=torch.int64))
        nH = g.op("Gather", shp, iH, axis_i=0)
        nWr = g.op("Gather", shp, iWr, axis_i=0)

        one = g.op("Constant", value_t=torch.tensor(1, dtype=torch.int64))
        two = g.op("Constant", value_t=torch.tensor(2, dtype=torch.int64))
        nW = g.op("Mul", g.op("Sub", nWr, one), two)
        Hf = g.op("Cast", nH, to_i=torch.onnx.TensorProtoDataType.FLOAT)
        Wf = g.op("Cast", nW, to_i=torch.onnx.TensorProtoDataType.FLOAT)

        yH = g.op("DFT", x, nH, axis_i=2, inverse_i=1, onesided_i=0)
        yH = g.op("Mul", yH, g.op("Sqrt", Hf))

        start = g.op("Sub", nWr, two)
        start = g.op(
            "Squeeze",
            start,
            g.op("Constant", value_t=torch.tensor([0], dtype=torch.int64)),
        )
        limit = g.op("Constant", value_t=torch.tensor(0, dtype=torch.int64))
        step = g.op("Constant", value_t=torch.tensor(-1, dtype=torch.int64))
        idx_r = g.op("Range", start, limit, step)

        mirW = g.op("Gather", yH, idx_r, axis_i=3)
        maskW = g.op("Constant", value_t=torch.tensor([1.0, -1.0], dtype=torch.float32))
        maskW = g.op(
            "Unsqueeze",
            maskW,
            g.op("Constant", value_t=torch.tensor([0, 1, 2, 3], dtype=torch.int64)),
        )
        mirWc = g.op("Mul", mirW, maskW)
        x_full = g.op("Concat", yH, mirWc, axis_i=3)

        y = g.op("DFT", x_full, nW, axis_i=3, inverse_i=1, onesided_i=0)
        y = g.op("Mul", y, g.op("Sqrt", Wf))

        s0 = g.op("Constant", value_t=torch.tensor([0], dtype=torch.int64))
        s1 = g.op("Constant", value_t=torch.tensor([1], dtype=torch.int64))
        axC = g.op("Constant", value_t=torch.tensor([4], dtype=torch.int64))
        y = g.op("Slice", y, s0, s1, axC)
        y = g.op("Squeeze", y, axC)

        return y


class CustomRfft2Wrap(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        if self.training:
            y = torch.fft.rfft2(x, dim=(2, 3), norm="ortho")
            return torch.view_as_real(y)
        else:
            return CustomRFFT2().apply(x)


class CustomIrfft2Wrap(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        if self.training:
            x_c = torch.view_as_complex(x)
            return torch.fft.irfft2(x_c, dim=(2, 3), norm="ortho")
        else:
            return CustomIRFFT2().apply(x)


class FourierUnit(nn.Module):
    def __init__(self, in_channels: int = 48, out_channels: int = 48) -> None:
        super().__init__()
        self.rn = RMSNorm(out_channels * 2)
        self.post_norm = RMSNorm(out_channels)

        self.fdc = nn.Conv2d(
            in_channels=in_channels * 2,
            out_channels=out_channels * 2,
            kernel_size=1,
            bias=True,
        )

        self.fpe = nn.Conv2d(
            in_channels=in_channels * 2,
            out_channels=in_channels * 2,
            kernel_size=3,
            padding=1,
            groups=in_channels * 2,
            bias=True,
        )
        self.gelu = nn.GELU()
        self.irfft2 = CustomIrfft2Wrap()
        self.rfft2 = CustomRfft2Wrap()

    def forward(self, x: Tensor) -> Tensor:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        b, c, h, w = x.shape
        ffted = self.rfft2(x)
        ffted = ffted.permute(0, 4, 1, 2, 3).contiguous()
        ffted = ffted.view(b, c * 2, h, -1).to(orig_dtype)
        ffted = self.rn(ffted)
        ffted = self.fpe(ffted) + ffted
        ffted = self.fdc(ffted)
        ffted = self.gelu(ffted)
        ffted = ffted.view(b, c, 2, h, -1).permute(0, 1, 3, 4, 2).contiguous().float()
        out = self.irfft2(ffted)
        out = self.post_norm(out.to(orig_dtype))
        return out


class InceptionConv2d(nn.Module):
    def __init__(
        self,
        fu_dim: int = 24,
        gc: int = 8,
        square_kernel_size: int = 13,
        band_kernel_size: int = 17,
    ) -> None:
        super().__init__()

        self.fu = FourierUnit(fu_dim, fu_dim)
        self.convhw = nn.Conv2d(
            gc, gc, square_kernel_size, padding=square_kernel_size // 2
        )
        self.convw = nn.Conv2d(
            gc,
            gc,
            kernel_size=(1, band_kernel_size),
            padding=(0, band_kernel_size // 2),
        )
        self.convh = nn.Conv2d(
            gc,
            gc,
            kernel_size=(band_kernel_size, 1),
            padding=(band_kernel_size // 2, 0),
        )

    def forward(
        self, x: Tensor, x_hw: Tensor, x_w: Tensor, xh: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.fu(x), self.convhw(x_hw), self.convw(x_w), self.convh(xh)


class GatedCNNBlock(nn.Module):
    def __init__(
        self,
        dim: int = 64,
        expansion_ratio: float = 8 / 3,
        gc: int = 8,
        square_kernel_size: int = 13,
        band_kernel_size: int = 17,
    ) -> None:
        super().__init__()
        hidden = int(expansion_ratio * dim) // 8 * 8
        self.norm = RMSNorm(dim)
        self.fc1 = nn.Conv2d(dim, hidden * 2, 3, 1, 1)
        self.act = nn.SiLU()
        self.split_indices = [hidden, hidden - dim, dim - gc * 3, gc, gc, gc]
        self.conv = InceptionConv2d(
            dim - gc * 3, gc, square_kernel_size, band_kernel_size
        )
        self.fc2 = nn.Conv2d(hidden, dim, 3, 1, 1)

    def gated_forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        x = self.fc1(x)
        g, i, c, c_hw, c_w, c_h = torch.split(x, self.split_indices, dim=1)
        c, c_hw, c_w, c_h = self.conv(c, c_hw, c_w, c_h)
        x = self.fc2(self.act(g) * torch.cat((i, c, c_hw, c_w, c_h), dim=1))
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self.gated_forward(x) + x


class FIGSR(nn.Module):
    """Fourier Inception Gated Super Resolution."""

    def __init__(
        self,
        in_nc: int = 3,
        dim: int = 48,
        expansion_ratio: float = 8 / 3,
        scale: int = 4,
        out_nc: int = 3,
        upsampler: SampleMods = "pixelshuffledirect",
        mid_dim: int = 32,
        n_blocks: int = 24,
        gc: int = 8,
        square_kernel_size: int = 13,
        band_kernel_size: int = 17,
        **kwargs,
    ) -> None:
        super().__init__()
        self.in_to_dim = nn.Conv2d(in_nc, dim, 3, 1, 1)
        self.pad = 2
        self.gfisr_body_half = nn.Sequential(
            *[
                GatedCNNBlock(
                    dim, expansion_ratio, gc, square_kernel_size, band_kernel_size
                )
                for _ in range(n_blocks // 2)
            ]
        )
        self.gfisr_body_half_2 = nn.Sequential(
            *[
                GatedCNNBlock(
                    dim, expansion_ratio, gc, square_kernel_size, band_kernel_size
                )
                for _ in range(n_blocks - n_blocks // 2)
            ]
            + [nn.Conv2d(dim, dim, 3, 1, 1)]
        )
        self.cat_to_dim = nn.Conv2d(dim * 3, dim, 1)
        self.upscale = UniUpsampleV3(
            upsampler, scale, dim, out_nc, mid_dim, dysample_end_kernel=3
        )
        if upsampler == "pixelshuffledirect":
            weight = ICNR(
                self.upscale[0].weight,
                initializer=nn.init.kaiming_normal_,
                upscale_factor=scale,
            )
            self.upscale[0].weight.data.copy_(weight)

        self.scale = scale
        self.shift = nn.Parameter(torch.ones(1, 3, 1, 1) * 0.5, requires_grad=True)
        self.scale_norm = nn.Parameter(torch.ones(1, 3, 1, 1) / 6, requires_grad=True)

    def load_state_dict(self, state_dict, strict=True, assign=True):
        state_dict["upscale.MetaUpsample"] = self.upscale.MetaUpsample
        return super().load_state_dict(state_dict, strict, assign)

    def forward(self, x: Tensor) -> Tensor:
        x = (x - self.shift) / self.scale_norm

        _, _, H, W = x.shape
        mod_pad_h = (self.pad - H % self.pad) % self.pad
        mod_pad_w = (self.pad - W % self.pad) % self.pad
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")

        x = self.in_to_dim(x)
        x0 = self.gfisr_body_half(x)
        x1 = self.gfisr_body_half_2(x0)

        x = self.cat_to_dim(torch.cat([x1, x, x0], dim=1))
        x = self.upscale(x)[:, :, : H * self.scale, : W * self.scale]
        return x * self.scale_norm + self.shift
