"""
pix2pix UNet256 generator used by deepDeband-f
(https://github.com/RaymondLZhou/deepDeband, MIT license).

Architecture is the ``unet_256`` generator from
``pytorch-CycleGAN-and-pix2pix`` (BSD-style license, see
THIRD_PARTY_NOTICES.md), trained with ``--norm batch``. The original
inference pipeline expects an input whose H/W are multiples of 256;
``DeepDebandF`` enforces that by reflection-padding the input to the
next multiple of 256 and cropping the output back to the requested
size, so callers can pass arbitrary-resolution frames.

Pix2pix uses tanh + [-1, 1] data range. TAS frames live in [0, 1], so
we map in/out here rather than at every call site.
"""

from __future__ import annotations

import functools

import torch
import torch.nn as nn
import torch.nn.functional as F


class _UnetSkipConnectionBlock(nn.Module):
    """Single level of the symmetric U-Net used by pix2pix unet_256."""

    def __init__(
        self,
        outer_nc: int,
        inner_nc: int,
        input_nc: int | None = None,
        submodule: nn.Module | None = None,
        outermost: bool = False,
        innermost: bool = False,
        norm_layer=nn.BatchNorm2d,
        use_dropout: bool = False,
    ) -> None:
        super().__init__()
        self.outermost = outermost
        use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(
            input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias
        )
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(
                inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1
            )
            model = [downconv, submodule, uprelu, upconv, nn.Tanh()]
        elif innermost:
            upconv = nn.ConvTranspose2d(
                inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias
            )
            model = [downrelu, downconv, uprelu, upconv, upnorm]
        else:
            upconv = nn.ConvTranspose2d(
                inner_nc * 2,
                outer_nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=use_bias,
            )
            model = [downrelu, downconv, downnorm, submodule, uprelu, upconv, upnorm]
            if use_dropout:
                model.append(nn.Dropout(0.5))

        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.outermost:
            return self.model(x)
        return torch.cat([x, self.model(x)], 1)


class UnetGenerator256(nn.Module):
    """pix2pix ``unet_256`` generator (8 downsamplings, ngf=64, BatchNorm)."""

    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 3,
        num_downs: int = 8,
        ngf: int = 64,
        norm_layer=nn.BatchNorm2d,
        use_dropout: bool = False,
    ) -> None:
        super().__init__()
        block = _UnetSkipConnectionBlock(
            ngf * 8, ngf * 8, submodule=None, norm_layer=norm_layer, innermost=True
        )
        for _ in range(num_downs - 5):
            block = _UnetSkipConnectionBlock(
                ngf * 8,
                ngf * 8,
                submodule=block,
                norm_layer=norm_layer,
                use_dropout=use_dropout,
            )
        block = _UnetSkipConnectionBlock(
            ngf * 4, ngf * 8, submodule=block, norm_layer=norm_layer
        )
        block = _UnetSkipConnectionBlock(
            ngf * 2, ngf * 4, submodule=block, norm_layer=norm_layer
        )
        block = _UnetSkipConnectionBlock(
            ngf, ngf * 2, submodule=block, norm_layer=norm_layer
        )
        self.model = _UnetSkipConnectionBlock(
            output_nc,
            ngf,
            input_nc=input_nc,
            submodule=block,
            outermost=True,
            norm_layer=norm_layer,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class DeepDebandF(nn.Module):
    """deepDeband-f wrapper: 256-pad + [0,1] <-> [-1,1] mapping + crop."""

    PAD_MULTIPLE = 256

    def __init__(self) -> None:
        super().__init__()
        self.net = UnetGenerator256(
            input_nc=3,
            output_nc=3,
            num_downs=8,
            ngf=64,
            norm_layer=functools.partial(
                nn.BatchNorm2d, affine=True, track_running_stats=True
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        m = self.PAD_MULTIPLE
        padH = (m - h % m) % m
        padW = (m - w % m) % m
        if padH or padW:
            x = F.pad(x, (0, padW, 0, padH), mode="reflect")
        y = self.net(x * 2.0 - 1.0)
        y = (y + 1.0) * 0.5
        if padH or padW:
            y = y[:, :, :h, :w]
        return y
