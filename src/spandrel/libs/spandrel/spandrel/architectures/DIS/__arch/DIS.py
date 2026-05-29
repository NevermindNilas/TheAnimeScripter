# DIS (Direct Image Supersampling) Architecture
# Original implementation by Kim2091
# Repository: https://github.com/Kim2091/DIS
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable conv - much faster than regular conv"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class FastResBlock(nn.Module):
    """Ultra-fast residual block with minimal operations"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.act = nn.PReLU(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        return out + residual


class LightBlock(nn.Module):
    """Lightweight block using depthwise separable convolutions"""

    def __init__(self, channels: int):
        super().__init__()
        self.dw_conv = DepthwiseSeparableConv(channels, channels, 3)
        self.act = nn.PReLU(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.dw_conv(x))


class PixelShuffleUpsampler(nn.Module):
    """Efficient upsampling using pixel shuffle (ESPCN style)"""

    def __init__(self, in_channels: int, out_channels: int, scale: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * (scale**2), 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale)
        self.act = nn.PReLU(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.pixel_shuffle(self.conv(x)))


class DIS(nn.Module):
    """Direct Image Supersampling - minimal, efficient architecture"""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_features: int = 32,
        num_blocks: int = 4,
        scale: int = 4,
        use_depthwise: bool = False,
    ):
        super().__init__()

        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.head = nn.Conv2d(in_channels, num_features, 3, padding=1)
        self.head_act = nn.PReLU(num_features)

        if use_depthwise:
            self.body = nn.Sequential(
                *[LightBlock(num_features) for _ in range(num_blocks)]
            )
        else:
            self.body = nn.Sequential(
                *[FastResBlock(num_features) for _ in range(num_blocks)]
            )

        self.fusion = nn.Conv2d(num_features, num_features, 3, padding=1)

        if scale == 4:
            self.upsampler = nn.Sequential(
                PixelShuffleUpsampler(num_features, num_features, 2),
                PixelShuffleUpsampler(num_features, num_features, 2),
            )
        elif scale == 3:
            self.upsampler = PixelShuffleUpsampler(num_features, num_features, 3)
        elif scale == 2:
            self.upsampler = PixelShuffleUpsampler(num_features, num_features, 2)
        elif scale == 1:
            self.upsampler = nn.Identity()
        else:
            raise ValueError(f"Unsupported scale factor: {scale}")

        self.tail = nn.Conv2d(num_features, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale == 1:
            base = x
        else:
            base = F.interpolate(
                x, scale_factor=self.scale, mode="bilinear", align_corners=False
            )

        feat = self.head_act(self.head(x))
        body_out = self.body(feat)
        body_out = self.fusion(body_out) + feat
        out = self.upsampler(body_out)
        out = self.tail(out)

        return out + base
