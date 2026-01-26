"""
This is a fully functional DIRECTML Compatible version of the original UpCunet 2X Fast, aka ShuffleCugan by Sudo.
The fallin models from Renarchi also use this architecture.
"""

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F


class SEBlock(nn.Module):
    def __init__(self, in_channels: int, reduction=8, bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, in_channels // reduction, 1, 1, 0, bias=bias
        )
        self.conv2 = nn.Conv2d(
            in_channels // reduction, in_channels, 1, 1, 0, bias=bias
        )

    def forward(self, x):
        if "Half" in x.type():  # torch.HalfTensor/torch.cuda.HalfTensor
            x0 = torch.mean(x.float(), dim=(2, 3), keepdim=True).half()
        else:
            x0 = torch.mean(x, dim=(2, 3), keepdim=True)
        x0 = self.conv1(x0)
        x0 = F.relu(x0, inplace=False)
        x0 = self.conv2(x0)
        x0 = torch.sigmoid(x0)
        x = torch.mul(x, x0)
        return x

    def forward_mean(self, x, x0):
        x0 = self.conv1(x0)
        x0 = F.relu(x0, inplace=False)
        x0 = self.conv2(x0)
        x0 = torch.sigmoid(x0)
        x = torch.mul(x, x0)
        return x


class UNetConv(nn.Module):
    def __init__(
        self, in_channels: int, mid_channels: int, out_channels: int, se: bool
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, 1, 0),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Conv2d(mid_channels, out_channels, 3, 1, 0),
            nn.LeakyReLU(0.1, inplace=False),
        )
        if se:
            self.seblock = SEBlock(out_channels, reduction=8, bias=True)
        else:
            self.seblock = None

    def forward(self, x):
        z = self.conv(x)
        if self.seblock is not None:
            z = self.seblock(z)
        return z


class UNet1(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, deconv: bool):
        super().__init__()
        self.conv1 = UNetConv(in_channels, 32, 64, se=False)
        self.conv1_down = nn.Conv2d(64, 64, 2, 2, 0)
        self.conv2 = UNetConv(64, 128, 64, se=True)
        self.conv2_up = nn.ConvTranspose2d(64, 64, 2, 2, 0)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)

        if deconv:
            self.conv_bottom = nn.ConvTranspose2d(64, out_channels, 4, 2, 3)
        else:
            self.conv_bottom = nn.Conv2d(64, out_channels, 3, 1, 0)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:  # type: ignore
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x1 = x1[..., 4:-4, 4:-4]
        x2 = F.leaky_relu(x2, 0.1, inplace=False)
        x2 = self.conv2(x2)
        x2 = self.conv2_up(x2)
        x2 = F.leaky_relu(x2, 0.1, inplace=False)
        x3 = self.conv3(x1 + x2)
        x3 = F.leaky_relu(x3, 0.1, inplace=False)
        z = self.conv_bottom(x3)
        return z

    def forward_a(self, x):
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x1 = x1[..., 4:-4, 4:-4]
        x2 = F.leaky_relu(x2, 0.1, inplace=False)
        x2 = self.conv2.conv(x2)
        return x1, x2

    def forward_b(self, x1, x2):
        x2 = self.conv2_up(x2)
        x2 = F.leaky_relu(x2, 0.1, inplace=False)
        x3 = self.conv3(x1 + x2)
        x3 = F.leaky_relu(x3, 0.1, inplace=False)
        z = self.conv_bottom(x3)
        return z


class UNet2(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, deconv: bool):
        super().__init__()

        self.conv1 = UNetConv(in_channels, 32, 64, se=False)
        self.conv1_down = nn.Conv2d(64, 64, 2, 2, 0)
        self.conv2 = UNetConv(64, 64, 128, se=True)
        self.conv2_down = nn.Conv2d(128, 128, 2, 2, 0)
        self.conv3 = UNetConv(128, 256, 128, se=True)
        self.conv3_up = nn.ConvTranspose2d(128, 128, 2, 2, 0)
        self.conv4 = UNetConv(128, 64, 64, se=True)
        self.conv4_up = nn.ConvTranspose2d(64, 64, 2, 2, 0)
        self.conv5 = nn.Conv2d(64, 64, 3, 1, 0)

        if deconv:
            self.conv_bottom = nn.ConvTranspose2d(64, out_channels, 4, 2, 3)
        else:
            self.conv_bottom = nn.Conv2d(64, out_channels, 3, 1, 0)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:  # type: ignore
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, alpha: float = 1):
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x1 = x1[..., 16:-16, 16:-16]
        x2 = F.leaky_relu(x2, 0.1, inplace=False)
        x2 = self.conv2(x2)
        x3 = self.conv2_down(x2)
        x2 = x2[..., 4:-4, 4:-4]
        x3 = F.leaky_relu(x3, 0.1, inplace=False)
        x3 = self.conv3(x3)
        x3 = self.conv3_up(x3)
        x3 = F.leaky_relu(x3, 0.1, inplace=False)
        x4 = self.conv4(x2 + x3)
        x4 *= alpha
        x4 = self.conv4_up(x4)
        x4 = F.leaky_relu(x4, 0.1, inplace=False)
        x5 = self.conv5(x1 + x4)
        x5 = F.leaky_relu(x5, 0.1, inplace=False)
        z = self.conv_bottom(x5)
        return z

    def forward_a(self, x):  # conv234结尾有se
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x1 = x1[..., 16:-16, 16:-16]
        x2 = F.leaky_relu(x2, 0.1, inplace=False)
        x2 = self.conv2.conv(x2)
        return x1, x2

    def forward_b(self, x2):  # conv234结尾有se
        x3 = self.conv2_down(x2)
        x2 = x2[..., 4:-4, 4:-4]
        x3 = F.leaky_relu(x3, 0.1, inplace=False)
        x3 = self.conv3.conv(x3)
        return x2, x3

    def forward_c(self, x2, x3):  # conv234结尾有se
        x3 = self.conv3_up(x3)
        x3 = F.leaky_relu(x3, 0.1, inplace=False)
        x4 = self.conv4.conv(x2 + x3)
        return x4

    def forward_d(self, x1, x4):  # conv234结尾有se
        x4 = self.conv4_up(x4)
        x4 = F.leaky_relu(x4, 0.1, inplace=False)
        x5 = self.conv5(x1 + x4)
        x5 = F.leaky_relu(x5, 0.1, inplace=False)

        z = self.conv_bottom(x5)
        return z


def pad_reflect(x, pad):
    """
    Manual implementation of Reflection Padding logic.
    """
    if len(pad) != 4:
        raise ValueError("Only supports 2D padding")

    left, right, top, bottom = pad

    if left > 0:
        left_pad = x[..., 1 : left + 1].flip(-1)
        x = torch.cat([left_pad, x], dim=-1)

    if right > 0:
        right_pad = x[..., -(right + 1) : -1].flip(-1)
        x = torch.cat([x, right_pad], dim=-1)

    if top > 0:
        top_pad = x[..., 1 : top + 1, :].flip(-2)
        x = torch.cat([top_pad, x], dim=-2)

    if bottom > 0:
        bottom_pad = x[..., -(bottom + 1) : -1, :].flip(-2)
        x = torch.cat([x, bottom_pad], dim=-2)

    return x


class PixelUnshuffle(nn.Module):
    """
    Manual implementation of PixelUnshuffle to bypass DirectML/ONNX Runtime issues.
    """

    def __init__(self, downscale_factor):
        super().__init__()
        self.downscale_factor = downscale_factor

    def forward(self, input):
        batch_size, channels, in_height, in_width = input.shape
        out_height = in_height // self.downscale_factor
        out_width = in_width // self.downscale_factor

        input_view = input.contiguous().view(
            batch_size,
            channels,
            out_height,
            self.downscale_factor,
            out_width,
            self.downscale_factor,
        )

        channels_view = input_view.permute(0, 1, 3, 5, 2, 4)
        return channels_view.contiguous().view(
            batch_size,
            channels * self.downscale_factor * self.downscale_factor,
            out_height,
            out_width,
        )


class UpCunet2x_fast(nn.Module):
    hyperparameters = {}

    def __init__(self, *, in_channels=3, out_channels=3):
        super().__init__()
        self.unet1 = UNet1(4 * in_channels, 64, deconv=True)
        self.unet2 = UNet2(64, 64, deconv=False)
        self.ps = nn.PixelShuffle(2)
        self.conv_final = nn.Conv2d(64, 4 * out_channels, 3, 1, padding=0, bias=True)
        self.inv = PixelUnshuffle(2)

    def forward(self, x: Tensor):
        _, _, h0, w0 = x.shape
        x00 = x
        ph = ((h0 - 1) // 2 + 1) * 2
        pw = ((w0 - 1) // 2 + 1) * 2

        x = pad_reflect(x, (38, 38 + pw - w0, 38, 38 + ph - h0))

        x = self.inv(x)  # +18
        x = self.unet1.forward(x)
        x0 = self.unet2.forward(x)
        x1 = x[..., 20:-20, 20:-20]
        x = torch.add(x0, x1)
        x = self.conv_final(x)
        x = x[..., 1:-1, 1:-1]
        x = self.ps(x)
        if w0 != pw or h0 != ph:
            x = x[:, :, : h0 * 2, : w0 * 2]
        x += F.interpolate(x00, scale_factor=2, mode="nearest")
        return x
