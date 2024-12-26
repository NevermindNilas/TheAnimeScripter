import torch.nn as nn
from torch import Tensor

def conv(batchNorm: bool, in_planes: int, out_planes: int, kernel_size: int = 3, stride: int = 1) -> nn.Sequential:
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                bias=False,
            ),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                bias=True,
            ),
            nn.LeakyReLU(0.1, inplace=True),
        )


def predict_flow(in_planes: int) -> nn.Conv2d:
    return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=False)


def deconv(in_planes: int, out_planes: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False
        ),
        nn.LeakyReLU(0.1, inplace=True),
    )


def crop_like(input: Tensor, target: Tensor) -> Tensor:
    if input.size()[2:] == target.size()[2:]:
        return input
    else:
        return input[:, :, : target.size(2), : target.size(3)]