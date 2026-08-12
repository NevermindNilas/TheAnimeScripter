"""PWC-Net optical flow (sniklaus/pytorch-pwc as vendored upstream), with the
cupy correlation layer swapped for the pure-PyTorch port in correlation.py."""

import math

import torch

from . import correlation

backwarp_tenGrid = {}
backwarp_tenPartial = {}


def backwarp(tenInput, tenFlow):
    if str(tenFlow.size()) not in backwarp_tenGrid:
        tenHorizontal = (
            torch.linspace(-1.0, 1.0, tenFlow.shape[3])
            .view(1, 1, 1, tenFlow.shape[3])
            .expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        )
        tenVertical = (
            torch.linspace(-1.0, 1.0, tenFlow.shape[2])
            .view(1, 1, tenFlow.shape[2], 1)
            .expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
        )

        backwarp_tenGrid[str(tenFlow.size())] = torch.cat(
            [tenHorizontal, tenVertical], 1
        ).to(tenFlow.device)

    if str(tenFlow.size()) not in backwarp_tenPartial:
        backwarp_tenPartial[str(tenFlow.size())] = tenFlow.new_ones(
            [tenFlow.shape[0], 1, tenFlow.shape[2], tenFlow.shape[3]]
        )

    tenFlow = torch.cat(
        [
            tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
            tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0),
        ],
        1,
    )
    tenInput = torch.cat([tenInput, backwarp_tenPartial[str(tenFlow.size())]], 1)

    tenOutput = torch.nn.functional.grid_sample(
        input=tenInput,
        grid=(backwarp_tenGrid[str(tenFlow.size())] + tenFlow).permute(0, 2, 3, 1),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )

    tenMask = tenOutput[:, -1:, :, :]
    tenMask[tenMask > 0.999] = 1.0
    tenMask[tenMask < 1.0] = 0.0

    return tenOutput[:, :-1, :, :] * tenMask


class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()

        class Extractor(torch.nn.Module):
            def __init__(self):
                super().__init__()

                def down(inCh, outCh):
                    return torch.nn.Sequential(
                        torch.nn.Conv2d(
                            in_channels=inCh,
                            out_channels=outCh,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                        ),
                        torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                        torch.nn.Conv2d(
                            in_channels=outCh,
                            out_channels=outCh,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        ),
                        torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                        torch.nn.Conv2d(
                            in_channels=outCh,
                            out_channels=outCh,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        ),
                        torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    )

                self.netOne = down(3, 16)
                self.netTwo = down(16, 32)
                self.netThr = down(32, 64)
                self.netFou = down(64, 96)
                self.netFiv = down(96, 128)
                self.netSix = down(128, 196)

            def forward(self, tenInput):
                tenOne = self.netOne(tenInput)
                tenTwo = self.netTwo(tenOne)
                tenThr = self.netThr(tenTwo)
                tenFou = self.netFou(tenThr)
                tenFiv = self.netFiv(tenFou)
                tenSix = self.netSix(tenFiv)

                return [tenOne, tenTwo, tenThr, tenFou, tenFiv, tenSix]

        class Decoder(torch.nn.Module):
            def __init__(self, intLevel):
                super().__init__()

                intPrevious = [
                    None,
                    None,
                    81 + 32 + 2 + 2,
                    81 + 64 + 2 + 2,
                    81 + 96 + 2 + 2,
                    81 + 128 + 2 + 2,
                    81,
                    None,
                ][intLevel + 1]
                intCurrent = [
                    None,
                    None,
                    81 + 32 + 2 + 2,
                    81 + 64 + 2 + 2,
                    81 + 96 + 2 + 2,
                    81 + 128 + 2 + 2,
                    81,
                    None,
                ][intLevel + 0]

                if intLevel < 6:
                    self.netUpflow = torch.nn.ConvTranspose2d(
                        in_channels=2,
                        out_channels=2,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    )
                if intLevel < 6:
                    self.netUpfeat = torch.nn.ConvTranspose2d(
                        in_channels=intPrevious + 128 + 128 + 96 + 64 + 32,
                        out_channels=2,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    )
                if intLevel < 6:
                    self.fltBackwarp = [None, None, None, 5.0, 2.5, 1.25, 0.625, None][
                        intLevel + 1
                    ]

                def conv(inCh, outCh):
                    return torch.nn.Sequential(
                        torch.nn.Conv2d(
                            in_channels=inCh,
                            out_channels=outCh,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        ),
                        torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    )

                self.netOne = conv(intCurrent, 128)
                self.netTwo = conv(intCurrent + 128, 128)
                self.netThr = conv(intCurrent + 128 + 128, 96)
                self.netFou = conv(intCurrent + 128 + 128 + 96, 64)
                self.netFiv = conv(intCurrent + 128 + 128 + 96 + 64, 32)
                self.netSix = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=intCurrent + 128 + 128 + 96 + 64 + 32,
                        out_channels=2,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )

            def forward(self, tenFirst, tenSecond, objPrevious):
                tenFlow = None
                tenFeat = None

                if objPrevious is None:
                    tenVolume = torch.nn.functional.leaky_relu(
                        input=correlation.FunctionCorrelation(
                            tenFirst=tenFirst, tenSecond=tenSecond
                        ),
                        negative_slope=0.1,
                        inplace=False,
                    )

                    tenFeat = torch.cat([tenVolume], 1)

                else:
                    tenFlow = self.netUpflow(objPrevious["tenFlow"])
                    tenFeat = self.netUpfeat(objPrevious["tenFeat"])

                    tenVolume = torch.nn.functional.leaky_relu(
                        input=correlation.FunctionCorrelation(
                            tenFirst=tenFirst,
                            tenSecond=backwarp(
                                tenInput=tenSecond, tenFlow=tenFlow * self.fltBackwarp
                            ),
                        ),
                        negative_slope=0.1,
                        inplace=False,
                    )

                    tenFeat = torch.cat([tenVolume, tenFirst, tenFlow, tenFeat], 1)

                tenFeat = torch.cat([self.netOne(tenFeat), tenFeat], 1)
                tenFeat = torch.cat([self.netTwo(tenFeat), tenFeat], 1)
                tenFeat = torch.cat([self.netThr(tenFeat), tenFeat], 1)
                tenFeat = torch.cat([self.netFou(tenFeat), tenFeat], 1)
                tenFeat = torch.cat([self.netFiv(tenFeat), tenFeat], 1)

                tenFlow = self.netSix(tenFeat)

                return {"tenFlow": tenFlow, "tenFeat": tenFeat}

        class Refiner(torch.nn.Module):
            def __init__(self):
                super().__init__()

                self.netMain = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=81 + 32 + 2 + 2 + 128 + 128 + 96 + 64 + 32,
                        out_channels=128,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        dilation=1,
                    ),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=128,
                        out_channels=128,
                        kernel_size=3,
                        stride=1,
                        padding=2,
                        dilation=2,
                    ),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=128,
                        out_channels=128,
                        kernel_size=3,
                        stride=1,
                        padding=4,
                        dilation=4,
                    ),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=128,
                        out_channels=96,
                        kernel_size=3,
                        stride=1,
                        padding=8,
                        dilation=8,
                    ),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=96,
                        out_channels=64,
                        kernel_size=3,
                        stride=1,
                        padding=16,
                        dilation=16,
                    ),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=64,
                        out_channels=32,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        dilation=1,
                    ),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=32,
                        out_channels=2,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        dilation=1,
                    ),
                )

            def forward(self, tenInput):
                return self.netMain(tenInput)

        self.netExtractor = Extractor()

        self.netTwo = Decoder(2)
        self.netThr = Decoder(3)
        self.netFou = Decoder(4)
        self.netFiv = Decoder(5)
        self.netSix = Decoder(6)

        self.netRefiner = Refiner()

    def forward(self, tenFirst, tenSecond):
        tenFirst = self.netExtractor(tenFirst)
        tenSecond = self.netExtractor(tenSecond)

        objEstimate = self.netSix(tenFirst[-1], tenSecond[-1], None)
        objEstimate = self.netFiv(tenFirst[-2], tenSecond[-2], objEstimate)
        objEstimate = self.netFou(tenFirst[-3], tenSecond[-3], objEstimate)
        objEstimate = self.netThr(tenFirst[-4], tenSecond[-4], objEstimate)
        objEstimate = self.netTwo(tenFirst[-5], tenSecond[-5], objEstimate)

        return objEstimate["tenFlow"] + self.netRefiner(objEstimate["tenFeat"])

    def loadWeights(self, path):
        self.load_state_dict(
            {
                strKey.replace("module", "net"): tenWeight
                for strKey, tenWeight in torch.load(
                    path, map_location="cpu", weights_only=False
                ).items()
            }
        )


def estimate(tenFirst, tenSecond, netNetwork):
    """Flow from tenFirst to tenSecond, both [B, 3, H, W] in [0, 1] BGR."""
    assert tenFirst.shape[1] == tenSecond.shape[1]
    assert tenFirst.shape[2] == tenSecond.shape[2]

    intWidth = tenFirst.shape[3]
    intHeight = tenFirst.shape[2]

    tenPreprocessedFirst = tenFirst.view(-1, 3, intHeight, intWidth)
    tenPreprocessedSecond = tenSecond.view(-1, 3, intHeight, intWidth)

    intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0))
    intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))

    tenPreprocessedFirst = torch.nn.functional.interpolate(
        input=tenPreprocessedFirst,
        size=(intPreprocessedHeight, intPreprocessedWidth),
        mode="bilinear",
        align_corners=False,
    )
    tenPreprocessedSecond = torch.nn.functional.interpolate(
        input=tenPreprocessedSecond,
        size=(intPreprocessedHeight, intPreprocessedWidth),
        mode="bilinear",
        align_corners=False,
    )

    tenFlow = 20.0 * torch.nn.functional.interpolate(
        input=netNetwork(tenPreprocessedFirst, tenPreprocessedSecond),
        size=(intHeight, intWidth),
        mode="bilinear",
        align_corners=False,
    )

    tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
    tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

    global backwarp_tenGrid
    global backwarp_tenPartial
    backwarp_tenGrid = {}
    backwarp_tenPartial = {}

    return tenFlow
