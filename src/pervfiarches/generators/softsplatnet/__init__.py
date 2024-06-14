#!/usr/bin/env python

import torch
import torch.nn.functional as TF

from . import correlation  # the custom cost volume layer
from . import softsplat  # the custom softmax splatting layer

backwarp_tenGrid = {}


def backwarp(tenIn, tenFlow):
    if str(tenFlow.shape) not in backwarp_tenGrid:
        tenHor = (
            torch.linspace(
                start=-1.0,
                end=1.0,
                steps=tenFlow.shape[3],
                dtype=tenFlow.dtype,
                device=tenFlow.device,
            )
            .view(1, 1, 1, -1)
            .repeat(1, 1, tenFlow.shape[2], 1)
        )
        tenVer = (
            torch.linspace(
                start=-1.0,
                end=1.0,
                steps=tenFlow.shape[2],
                dtype=tenFlow.dtype,
                device=tenFlow.device,
            )
            .view(1, 1, -1, 1)
            .repeat(1, 1, 1, tenFlow.shape[3])
        )

        backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([tenHor, tenVer], 1).cuda()
    # end

    tenFlow = torch.cat(
        [
            tenFlow[:, 0:1, :, :] / ((tenIn.shape[3] - 1.0) / 2.0),
            tenFlow[:, 1:2, :, :] / ((tenIn.shape[2] - 1.0) / 2.0),
        ],
        1,
    )

    return torch.nn.functional.grid_sample(
        input=tenIn,
        grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )


def binary(x, threshold):
    ones = torch.ones_like(x, device=x.device)
    zeros = torch.zeros_like(x, device=x.device)
    return torch.where(x <= threshold, ones, zeros)


def calc_hole(x, flow):
    hole = binary(
        softsplat.softsplat(
            tenIn=torch.ones_like(x, device=x.device),
            tenFlow=flow,
            tenMetric=None,
            strMode="avg",
        ),
        0.5,
    )
    return hole


class Decoder(torch.nn.Module):
    def __init__(self, intChannels):
        super().__init__()

        self.netMain = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=intChannels,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(
                in_channels=128,
                out_channels=96,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(
                in_channels=96,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(
                in_channels=32,
                out_channels=2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

    # end

    def forward(self, tenOne, tenTwo, objPrevious):
        intWidth = tenOne.shape[3] and tenTwo.shape[3]
        intHeight = tenOne.shape[2] and tenTwo.shape[2]

        tenMain = None

        if objPrevious is None:
            tenVolume = correlation.FunctionCorrelation(tenOne=tenOne, tenTwo=tenTwo)

            tenMain = torch.cat([tenOne, tenVolume], 1)

        elif objPrevious is not None:
            tenForward = (
                torch.nn.functional.interpolate(
                    input=objPrevious["tenForward"],
                    size=(intHeight, intWidth),
                    mode="bilinear",
                )
                / float(objPrevious["tenForward"].shape[3])
                * float(intWidth)
            )

            tenVolume = correlation.FunctionCorrelation(
                tenOne=tenOne, tenTwo=backwarp(tenTwo, tenForward)
            )

            tenMain = torch.cat([tenOne, tenVolume, tenForward], 1)

        # end

        return {"tenForward": self.netMain(tenMain)}

    # end


class Extractor(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.netFirst = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
        )

        self.netSecond = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
        )

        self.netThird = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
        )

        self.netFourth = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=96,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(
                in_channels=96,
                out_channels=96,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
        )

        self.netFifth = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=96,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
        )

        self.netSixth = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=128,
                out_channels=192,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(
                in_channels=192,
                out_channels=192,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
        )

    # end

    def forward(self, tenInput):
        tenFirst = self.netFirst(tenInput)
        tenSecond = self.netSecond(tenFirst)
        tenThird = self.netThird(tenSecond)
        tenFourth = self.netFourth(tenThird)
        tenFifth = self.netFifth(tenFourth)
        tenSixth = self.netSixth(tenFifth)

        return [tenFirst, tenSecond, tenThird, tenFourth, tenFifth, tenSixth]

    # end


class Basic(torch.nn.Module):
    def __init__(self, strType, intChannels, boolSkip):
        super().__init__()

        if strType == "relu-conv-relu-conv":
            self.netMain = torch.nn.Sequential(
                torch.nn.PReLU(num_parameters=intChannels[0], init=0.25),
                torch.nn.Conv2d(
                    in_channels=intChannels[0],
                    out_channels=intChannels[1],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                torch.nn.PReLU(num_parameters=intChannels[1], init=0.25),
                torch.nn.Conv2d(
                    in_channels=intChannels[1],
                    out_channels=intChannels[2],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
            )

        elif strType == "conv-relu-conv":
            self.netMain = torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=intChannels[0],
                    out_channels=intChannels[1],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                torch.nn.PReLU(num_parameters=intChannels[1], init=0.25),
                torch.nn.Conv2d(
                    in_channels=intChannels[1],
                    out_channels=intChannels[2],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
            )

        # end

        self.boolSkip = boolSkip

        if boolSkip == True:
            if intChannels[0] == intChannels[2]:
                self.netShortcut = None

            elif intChannels[0] != intChannels[2]:
                self.netShortcut = torch.nn.Conv2d(
                    in_channels=intChannels[0],
                    out_channels=intChannels[2],
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                )

    def forward(self, tenInput):
        if self.boolSkip == False:
            return self.netMain(tenInput)
        # end

        if self.netShortcut is None:
            return self.netMain(tenInput) + tenInput

        elif self.netShortcut is not None:
            return self.netMain(tenInput) + self.netShortcut(tenInput)


class Downsample(torch.nn.Module):
    def __init__(self, intChannels):
        super().__init__()

        self.netMain = torch.nn.Sequential(
            torch.nn.PReLU(num_parameters=intChannels[0], init=0.25),
            torch.nn.Conv2d(
                in_channels=intChannels[0],
                out_channels=intChannels[1],
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            torch.nn.PReLU(num_parameters=intChannels[1], init=0.25),
            torch.nn.Conv2d(
                in_channels=intChannels[1],
                out_channels=intChannels[2],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        )

    # end

    def forward(self, tenInput):
        return self.netMain(tenInput)


class Upsample(torch.nn.Module):
    def __init__(self, intChannels):
        super().__init__()

        self.netMain = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            torch.nn.PReLU(num_parameters=intChannels[0], init=0.25),
            torch.nn.Conv2d(
                in_channels=intChannels[0],
                out_channels=intChannels[1],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            torch.nn.PReLU(num_parameters=intChannels[1], init=0.25),
            torch.nn.Conv2d(
                in_channels=intChannels[1],
                out_channels=intChannels[2],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        )

    # end

    def forward(self, tenInput):
        return self.netMain(tenInput)


class Encode(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.netOne = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            torch.nn.PReLU(num_parameters=32, init=0.25),
            torch.nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            torch.nn.PReLU(num_parameters=32, init=0.25),
        )

        self.netTwo = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            torch.nn.PReLU(num_parameters=64, init=0.25),
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            torch.nn.PReLU(num_parameters=64, init=0.25),
        )

        self.netThr = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=96,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            torch.nn.PReLU(num_parameters=96, init=0.25),
            torch.nn.Conv2d(
                in_channels=96,
                out_channels=96,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            torch.nn.PReLU(num_parameters=96, init=0.25),
        )

    # end

    def forward(self, tenInput):
        tenOutput = []

        tenOutput.append(self.netOne(tenInput))
        tenOutput.append(self.netTwo(tenOutput[-1]))
        tenOutput.append(self.netThr(tenOutput[-1]))

        return [torch.cat([tenInput, tenOutput[0]], 1)] + tenOutput[1:]


class Softmetric(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.netInput = torch.nn.Conv2d(
            in_channels=3,
            out_channels=12,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.netError = torch.nn.Conv2d(
            in_channels=1,
            out_channels=4,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        for intRow, intFeatures in [(0, 16), (1, 32), (2, 64), (3, 96)]:
            self.add_module(
                str(intRow) + "x0" + " - " + str(intRow) + "x1",
                Basic(
                    "relu-conv-relu-conv",
                    [intFeatures, intFeatures, intFeatures],
                    True,
                ),
            )
        # end

        for intCol in [0]:
            self.add_module(
                "0x" + str(intCol) + " - " + "1x" + str(intCol),
                Downsample([16, 32, 32]),
            )
            self.add_module(
                "1x" + str(intCol) + " - " + "2x" + str(intCol),
                Downsample([32, 64, 64]),
            )
            self.add_module(
                "2x" + str(intCol) + " - " + "3x" + str(intCol),
                Downsample([64, 96, 96]),
            )
        # end

        for intCol in [1]:
            self.add_module(
                "3x" + str(intCol) + " - " + "2x" + str(intCol),
                Upsample([96, 64, 64]),
            )
            self.add_module(
                "2x" + str(intCol) + " - " + "1x" + str(intCol),
                Upsample([64, 32, 32]),
            )
            self.add_module(
                "1x" + str(intCol) + " - " + "0x" + str(intCol),
                Upsample([32, 16, 16]),
            )
        # end

        self.netOutput = Basic("conv-relu-conv", [16, 16, 1], True)

    # end

    def forward(self, tenEncone, tenEnctwo, tenFlow):
        tenColumn = [None, None, None, None]

        tenColumn[0] = torch.cat(
            [
                self.netInput(tenEncone[0][:, 0:3, :, :]),
                self.netError(
                    torch.nn.functional.l1_loss(
                        input=tenEncone[0],
                        target=backwarp(tenEnctwo[0], tenFlow),
                        reduction="none",
                    ).mean([1], True)
                ),
            ],
            1,
        )
        tenColumn[1] = self._modules["0x0 - 1x0"](tenColumn[0])
        tenColumn[2] = self._modules["1x0 - 2x0"](tenColumn[1])
        tenColumn[3] = self._modules["2x0 - 3x0"](tenColumn[2])

        intColumn = 1
        for intRow in range(len(tenColumn) - 1, -1, -1):
            tenColumn[intRow] = self._modules[
                str(intRow)
                + "x"
                + str(intColumn - 1)
                + " - "
                + str(intRow)
                + "x"
                + str(intColumn)
            ](tenColumn[intRow])
            if intRow != len(tenColumn) - 1:
                tenUp = self._modules[
                    str(intRow + 1)
                    + "x"
                    + str(intColumn)
                    + " - "
                    + str(intRow)
                    + "x"
                    + str(intColumn)
                ](tenColumn[intRow + 1])

                if tenUp.shape[2] != tenColumn[intRow].shape[2]:
                    tenUp = torch.nn.functional.pad(
                        input=tenUp,
                        pad=[0, 0, 0, -1],
                        mode="constant",
                        value=0.0,
                    )
                if tenUp.shape[3] != tenColumn[intRow].shape[3]:
                    tenUp = torch.nn.functional.pad(
                        input=tenUp,
                        pad=[0, -1, 0, 0],
                        mode="constant",
                        value=0.0,
                    )

                tenColumn[intRow] = tenColumn[intRow] + tenUp
            # end
        # end

        return self.netOutput(tenColumn[0])


class Warp(torch.nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(
        self,
        tenEncone,
        tenEnctwo,
        tenMetricone,
        tenMetrictwo,
        tenForward,
        tenBackward,
    ):
        tenOutput = []
        tenMasks = []

        for intLevel in range(3):
            tenOne = tenEncone[intLevel]
            tenTwo = tenEnctwo[intLevel]
            H, W = tenOne.shape[-2:]
            h, w = tenForward.shape[-2:]
            if intLevel != 0:
                tenMetricone = TF.interpolate(
                    tenMetricone, size=(H, W), mode="bilinear"
                )
                tenMetrictwo = TF.interpolate(
                    tenMetrictwo, size=(H, W), mode="bilinear"
                )

                tenForward = TF.interpolate(
                    tenForward, size=(H, W), mode="bilinear"
                ) * (H / h)
                tenBackward = TF.interpolate(
                    tenBackward, size=(H, W), mode="bilinear"
                ) * (H / h)

            tenOutput.append(
                [
                    softsplat.softsplat(
                        tenIn=torch.cat([tenOne, tenMetricone], 1),
                        tenFlow=tenForward,
                        tenMetric=tenMetricone.neg().clip(-20.0, 20.0),
                        strMode="soft",
                    ),
                    softsplat.softsplat(
                        tenIn=torch.cat([tenTwo, tenMetrictwo], 1),
                        tenFlow=tenBackward,
                        tenMetric=tenMetrictwo.neg().clip(-20.0, 20.0),
                        strMode="soft",
                    ),
                ]
            )
            tenMasks.append(
                [
                    calc_hole(tenMetricone, tenForward),
                    calc_hole(tenMetrictwo, tenBackward),
                ]
            )
        tenMetricone = TF.interpolate(
            tenMetricone, size=(H // 2, W // 2), mode="bilinear"
        )
        tenMetrictwo = TF.interpolate(
            tenMetrictwo, size=(H // 2, W // 2), mode="bilinear"
        )

        tenForward = (
            TF.interpolate(tenForward, size=(H // 2, W // 2), mode="bilinear")
            * (H // 2)
            / h
        )
        tenBackward = (
            TF.interpolate(tenBackward, size=(H // 2, W // 2), mode="bilinear")
            * (H // 2)
            / h
        )
        tenMasks.append(
            [
                calc_hole(tenMetricone, tenForward),
                calc_hole(tenMetrictwo, tenBackward),
            ]
        )

        return tenOutput, tenMasks


class Flow(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.netExtractor = Extractor()

        self.netFirst = Decoder(16 + 81 + 2)
        self.netSecond = Decoder(32 + 81 + 2)
        self.netThird = Decoder(64 + 81 + 2)
        self.netFourth = Decoder(96 + 81 + 2)
        self.netFifth = Decoder(128 + 81 + 2)
        self.netSixth = Decoder(192 + 81)

    # end

    def forward(self, tenOne, tenTwo):
        intWidth = tenOne.shape[3] and tenTwo.shape[3]
        intHeight = tenOne.shape[2] and tenTwo.shape[2]

        tenOne = self.netExtractor(tenOne)
        tenTwo = self.netExtractor(tenTwo)

        objForward = None
        objBackward = None

        objForward = self.netSixth(tenOne[-1], tenTwo[-1], objForward)
        objBackward = self.netSixth(tenTwo[-1], tenOne[-1], objBackward)

        objForward = self.netFifth(tenOne[-2], tenTwo[-2], objForward)
        objBackward = self.netFifth(tenTwo[-2], tenOne[-2], objBackward)

        objForward = self.netFourth(tenOne[-3], tenTwo[-3], objForward)
        objBackward = self.netFourth(tenTwo[-3], tenOne[-3], objBackward)

        objForward = self.netThird(tenOne[-4], tenTwo[-4], objForward)
        objBackward = self.netThird(tenTwo[-4], tenOne[-4], objBackward)

        objForward = self.netSecond(tenOne[-5], tenTwo[-5], objForward)
        objBackward = self.netSecond(tenTwo[-5], tenOne[-5], objBackward)

        objForward = self.netFirst(tenOne[-6], tenTwo[-6], objForward)
        objBackward = self.netFirst(tenTwo[-6], tenOne[-6], objBackward)

        return {
            "tenForward": torch.nn.functional.interpolate(
                input=objForward["tenForward"],
                size=(intHeight, intWidth),
                mode="bilinear",
                align_corners=False,
            )
            * (float(intWidth) / float(objForward["tenForward"].shape[3])),
            "tenBackward": torch.nn.functional.interpolate(
                input=objBackward["tenForward"],
                size=(intHeight, intWidth),
                mode="bilinear",
                align_corners=False,
            )
            * (float(intWidth) / float(objBackward["tenForward"].shape[3])),
        }
