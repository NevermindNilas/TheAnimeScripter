import torch
import torch.nn as nn
import math


from torch.nn.functional import interpolate


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        ),
        nn.LeakyReLU(0.2, True),
    )


def conv_bn(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        ),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, True),
    )


class MyPixelShuffle(nn.Module):
    def __init__(self, upscale_factor):
        super(MyPixelShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        b, c, hh, hw = x.size()
        out_channel = c // (self.upscale_factor**2)
        h = hh * self.upscale_factor
        w = hw * self.upscale_factor
        x_view = x.view(
            b, out_channel, self.upscale_factor, self.upscale_factor, hh, hw
        )
        return x_view.permute(0, 1, 4, 2, 5, 3).reshape(b, out_channel, h, w)


class Head(nn.Module):
    def __init__(self):
        super(Head, self).__init__()
        self.cnn0 = nn.Conv2d(3, 32, 3, 2, 1)
        self.cnn1 = nn.Conv2d(32, 32, 3, 1, 1)
        self.cnn2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(32, 8, 4, 2, 1)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x, feat=False):
        x0 = self.cnn0(x)
        x = self.relu(x0)
        x1 = self.cnn1(x)
        x = self.relu(x1)
        x2 = self.cnn2(x)
        x = self.relu(x2)
        x3 = self.cnn3(x)
        if feat:
            return [x0, x1, x2, x3]
        return x3


class ResConv(nn.Module):
    def __init__(self, c, dilation=1):
        super(ResConv, self).__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.relu(self.conv(x) * self.beta + x)


class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1),
            conv(c // 2, c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
        )
        self.lastconv = nn.Sequential(
            nn.ConvTranspose2d(c, 4 * 6, 4, 2, 1), nn.PixelShuffle(2)
        )

    def forward(self, x, flow=None, scale=1):
        x = interpolate(
            x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False
        )
        if flow is not None:
            flow = (
                interpolate(
                    flow, scale_factor=1.0 / scale, mode="bilinear", align_corners=False
                )
                / scale
            )
            x = torch.cat((x, flow), 1)
        feat = self.conv0(x)
        feat = self.convblock(feat)
        tmp = self.lastconv(feat)
        tmp = interpolate(tmp, scale_factor=scale, mode="bilinear", align_corners=False)
        flow = tmp[:, :4] * scale
        mask = tmp[:, 4:5]
        return flow, mask


class IFNet(nn.Module):
    def __init__(
        self,
        scale=1.0,
        ensemble=False,
        dtype=torch.float32,
        device="cuda",
        width=1920,
        height=1080,
        backWarp=None,
        tenFlow=None,
    ):
        super(IFNet, self).__init__()
        self.block0 = IFBlock(7 + 16, c=384)
        self.block1 = IFBlock(8 + 4 + 16, c=192)
        self.block2 = IFBlock(8 + 4 + 16, c=96)
        self.block3 = IFBlock(8 + 4 + 16, c=48)
        self.encode = Head()
        self.device = device
        self.dtype = dtype
        self.scaleList = [8 / scale, 4 / scale, 2 / scale, 1 / scale]
        self.ensemble = ensemble
        self.width = width
        self.height = height
        self.backWarp = backWarp
        self.tenFlow = tenFlow

        self.blocks = [self.block0, self.block1, self.block2, self.block3]
        tmp = max(64, int(64 / 1.0))
        self.pw = math.ceil(self.width / tmp) * tmp
        self.ph = math.ceil(self.height / tmp) * tmp
        self.padding = (0, self.pw - self.width, 0, self.ph - self.height)
        hMul = 2 / (self.pw - 1)
        vMul = 2 / (self.ph - 1)
        self.tenFlow = (
            torch.Tensor([hMul, vMul])
            .to(device=self.device, dtype=self.dtype)
            .reshape(1, 2, 1, 1)
        )
        self.backWarp = torch.cat(
            (
                (torch.arange(self.pw) * hMul - 1)
                .reshape(1, 1, 1, -1)
                .expand(-1, -1, self.ph, -1),
                (torch.arange(self.ph) * vMul - 1)
                .reshape(1, 1, -1, 1)
                .expand(-1, -1, -1, self.pw),
            ),
            dim=1,
        ).to(device=self.device, dtype=self.dtype)

    def forward(self, img0, img1, timestep, f0):
        imgs = torch.cat([img0, img1], dim=1)
        imgs_2 = torch.reshape(imgs, (2, 3, self.paddedHeight, self.paddedWidth))
        f1 = self.encode(img1[:, :3])
        fs = torch.cat([f0, f1], dim=1)
        fs_2 = torch.reshape(fs, (2, 8, self.paddedHeight, self.paddedWidth))
        if self.ensemble:
            fs_rev = torch.cat(torch.split(fs, [8, 8], dim=1)[::-1], dim=1)
            imgs_rev = torch.cat([img1, img0], dim=1)
        warped_img0 = img0
        warped_img1 = img1
        flows = None
        mask = None
        for block, scale in zip(self.blocks, self.scaleList):
            if flows is None:
                if self.ensemble:
                    temp = torch.cat((imgs, fs, timestep), 1)
                    temp_ = torch.cat((imgs_rev, fs_rev, 1 - timestep), 1)
                    flowss, masks = block(torch.cat((temp, temp_), 0), scale=scale)
                    flows, flows_ = torch.split(flowss, [1, 1], dim=0)
                    mask, mask_ = torch.split(masks, [1, 1], dim=0)
                    flows = (
                        flows
                        + torch.cat(torch.split(flows_, [2, 2], dim=1)[::-1], dim=1)
                    ) / 2
                    mask = (mask - mask_) / 2

                    flows_rev = torch.cat(
                        torch.split(flows, [2, 2], dim=1)[::-1], dim=1
                    )
                else:
                    temp = torch.cat((imgs, fs, timestep), 1)
                    flows, mask = block(temp, scale=scale)
            else:
                if self.ensemble:
                    temp = torch.cat(
                        (
                            wimg,  # noqa
                            wf,  # noqa
                            timestep,
                            mask,
                            (flows * (1 / scale) if scale != 1 else flows),
                        ),
                        1,
                    )
                    temp_ = torch.cat(
                        (
                            wimg_rev,  # noqa
                            wf_rev,  # noqa
                            1 - timestep,
                            -mask,
                            (flows_rev * (1 / scale) if scale != 1 else flows_rev),
                        ),
                        1,
                    )
                    fdss, masks = block(torch.cat((temp, temp_), 0), scale=scale)
                    fds, fds_ = torch.split(fdss, [1, 1], dim=0)
                    mask, mask_ = torch.split(masks, [1, 1], dim=0)
                    fds = (
                        fds + torch.cat(torch.split(fds_, [2, 2], dim=1)[::-1], dim=1)
                    ) / 2
                    mask = (mask - mask_) / 2
                else:
                    temp = torch.cat(
                        (
                            wimg,  # noqa
                            wf,  # noqa
                            timestep,
                            mask,
                            (flows * (1 / scale) if scale != 1 else flows),
                        ),
                        1,
                    )
                    fds, mask = block(temp, scale=scale)

                flows = flows + fds

                if self.ensemble:
                    flows_rev = torch.cat(
                        torch.split(flows, [2, 2], dim=1)[::-1], dim=1
                    )
            precomp = (
                (
                    self.backWarp
                    + flows.reshape((2, 2, self.paddedHeight, self.paddedWidth))
                    * self.tenFlow
                )
                .permute(0, 2, 3, 1)
                .to(dtype=self.dtype)
            )
            if scale == 1:
                warped_imgs = torch.nn.functional.grid_sample(
                    imgs_2,
                    precomp,
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=True,
                )
            else:
                warps = torch.nn.functional.grid_sample(
                    torch.cat(
                        (imgs_2.to(dtype=self.dtype), fs_2.to(dtype=self.dtype)), 1
                    ).to(dtype=self.dtype),
                    precomp,
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=True,
                )
                wimg, wf = torch.split(warps, [3, 8], dim=1)
                wimg = torch.reshape(wimg, (1, 6, self.paddedHeight, self.paddedWidth))
                wf = torch.reshape(wf, (1, 16, self.paddedHeight, self.paddedWidth))
                if self.ensemble:
                    wimg_rev = torch.cat(torch.split(wimg, [3, 3], dim=1)[::-1], dim=1)  # noqa
                    wf_rev = torch.cat(torch.split(wf, [8, 8], dim=1)[::-1], dim=1)  # noqa
        mask = torch.sigmoid(mask)
        warped_img0, warped_img1 = torch.split(warped_imgs, [1, 1])
        return (
            (warped_img0 * mask + warped_img1 * (1 - mask))[
                :, :, : self.height, : self.width
            ][0].permute(1, 2, 0)
        ), f1
