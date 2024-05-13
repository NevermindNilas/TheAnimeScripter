import torch
import torch.nn as nn
import torch.nn.functional as F

def warp(tenInput, tenFlow, base_grid, multiplier):
    tenFlow = base_grid + tenFlow * multiplier

    grid = tenFlow.permute(0, 2, 3, 1)

    return torch.nn.functional.grid_sample(input=tenInput, grid=grid, mode='bilinear', padding_mode='border', align_corners=True)

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.LeakyReLU(0.2, True)
    )

class ResConv(nn.Module):
    def __init__(self, c, dilation=1):
        super(ResConv, self).__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1\
)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.relu(self.conv(x) * self.beta + x)

class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c//2, 3, 2, 1),
            conv(c//2, c, 3, 2, 1),
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
            nn.ConvTranspose2d(c, 4*6, 4, 2, 1),
            nn.PixelShuffle(2)
        )

    def forward(self, x, flow, scale):
        x = F.interpolate(
            x, scale_factor=scale,
            mode="bilinear", align_corners=False
        )

        if flow is not None:
            flow = F.interpolate(
                flow, scale_factor=scale,
                mode="bilinear", align_corners=False
            )
            flow *= scale

            x = torch.cat((x, flow), 1)

        feat = self.conv0(x)
        feat = self.convblock(feat)

        tmp = self.lastconv(feat)
        tmp = F.interpolate(tmp, scale_factor=1/scale, mode="bilinear", align_corners=False)

        flow, mask, _ = torch.split(tmp, split_size_or_sections=[4, 1, 1], dim=1)
        return flow, mask

class IFNet(nn.Module):
    def __init__(self):
        super(IFNet, self).__init__()
        self.block0 = IFBlock(7, c=192)
        self.block1 = IFBlock(8+4, c=128)
        self.block2 = IFBlock(8+4, c=96)
        self.block3 = IFBlock(8+4, c=64)
        # self.contextnet = Contextnet(device) # fastmode
        # self.unet = Unet() # fastmode

    def forward(self, x):
        # img0, img1, timestep, base_grid, multiplier
        h, w = 128, 128
        img0, img1, timestep, base_grid, multiplier = torch.split(x, split_size_or_sections=[3, 3, 1, 2, 2], dim=1)
        img = torch.cat((img0, img1), dim=0)

        # iter 0
        flow, mask = self.block0(
            torch.cat((img0, img1, timestep), 1),
            None,
            scale=1/8
        )
        flow = flow * 8
        warped_img = warp(img, flow.reshape(2, 2, h, w), base_grid, multiplier).reshape(1, 6, h, w)

        # iter 1
        f0, m0 = self.block1(
            torch.cat((warped_img, timestep, mask), 1),
            flow,
            scale=1/4
        )
        flow += f0 * 4
        warped_img = warp(img, flow.reshape(2, 2, h, w), base_grid, multiplier).reshape(1, 6, h, w)
        mask = mask + m0

        # iter 2
        f0, m0 = self.block2(
            torch.cat((warped_img, timestep, mask), 1),
            flow,
            scale=1/2
        )
        flow += f0 * 2
        warped_img = warp(img, flow.reshape(2, 2, h, w), base_grid, multiplier).reshape(1, 6, h, w)
        mask = mask + m0

        # iter 3
        f0, m0 = self.block3(
            torch.cat((warped_img, timestep, mask), 1),
            flow,
            scale=1.0
        )
        flow += f0 * 1.0
        warped_img = warp(img, flow.reshape(2, 2, h, w), base_grid, multiplier)
        mask = mask + m0

        # epilogue
        mask = torch.sigmoid(mask)
        warped_img0, warped_img1 = torch.split(warped_img, split_size_or_sections=[1, 1], dim=0)
        return warped_img0 * mask + warped_img1 * (1 - mask)
