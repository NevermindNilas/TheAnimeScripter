import torch
import torch.nn as nn
import torch.nn.functional as F

from .warplayer import warp

torch.fx.wrap('warp')

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.LeakyReLU(0.2, True)
    )

class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c//2, 3, 2, 1),
            conv(c//2, c, 3, 2, 1),
            )
        self.convblock = nn.Sequential(
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
        )
        self.lastconv = nn.ConvTranspose2d(c, 5, 4, 2, 1)

    def forward(self, x, flow=None, scale=1):
        x = F.interpolate(x, scale_factor= 1. / scale, mode="bilinear")
        if flow is not None:
            flow = F.interpolate(flow, scale_factor= 1. / scale, mode="bilinear") * 1. / scale
            x = torch.cat((x, flow), 1)
        feat = self.conv0(x)
        feat = self.convblock(feat)
        tmp = self.lastconv(feat)
        tmp = F.interpolate(tmp, scale_factor=scale*2, mode="bilinear")
        flow = tmp[:, :4] * scale * 2
        mask = tmp[:, 4:5]
        return flow, mask

class IFNet(nn.Module):
    def __init__(self, scale=1, ensemble=False):
        super(IFNet, self).__init__()
        self.block0 = IFBlock(7, c=192)
        self.block1 = IFBlock(8+4, c=128)
        self.block2 = IFBlock(8+4, c=96)
        self.block3 = IFBlock(8+4, c=64)
        self.scale_list = [8/scale, 4/scale, 2/scale, 1/scale]
        self.ensemble = ensemble

    def forward(self, img0, img1, timestep):
        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        mask = None
        block = [self.block0, self.block1, self.block2, self.block3]
        for i in range(4):
            if flow is None:
                flow, mask = block[i](torch.cat((img0[:, :3], img1[:, :3], timestep), 1), None, scale=self.scale_list[i])
                if self.ensemble:
                    f1, m1 = block[i](torch.cat((img1[:, :3], img0[:, :3], 1-timestep), 1), None, scale=self.scale_list[i])
                    flow = (flow + torch.cat((f1[:, 2:4], f1[:, :2]), 1)) / 2
                    mask = (mask + (-m1)) / 2
            else:
                f0, m0 = block[i](torch.cat((warped_img0[:, :3], warped_img1[:, :3], timestep, mask), 1), flow, scale=self.scale_list[i])
                if self.ensemble:
                    f1, m1 = block[i](torch.cat((warped_img1[:, :3], warped_img0[:, :3], 1-timestep, -mask), 1), torch.cat((flow[:, 2:4], flow[:, :2]), 1), scale=self.scale_list[i])
                    f0 = (f0 + torch.cat((f1[:, 2:4], f1[:, :2]), 1)) / 2
                    m0 = (m0 + (-m1)) / 2
                flow = flow + f0
                mask = mask + m0
            mask_list.append(mask)
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged.append((warped_img0, warped_img1))
        mask_list[3] = torch.sigmoid(mask_list[3])
        return merged[3][0] * mask_list[3] + merged[3][1] * (1 - mask_list[3])
