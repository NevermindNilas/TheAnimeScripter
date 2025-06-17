import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from .warplayer import warp
from .refine import *

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )


class Head(nn.Module):
    def __init__(self, in_planes, scale, c, in_else=17):
        super(Head, self).__init__()
        self.upsample = nn.Sequential(nn.PixelShuffle(2), nn.PixelShuffle(2))
        self.scale = scale
        self.conv = nn.Sequential(
                                  conv(in_planes*2 // (4*4) + in_else, c),
                                  conv(c, c),
                                  conv(c, 5),
                                  )  

    def forward(self, motion_feature, x, flow): 
        motion_feature = self.upsample(motion_feature) 
        if self.scale != 4:
            x = F.interpolate(x, scale_factor = 4. / self.scale, mode="bilinear", align_corners=False)
        if flow != None:
            if self.scale != 4:
                flow = F.interpolate(flow, scale_factor = 4. / self.scale, mode="bilinear", align_corners=False) * 4. / self.scale
            x = torch.cat((x, flow), 1)
        x = self.conv(torch.cat([motion_feature, x], 1))
        if self.scale != 4:
            x = F.interpolate(x, scale_factor = self.scale // 4, mode="bilinear", align_corners=False)
            flow = x[:, :4] * (self.scale // 4)
        else:
            flow = x[:, :4]
        mask = x[:, 4:5]
        return flow, mask

class IFBlock(nn.Module):
    def __init__(self, in_planes, c, scale):
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
        self.scale = scale

    def forward(self, x, flow):
        scale = self.scale
        if scale != 1:
            x = F.interpolate(x, scale_factor = 1. / scale, mode="bilinear", align_corners=False)
            flow = F.interpolate(flow, scale_factor = 1. / scale, mode="bilinear", align_corners=False) * 1. / scale
        x = torch.cat((x, flow), 1)
        x = self.conv0(x)
        x = self.convblock(x) + x
        tmp = self.lastconv(x)
        tmp = F.interpolate(tmp, scale_factor = scale * 2, mode="bilinear", align_corners=False)
        flow = tmp[:, :4] * scale * 2
        mask = tmp[:, 4:5]
        return flow, mask
    
class MultiScaleFlow(nn.Module):
    def __init__(self, backbone, **kargs):
        super(MultiScaleFlow, self).__init__()
        self.flow_num_stage = len(kargs['hidden_dims'])
        self.local_num = kargs['local_num']
        self.feature_bone = backbone
        self.block = nn.ModuleList([Head( kargs['embed_dims'][-1-i], 
                            kargs['scales'][-1-i], 
                            kargs['hidden_dims'][-1-i],
                            7 if i==0 else 18) 
                            for i in range(self.flow_num_stage)])
        self.local_block = nn.ModuleList([IFBlock(17, c=kargs['local_hidden_dims'], scale=2-i) for i in range(self.local_num)])
        self.unet = Unet(kargs['c'] * 2, kargs['M'])

    def warp_features(self, xs, flow):
        y0 = []
        y1 = []
        B = xs[0].size(0) // 2
        for x in xs:
            y0.append(warp(x[:B], flow[:, 0:2]))
            y1.append(warp(x[B:], flow[:, 2:4]))
            flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
        return y0, y1

    def calculate_flow(self, imgs, timestep, local=False, af=None):
        img0, img1 = imgs[:, :3], imgs[:, 3:6]
        B = img0.size(0)
        flow, mask = None, None
        if af is None:
            af = self.feature_bone(img0, img1)
        timestep = (img0[:, :1].clone() * 0 + 1) * timestep
        for i in range(self.flow_num_stage):
            if flow != None:
                warped_img0 = warp(img0, flow[:, :2])
                warped_img1 = warp(img1, flow[:, 2:4])
                flow_, mask_ = self.block[i](
                    torch.cat([af[-1-i][:B],af[-1-i][B:]],1),
                    torch.cat((img0, img1, warped_img0, warped_img1, mask, timestep), 1),
                    flow
                    )
                flow = flow + flow_
                mask = mask + mask_
            else:
                flow, mask = self.block[i](
                    torch.cat([af[-1-i][:B],af[-1-i][B:]],1),
                    torch.cat((img0, img1, timestep), 1),
                    None
                    )

        if local:
            for i in range(self.local_num):
                warped_img0 = warp(img0, flow[:, :2])
                warped_img1 = warp(img1, flow[:, 2:4])

                flow_d, mask_d = self.local_block[i](torch.cat((img0, img1, warped_img0, warped_img1, mask), 1), flow)
                flow = flow + flow_d
                mask = mask + mask_d

        return flow, mask

    def coraseWarp_and_Refine(self, imgs, af, flow, mask):
        img0, img1 = imgs[:, :3], imgs[:, 3:6]
        warped_img0 = warp(img0, flow[:, :2])
        warped_img1 = warp(img1, flow[:, 2:4])
        c0, c1 = self.warp_features(af, flow)
        tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)
        res = tmp[:, :3] * 2 - 1
        mask_ = torch.sigmoid(mask)
        merged = warped_img0 * mask_ + warped_img1 * (1 - mask_)
        pred = torch.clamp(merged + res, 0, 1)
        return pred


    def forward(self, x, local=False, timestep=0.5, scale=0):
        if scale > 0: 
            x_o = x
            x = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False)

        img0, img1 = x[:, :3], x[:, 3:6]
        B = x.size(0)
        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        af = self.feature_bone(img0, img1)
        timestep = (x[:, :1].clone() * 0 + 1) * (timestep.float() if type(timestep) is not float else timestep)
        for i in range(self.flow_num_stage):
            if flow != None:
                flow_d, mask_d = self.block[i]( torch.cat([af[-1-i][:B],af[-1-i][B:]],1), 
                                                torch.cat((img0, img1, warped_img0, warped_img1, mask, timestep), 1), flow)
                flow = flow + flow_d
                mask = mask + mask_d
            else:
                flow, mask = self.block[i]( torch.cat([af[-1-i][:B],af[-1-i][B:]],1), 
                                            torch.cat((img0, img1, timestep), 1), None)
            mask_list.append(torch.sigmoid(mask))
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged.append(warped_img0 * mask_list[i] + warped_img1 * (1 - mask_list[i]))
        
        if scale>0:
            img0, img1 = x_o[:, :3], x_o[:, 3:6]
            af1 = self.feature_bone(img0, img1)
            scale = img0.shape[3] / flow.shape[3]
            flow = F.interpolate(flow, scale_factor = scale, mode="bilinear", align_corners=False) * scale
            mask = F.interpolate(mask, scale_factor = scale, mode="bilinear", align_corners=False)
            mask_ = torch.sigmoid(mask)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged.append(warped_img0 * mask_ + warped_img1 * (1 - mask_))

        if local:
            flow_list = []
            merged = []
            mask_list = []
            
            for i in range(self.local_num):
                flow_d, mask_d = self.local_block[i](torch.cat((img0, img1, warped_img0, warped_img1, mask), 1), flow)
                flow = flow + flow_d
                mask = mask + mask_d

                mask_list.append(torch.sigmoid(mask))
                flow_list.append(flow)
                warped_img0 = warp(img0, flow[:, :2])
                warped_img1 = warp(img1, flow[:, 2:4])
                merged.append(warped_img0 * mask_list[i] + warped_img1 * (1 - mask_list[i]))
        
        if scale: 
            c0, c1 = self.warp_features(af1, flow)
        else:
            c0, c1 = self.warp_features(af, flow)
        tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)
        res = tmp[:, :3] * 2 - 1
        pred = torch.clamp(merged[-1] + res, 0, 1)
        return flow_list, mask_list, merged, pred