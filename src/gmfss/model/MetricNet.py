import torch
import torch.nn as nn
import torch.nn.functional as F

from .gmflow.geometry import forward_backward_consistency_check
from .util import MyPReLU
from .warplayer import get_backwarp_grid

torch.fx.wrap('backwarp')
torch.fx.wrap('forward_backward_consistency_check')


def backwarp(tenIn, tenflow):
    tenflow = torch.cat(
        [
            tenflow[:, 0:1, :, :] / ((tenIn.shape[3] - 1.0) / 2.0),
            tenflow[:, 1:2, :, :] / ((tenIn.shape[2] - 1.0) / 2.0),
        ],
        1,
    )

    return torch.nn.functional.grid_sample(
        input=tenIn,
        grid=(get_backwarp_grid(tenflow) + tenflow).permute(0, 2, 3, 1),
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True,
    )


class MetricNet(nn.Module):
    def __init__(self):
        super(MetricNet, self).__init__()
        self.metric_in = nn.Conv2d(14, 64, 3, 1, 1)
        self.metric_net1 = nn.Sequential(
            MyPReLU(),
            nn.Conv2d(64, 64, 3, 1, 1)
        )
        self.metric_net2 = nn.Sequential(
            MyPReLU(),
            nn.Conv2d(64, 64, 3, 1, 1)
        )
        self.metric_net3 = nn.Sequential(
            MyPReLU(),
            nn.Conv2d(64, 64, 3, 1, 1)
        )
        self.metric_out = nn.Sequential(
            MyPReLU(),
            nn.Conv2d(64, 2, 3, 1, 1)
        )

    def forward(self, img0, img1, flow01, flow10):
        metric0 = F.l1_loss(img0, backwarp(img1, flow01), reduction='none').mean([1], True)
        metric1 = F.l1_loss(img1, backwarp(img0, flow10), reduction='none').mean([1], True)

        fwd_occ, bwd_occ = forward_backward_consistency_check(flow01, flow10)

        flow01 = torch.cat([flow01[:, 0:1, :, :] / ((flow01.shape[3] - 1.0) / 2.0), flow01[:, 1:2, :, :] / ((flow01.shape[2] - 1.0) / 2.0)], 1)
        flow10 = torch.cat([flow10[:, 0:1, :, :] / ((flow10.shape[3] - 1.0) / 2.0), flow10[:, 1:2, :, :] / ((flow10.shape[2] - 1.0) / 2.0)], 1)

        img = torch.cat((img0, img1), 1)
        metric = torch.cat((-metric0, -metric1), 1)
        flow = torch.cat((flow01, flow10), 1)
        occ = torch.cat((fwd_occ.unsqueeze(1), bwd_occ.unsqueeze(1)), 1)

        feat = self.metric_in(
            torch.cat((img, metric, flow, occ), 1).contiguous(
                memory_format=torch.channels_last
            )
        )
        feat = self.metric_net1(feat) + feat
        feat = self.metric_net2(feat) + feat
        feat = self.metric_net3(feat) + feat
        metric = self.metric_out(feat)

        metric = torch.tanh(metric) * 10

        return metric[:, :1], metric[:, 1:2]
