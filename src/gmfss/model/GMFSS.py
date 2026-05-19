import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .FeatureNet import FeatureNet
from .gmflow.gmflow import GMFlow
from .IFNet_HDv3 import IFNet, ResConv
from .MetricNet import MetricNet
from .softsplat import softsplat as warp

torch.fx.wrap("warp")


class GMFSS(nn.Module):
    def __init__(self, model_dir, scale, ensemble):
        super(GMFSS, self).__init__()
        from .FusionNet_u import GridNet

        self.ifnet = IFNet(ensemble)
        ifnetState = torch.load(
            os.path.join(model_dir, "rife.pkl"), map_location="cpu"
        )
        ifnetState = {k: v for k, v in ifnetState.items() if k.startswith("block0.")}
        self.ifnet.load_state_dict(ifnetState)
        del ifnetState

        # Fold ResConv per-channel beta into preceding conv weight/bias.
        # Done after load (still fp32) and before any later .half()/.to() so the
        # baked weights ride along with the rest of the model.
        for m in self.ifnet.modules():
            if isinstance(m, ResConv):
                m.fold_beta()

        self.flownet = GMFlow()
        self.metricnet = MetricNet()
        self.feat_ext = FeatureNet()
        self.fusionnet = GridNet()

        flownetState = torch.load(
            os.path.join(model_dir, "flownet.pkl"), map_location="cpu"
        )
        self.flownet.load_state_dict(flownetState)
        del flownetState

        metricState = torch.load(
            os.path.join(model_dir, "metric_union.pkl"), map_location="cpu"
        )
        self.metricnet.load_state_dict(metricState)
        del metricState

        featState = torch.load(
            os.path.join(model_dir, "feat_union.pkl"), map_location="cpu"
        )
        self.feat_ext.load_state_dict(featState)
        del featState

        fusionState = torch.load(
            os.path.join(model_dir, "fusionnet_union.pkl"),
            map_location="cpu",
        )
        self.fusionnet.load_state_dict(fusionState)
        del fusionState

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.scale = scale

    def reuse(self, img0, img1):
        # Batch feature extraction: process both frames in one forward pass
        imgs_cat = torch.cat([img0, img1], dim=0)
        feats = self.feat_ext(imgs_cat)
        feat11, feat21 = feats[0].chunk(2, dim=0)
        feat12, feat22 = feats[1].chunk(2, dim=0)
        feat13, feat23 = feats[2].chunk(2, dim=0)
        feat_ext0 = [feat11, feat12, feat13]
        feat_ext1 = [feat21, feat22, feat23]

        img0_half = F.interpolate(img0, scale_factor=0.5, mode="bilinear").contiguous(
            memory_format=torch.channels_last
        )
        img1_half = F.interpolate(img1, scale_factor=0.5, mode="bilinear").contiguous(
            memory_format=torch.channels_last
        )

        if self.scale != 1.0:
            imgf0 = F.interpolate(
                img0_half, scale_factor=self.scale, mode="bilinear"
            ).contiguous(memory_format=torch.channels_last)
            imgf1 = F.interpolate(
                img1_half, scale_factor=self.scale, mode="bilinear"
            ).contiguous(memory_format=torch.channels_last)
        else:
            imgf0 = img0_half
            imgf1 = img1_half
        # Use forward_bidir to share backbone feature extraction
        flow01, flow10 = self.flownet.forward_bidir(imgf0, imgf1)
        if self.scale != 1.0:
            flow01 = (
                F.interpolate(flow01, scale_factor=1.0 / self.scale, mode="bilinear")
                / self.scale
            )
            flow10 = (
                F.interpolate(flow10, scale_factor=1.0 / self.scale, mode="bilinear")
                / self.scale
            )

        metric0, metric1 = self.metricnet(img0_half, img1_half, flow01, flow10)

        return (
            img0_half,
            img1_half,
            flow01,
            flow10,
            metric0,
            metric1,
            feat_ext0,
            feat_ext1,
        )

    def forward_from_reuse(self, reuse_things, timestep):
        (
            img0_half,
            img1_half,
            flow01,
            flow10,
            metric0,
            metric1,
            feat_ext0,
            feat_ext1,
        ) = reuse_things
        feat11, feat12, feat13 = feat_ext0
        feat21, feat22, feat23 = feat_ext1

        timestep = timestep.view(-1, 1, 1, 1)
        inverse_timestep = 1 - timestep

        F1t = timestep * flow01
        F2t = inverse_timestep * flow10

        Z1t = timestep * metric0
        Z2t = inverse_timestep * metric1

        rife = self.ifnet(img0_half, img1_half, timestep.view(-1))

        # Batched pyramid interpolate: cat [F1t(2), Z1t(1), F2t(2), Z2t(1)] = 6ch.
        pyr = torch.cat([F1t, Z1t, F2t, Z2t], dim=1)
        pyr_d = F.interpolate(pyr, scale_factor=0.5, mode="bilinear")
        pyr_dd = F.interpolate(pyr, scale_factor=0.25, mode="bilinear")

        F1td = pyr_d[:, 0:2] * 0.5
        Z1d = pyr_d[:, 2:3]
        F2td = pyr_d[:, 3:5] * 0.5
        Z2d = pyr_d[:, 5:6]

        F1tdd = pyr_dd[:, 0:2] * 0.25
        Z1dd = pyr_dd[:, 2:3]
        F2tdd = pyr_dd[:, 3:5] * 0.25
        Z2dd = pyr_dd[:, 5:6]

        b = img0_half.shape[0]
        img_ch = img0_half.shape[1]

        # /2 res: batch directions, concat (img, feat1) channels → single splat.
        in_l1 = torch.cat(
            [
                torch.cat([img0_half, feat11], dim=1),
                torch.cat([img1_half, feat21], dim=1),
            ],
            dim=0,
        )
        flow_l1 = torch.cat([F1t, F2t], dim=0)
        metric_l1 = torch.cat([Z1t, Z2t], dim=0)
        out_l1 = warp(in_l1, flow_l1, metric_l1, strMode="soft")
        out_l1_0, out_l1_1 = out_l1[:b], out_l1[b:]
        I1t = out_l1_0[:, :img_ch]
        feat1t1 = out_l1_0[:, img_ch:]
        I2t = out_l1_1[:, :img_ch]
        feat2t1 = out_l1_1[:, img_ch:]

        # /4 res: batch directions for feat12/feat22.
        in_l2 = torch.cat([feat12, feat22], dim=0)
        flow_l2 = torch.cat([F1td, F2td], dim=0)
        metric_l2 = torch.cat([Z1d, Z2d], dim=0)
        out_l2 = warp(in_l2, flow_l2, metric_l2, strMode="soft")
        feat1t2, feat2t2 = out_l2[:b], out_l2[b:]

        # /8 res: batch directions for feat13/feat23.
        in_l3 = torch.cat([feat13, feat23], dim=0)
        flow_l3 = torch.cat([F1tdd, F2tdd], dim=0)
        metric_l3 = torch.cat([Z1dd, Z2dd], dim=0)
        out_l3 = warp(in_l3, flow_l3, metric_l3, strMode="soft")
        feat1t3, feat2t3 = out_l3[:b], out_l3[b:]

        out = self.fusionnet(
            torch.cat([I1t, rife, I2t], dim=1).contiguous(
                memory_format=torch.channels_last
            ),
            torch.cat([feat1t1, feat2t1], dim=1).contiguous(
                memory_format=torch.channels_last
            ),
            torch.cat([feat1t2, feat2t2], dim=1).contiguous(
                memory_format=torch.channels_last
            ),
            torch.cat([feat1t3, feat2t3], dim=1).contiguous(
                memory_format=torch.channels_last
            ),
        )

        return torch.clamp(out, 0, 1)

    def forward(self, img0, img1, timestep):
        return self.forward_from_reuse(self.reuse(img0, img1), timestep)
