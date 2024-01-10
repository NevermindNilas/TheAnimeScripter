import random
import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
import torch.optim as optim
import itertools
from torch.nn.parallel import DistributedDataParallel as DDP
from model.gmflow.gmflow import GMFlow
from model.MetricNet import MetricNet
from model.FeatureNet import FeatureNet
from model.FusionNet_b import GridNet
from model.softsplat import softsplat as warp
import torch.nn.functional as F
from model.loss import *

from model.lpips import LPIPS

device = torch.device("cuda")
    
class Model:
    def __init__(self, local_rank=-1):
        self.flownet = GMFlow()
        self.metricnet = MetricNet()
        self.feat_ext = FeatureNet()
        self.fusionnet = GridNet()
        self.device()
        # self.optimG = AdamW(self.fusionnet.parameters(), lr=1e-6, weight_decay=1e-4)
        self.optimG = AdamW(itertools.chain(
            self.metricnet.parameters(),
            self.feat_ext.parameters(),
            self.fusionnet.parameters()), lr=1e-6, weight_decay=1e-4)
        self.l1_loss = Charbonnier_L1().to(device)
        self.lpips = LPIPS(net='vgg').to(device)
        self.scaler = torch.cuda.amp.GradScaler()

    def train(self):
        self.flownet.eval()
        self.metricnet.train()
        self.feat_ext.train()
        self.fusionnet.train()

    def eval(self):
        self.flownet.eval()
        self.metricnet.eval()
        self.feat_ext.eval()
        self.fusionnet.eval()

    def device(self):
        self.flownet.to(device)
        self.metricnet.to(device)
        self.feat_ext.to(device)
        self.fusionnet.to(device)

    def load_model(self, path, rank=-1):
        def convert(param):
            return {
                k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k
            }

        self.flownet.load_state_dict(torch.load('{}/flownet.pkl'.format(path)))
        # self.metricnet.load_state_dict(torch.load('{}/metric.pkl'.format(path)))
        # self.feat_ext.load_state_dict(torch.load('{}/feat.pkl'.format(path)))
        # self.fusionnet.load_state_dict(torch.load('{}/fusionnet.pkl'.format(path)))
    
    def save_model(self, path, rank=0):
        torch.save(self.flownet.state_dict(), f'{path}/flownet.pkl')
        torch.save(self.metricnet.state_dict(), f'{path}/metric.pkl')
        torch.save(self.feat_ext.state_dict(), f'{path}/feat.pkl')
        torch.save(self.fusionnet.state_dict(), f'{path}/fusionnet.pkl')

    def inference(self, img0, img1, timestep, simple_color_aug):
        with torch.no_grad():
            # flow01 = self.flownet(img0, img1)
            # flow10 = self.flownet(img1, img0)

            img0_chunks = torch.chunk(img0, chunks=2, dim=0)
            img1_chunks = torch.chunk(img1, chunks=2, dim=0)
            flow0_chunks = list()
            flow1_chunks = list()
            for s in range(2):
                flow_01 = self.flownet(img0_chunks[s], img1_chunks[s])
                flow_10 = self.flownet(img1_chunks[s], img0_chunks[s])
                flow0_chunks.append(flow_01)
                flow1_chunks.append(flow_10)
            flow01 = torch.cat(flow0_chunks, dim=0)
            flow10 = torch.cat(flow1_chunks, dim=0)

            flow01 = F.interpolate(flow01, scale_factor = 0.5, mode="bilinear", align_corners=False) * 0.5
            flow10 = F.interpolate(flow10, scale_factor = 0.5, mode="bilinear", align_corners=False) * 0.5

            img0, img1 = simple_color_aug.augment(img0), simple_color_aug.augment(img1)

            img0s = F.interpolate(img0, scale_factor = 0.5, mode="bilinear", align_corners=False)
            img1s = F.interpolate(img1, scale_factor = 0.5, mode="bilinear", align_corners=False)

        with torch.autocast(device_type='cuda'):
            metric0, metric1 = self.metricnet(img0s, img1s, flow01, flow10)

            feat11, feat12, feat13 = self.feat_ext(img0)
            feat21, feat22, feat23 = self.feat_ext(img1)
            
            F1t = timestep * flow01
            F2t = (1-timestep) * flow10

            Z1t = timestep * metric0
            Z2t = (1-timestep) * metric1

            I1t = warp(img0s, F1t, Z1t, strMode='soft')
            I2t = warp(img1s, F2t, Z2t, strMode='soft')

            feat1t1 = warp(feat11, F1t, Z1t, strMode='soft')
            feat2t1 = warp(feat21, F2t, Z2t, strMode='soft')

            F1td = F.interpolate(F1t, scale_factor = 0.5, mode="bilinear", align_corners=False) * 0.5
            Z1d = F.interpolate(Z1t, scale_factor = 0.5, mode="bilinear", align_corners=False)
            feat1t2 = warp(feat12, F1td, Z1d, strMode='soft')
            F2td = F.interpolate(F2t, scale_factor = 0.5, mode="bilinear", align_corners=False) * 0.5
            Z2d = F.interpolate(Z2t, scale_factor = 0.5, mode="bilinear", align_corners=False)
            feat2t2 = warp(feat22, F2td, Z2d, strMode='soft')

            F1tdd = F.interpolate(F1t, scale_factor = 0.25, mode="bilinear", align_corners=False) * 0.25
            Z1dd = F.interpolate(Z1t, scale_factor = 0.25, mode="bilinear", align_corners=False)
            feat1t3 = warp(feat13, F1tdd, Z1dd, strMode='soft')
            F2tdd = F.interpolate(F2t, scale_factor = 0.25, mode="bilinear", align_corners=False) * 0.25
            Z2dd = F.interpolate(Z2t, scale_factor = 0.25, mode="bilinear", align_corners=False)
            feat2t3 = warp(feat23, F2tdd, Z2dd, strMode='soft')

            merged = self.fusionnet(torch.cat([img0s, I1t, I2t, img1s], dim=1), torch.cat([feat1t1, feat2t1], dim=1), torch.cat([feat1t2, feat2t2], dim=1), torch.cat([feat1t3, feat2t3], dim=1))

            merged = simple_color_aug.reverse_augment(merged)
        
        return flow01, flow10, metric0, metric1, merged

    def update(self, imgs, gt, learning_rate=0, training=True, timestep=0.5, step=0, spe=1136):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        img0 = imgs[:, :3]
        img1 = imgs[:, 3:]
        if training:
            self.train()
        else:
            self.eval()

        accum_iter = 2

        simple_color_aug = SimpleColorAugmentation(enable=True)

        flow01, flow10, metric0, metric1, merged = self.inference(img0, img1, timestep, simple_color_aug)

        with torch.autocast(device_type='cuda'):
            loss_l1 = self.l1_loss(merged - gt)

            # loss_lpips = self.lpips.forward(torch.clamp(merged, 0, 1), gt).mean()

            merged_chunks = torch.chunk(merged, chunks=2, dim=0)
            gt_chunks = torch.chunk(gt, chunks=2, dim=0)
            loss_lpips_chunks = list()
            for s in range(2):
                lpips_loss = self.lpips.forward(torch.clamp(merged_chunks[s], 0, 1), gt_chunks[s])
                loss_lpips_chunks.append(lpips_loss)
            loss_lpips = torch.cat(loss_lpips_chunks, dim=0).mean()

        if training:
            loss_G = (loss_l1 + loss_lpips) / accum_iter
            self.scaler.scale(loss_G).backward()
            if ((step + 1) % accum_iter == 0) or ((step + 1) % spe == 0):
                self.scaler.step(self.optimG)
                self.scaler.update()
                self.optimG.zero_grad()

        return merged, torch.cat((flow01, flow10), 1), metric0, metric1, loss_l1, loss_lpips


class SimpleColorAugmentation:
    def __init__(self, enable=True) -> None:
        self.seed = random.uniform(0, 1)
        if self.seed < 0.167:
            self.swap = [2, 1, 0]  # swap 1,3
            self.reverse_swap = [2, 1, 0]
        elif 0.167 < self.seed < 0.333:
            self.swap = [2, 0, 1]
            self.reverse_swap = [1, 2, 0]
        elif 0.333 < self.seed < 0.5:
            self.swap = [1, 2, 0]
            self.reverse_swap = [2, 0, 1]
        elif 0.5 < self.seed < 0.667:
            self.swap = [1, 0, 2]
            self.reverse_swap = [1, 0, 2]
        elif 0.667 < self.seed < 0.833:
            self.swap = [0, 2, 1]
            self.reverse_swap = [0, 2, 1]
        else:
            self.swap = [0, 1, 2]
            self.reverse_swap = [0, 1, 2]
        if not enable:
            self.swap = [0, 1, 2]  # no swap
            self.reverse_swap = self.swap
        pass

    def augment(self, img):
        """
        param: img, torch tensor, CHW
        """
        img = img[:, self.swap, :, :]
        return img

    def reverse_augment(self, img):
        img = img[:, self.reverse_swap, :, :]
        return img