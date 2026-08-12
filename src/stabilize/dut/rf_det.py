"""RFDet keypoint detector (rf_det_module.py + rf_det_so.py merged, detection only)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as cfg
from .image_utils import (
    L2Norm,
    filter_border,
    get_gauss_filter_weight,
    nms,
    soft_max_and_argmax_1d,
    soft_nms_3d,
    topk_map,
)


class RFDetSO(nn.Module):
    def __init__(self):
        super().__init__()

        self.score_com_strength = cfg.SCORE_COM_STRENGTH
        self.scale_com_strength = cfg.SCALE_COM_STRENGTH
        self.NMS_THRESH = cfg.NMS_THRESH
        self.NMS_KSIZE = cfg.NMS_KSIZE
        self.TOPK = cfg.TOPK
        self.GAUSSIAN_KSIZE = cfg.GAUSSIAN_KSIZE
        self.GAUSSIAN_SIGMA = cfg.GAUSSIAN_SIGMA

        ksize, padding, dilation = cfg.KSIZE, cfg.PADDING, cfg.DILATION

        def block(inCh):
            conv = nn.Conv2d(
                in_channels=inCh,
                out_channels=16,
                kernel_size=ksize,
                stride=1,
                padding=padding,
                dilation=dilation,
            )
            return conv

        self.conv1 = block(1)
        self.insnorm1 = nn.InstanceNorm2d(16, affine=True)
        for i, scale in enumerate((3, 5, 7, 9, 11, 13, 15, 17, 19, 21)):
            if i > 0:
                setattr(self, f"conv{i + 1}", block(16))
                setattr(self, f"insnorm{i + 1}", nn.InstanceNorm2d(16, affine=True))
            setattr(
                self,
                f"conv_s{scale}",
                nn.Conv2d(
                    in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0
                ),
            )
            setattr(self, f"insnorm_s{scale}", nn.InstanceNorm2d(1, affine=True))
            setattr(
                self,
                f"conv_o{scale}",
                nn.Conv2d(
                    in_channels=16, out_channels=2, kernel_size=1, stride=1, padding=0
                ),
            )

        self.scale_list = torch.tensor(cfg.SCALE_LIST)

    def forward(self, photos):
        # Extract score map in scale space from 3 to 21
        score_maps = []
        orint_maps = []
        feat = photos
        prev = None
        for i, scale in enumerate((3, 5, 7, 9, 11, 13, 15, 17, 19, 21)):
            conv = getattr(self, f"conv{i + 1}")
            insnorm = getattr(self, f"insnorm{i + 1}")
            feat = F.leaky_relu(insnorm(conv(feat)))
            score = getattr(self, f"insnorm_s{scale}")(
                getattr(self, f"conv_s{scale}")(feat)
            ).permute(0, 2, 3, 1)
            orint = (
                L2Norm(getattr(self, f"conv_o{scale}")(feat), dim=1)
                .permute(0, 2, 3, 1)
                .unsqueeze(-2)
            )
            score_maps.append(score)
            orint_maps.append(orint)
            # Residual accumulation between scales (the add after the final
            # scale is dead — nothing reads feat again — matching upstream,
            # which never accumulated past s21).
            if prev is not None:
                feat = feat + prev
            prev = feat

        score_maps = torch.cat(score_maps, -1)  # (B, H, W, 10)
        orint_maps = torch.cat(orint_maps, -2)  # (B, H, W, 10, 2)

        scale_probs = soft_nms_3d(score_maps, ksize=15, com_strength=3.0)
        score_map, scale_map, orint_map = soft_max_and_argmax_1d(
            input=scale_probs,
            orint_maps=orint_maps,
            dim=-1,
            scale_list=self.scale_list,
            keepdim=True,
            com_strength1=self.score_com_strength,
            com_strength2=self.scale_com_strength,
        )

        return score_map, scale_map, orint_map

    def process(self, im1w_score):
        """nms(n), topk(t), gaussian kernel(g) operation on the raw score map."""
        im1w_score = filter_border(im1w_score)

        nms_mask = nms(im1w_score, thresh=self.NMS_THRESH, ksize=self.NMS_KSIZE)
        im1w_score = im1w_score * nms_mask
        topk_value = im1w_score

        topk_mask = topk_map(im1w_score, self.TOPK)
        im1w_score = topk_mask.to(torch.float) * im1w_score

        psf = im1w_score.new_tensor(
            get_gauss_filter_weight(self.GAUSSIAN_KSIZE, self.GAUSSIAN_SIGMA)[
                None, None, :, :
            ]
        ).to(im1w_score.device)
        im1w_score = F.conv2d(
            input=im1w_score.permute(0, 3, 1, 2),
            weight=psf,
            stride=1,
            padding=self.GAUSSIAN_KSIZE // 2,
        ).permute(0, 2, 3, 1)  # (B, H, W, 1)

        im1w_score = im1w_score.clamp(min=0.0, max=1.0)

        return im1w_score, topk_mask, topk_value

    def loadWeights(self, path):
        pretrained = torch.load(path, map_location="cpu", weights_only=False)[
            "state_dict"
        ]
        model_dict = self.state_dict()
        pretrained = {
            k[4:]: v
            for k, v in pretrained.items()
            if k[:3] == "det" and k[4:] in model_dict
        }
        assert len(pretrained) == len(model_dict), (
            f"RFDet checkpoint mismatch: {len(pretrained)} of {len(model_dict)} keys"
        )
        self.load_state_dict(pretrained)

    def detect(self, gray):
        """
        @param gray [B, 1, H, W] gray images in [0, 1]
        @return im_topk [B, 1, H, W] float mask, kpts list of [N, 4] (b, 0, y, x)
        """
        im_rawsc, _, _ = self.forward(gray)
        im_score = self.process(im_rawsc)[0]
        im_topk = topk_map(im_score, self.TOPK).permute(0, 3, 1, 2)  # B, 1, H, W
        kpts = im_topk.nonzero()  # (B*topk, 4)
        kpts = [kpts[kpts[:, 0] == idx, :] for idx in range(gray.shape[0])]
        return im_topk.float(), kpts
