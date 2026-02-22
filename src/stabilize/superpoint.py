from types import SimpleNamespace
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SUPERPOINT_WEIGHTS_URL = "https://github.com/nagadomi/nunif/releases/download/0.0.0/superpoint_v6_from_tf.pth"


class VGGBlock(nn.Sequential):
    def __init__(self, c_in, c_out, kernel_size, relu=True):
        padding = (kernel_size - 1) // 2
        conv = nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=1, padding=padding)
        act = nn.ReLU(inplace=True) if relu else nn.Identity()
        bn = nn.BatchNorm2d(c_out)
        super().__init__(conv, act, bn)


def batched_nms(scores, nms_radius):
    assert nms_radius >= 0

    def max_pool(x):
        return F.max_pool2d(x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)

    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))

    return torch.where(max_mask, scores, zeros)


def sample_descriptors(keypoints, descriptors, stride=8):
    b, c, h, w = descriptors.shape
    keypoints = (keypoints + 0.5) / (keypoints.new_tensor([w, h]) * stride)
    keypoints = keypoints * 2 - 1
    sampled = F.grid_sample(
        descriptors,
        keypoints.view(b, 1, -1, 2),
        mode="bilinear",
        align_corners=False,
    )
    sampled = F.normalize(sampled.reshape(b, c, -1), p=2, dim=1)
    return sampled


def select_top_k_keypoints(keypoints, scores, k):
    if k >= len(keypoints):
        return keypoints, scores
    idx = torch.topk(scores, k, dim=0, sorted=True).indices
    return keypoints[idx], scores[idx]


class SuperPoint(nn.Module):
    default_conf = {
        "nms_radius": 4,
        "max_num_keypoints": None,
        "detection_threshold": 0.01,
        "remove_borders": 4,
        "descriptor_dim": 256,
        "channels": [64, 64, 128, 128, 256],
    }

    def __init__(self, **conf):
        super().__init__()
        merged = {**self.default_conf, **conf}
        self.conf = SimpleNamespace(**merged)
        self.stride = 2 ** (len(self.conf.channels) - 2)
        channels = [1, *self.conf.channels[:-1]]

        backbone = []
        for i, c in enumerate(channels[1:], 1):
            layers = [VGGBlock(channels[i - 1], c, 3), VGGBlock(c, c, 3)]
            if i < len(channels) - 1:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            backbone.append(nn.Sequential(*layers))
        self.backbone = nn.Sequential(*backbone)

        c = self.conf.channels[-1]
        self.detector = nn.Sequential(
            VGGBlock(channels[-1], c, 3),
            VGGBlock(c, self.stride**2 + 1, 1, relu=False),
        )
        self.descriptor = nn.Sequential(
            VGGBlock(channels[-1], c, 3),
            VGGBlock(c, self.conf.descriptor_dim, 1, relu=False),
        )

        self.requires_grad_(False)

    def forward(self, image):
        if image.shape[1] == 3:
            scale = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            image = (image * scale).sum(1, keepdim=True)

        features = self.backbone(image)
        descriptors_dense = F.normalize(self.descriptor(features), p=2, dim=1)

        scores = self.detector(features)
        scores = F.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, self.stride, self.stride)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h * self.stride, w * self.stride)
        scores = batched_nms(scores, self.conf.nms_radius)

        if self.conf.remove_borders:
            pad = self.conf.remove_borders
            scores[:, :pad] = -1
            scores[:, :, :pad] = -1
            scores[:, -pad:] = -1
            scores[:, :, -pad:] = -1

        if b > 1:
            idxs = torch.where(scores > self.conf.detection_threshold)
            mask = idxs[0] == torch.arange(b, device=scores.device)[:, None]
        else:
            scores = scores.squeeze(0)
            idxs = torch.where(scores > self.conf.detection_threshold)

        keypoints_all = torch.stack(idxs[-2:], dim=-1).flip(1).to(scores.dtype)
        scores_all = scores[idxs]

        keypoints = []
        keypoint_scores = []
        descriptors = []
        for i in range(b):
            if b > 1:
                keypoints_i = keypoints_all[mask[i]]
                scores_i = scores_all[mask[i]]
            else:
                keypoints_i = keypoints_all
                scores_i = scores_all

            if self.conf.max_num_keypoints is not None:
                keypoints_i, scores_i = select_top_k_keypoints(
                    keypoints_i,
                    scores_i,
                    self.conf.max_num_keypoints,
                )

            descriptors_i = sample_descriptors(keypoints_i[None], descriptors_dense[i, None], self.stride)
            keypoints.append(keypoints_i)
            keypoint_scores.append(scores_i)
            descriptors.append(descriptors_i.squeeze(0).transpose(0, 1))

        return {
            "keypoints": keypoints,
            "keypoint_scores": keypoint_scores,
            "descriptors": descriptors,
        }

    def load(self, map_location="cpu"):
        try:
            state_dict = torch.hub.load_state_dict_from_url(
                SUPERPOINT_WEIGHTS_URL,
                weights_only=True,
                map_location=map_location,
            )
        except TypeError:
            state_dict = torch.hub.load_state_dict_from_url(
                SUPERPOINT_WEIGHTS_URL,
                map_location=map_location,
            )
        self.load_state_dict(state_dict)
        return self

    @torch.inference_mode()
    def infer(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(0)
            is_batch = False
        else:
            is_batch = True

        out = self.forward(x)
        packed = []
        for i in range(x.shape[0]):
            packed.append(
                {
                    "keypoints": out["keypoints"][i],
                    "descriptors": out["descriptors"][i],
                    "keypoint_scores": out["keypoint_scores"][i],
                }
            )

        if is_batch:
            return packed
        return packed[0]


@torch.inference_mode()
def find_match_index(kp1, kp2, threshold=0.3, return_score=False):
    d1 = kp1["descriptors"]
    d2 = kp2["descriptors"]

    cosine_similarity = d1 @ d2.t()
    match_index = torch.argmax(cosine_similarity, dim=-1)
    max_similarity = torch.gather(cosine_similarity, dim=1, index=match_index.view(-1, 1)).view(-1)
    keep = max_similarity > threshold

    idx1 = torch.arange(d1.shape[0], device=d1.device)[keep]
    idx2 = match_index[keep]

    if return_score:
        return idx1, idx2, max_similarity[keep]
    return idx1, idx2


def cosine_annealing(min_v, max_v, t, max_t):
    if max_t <= 0:
        return min_v
    return min_v + 0.5 * (max_v - min_v) * (1.0 + math.cos((float(t) / float(max_t)) * math.pi))


def find_transform(
    xy1,
    xy2,
    center,
    mask=None,
    iteration=50,
    lr_translation=0.1,
    lr_scale_rotation=0.1,
    sigma=2.0,
    sigma_min=None,
    sigma_max=2.0,
    disable_shift=False,
    disable_scale=True,
    disable_rotate=False,
):
    if xy1.ndim == 2:
        batch = False
        xy1 = xy1.float().cpu().unsqueeze(0)
        xy2 = xy2.float().cpu().unsqueeze(0)
        if not torch.is_tensor(center):
            center = torch.tensor(center, dtype=torch.float32)
        center = center.view(1, 1, 2).float().cpu()
    else:
        batch = True
        xy1 = xy1.float()
        xy2 = xy2.float()

    if mask is None:
        mask = torch.ones((xy1.shape[0], xy1.shape[1]), dtype=torch.bool, device=xy1.device)
    elif mask.ndim == 3:
        mask = torch.logical_and(mask[:, :, 0], mask[:, :, 1])

    b = xy1.shape[0]

    translation = torch.zeros((b, 1, 2), dtype=xy1.dtype, device=xy1.device, requires_grad=True)
    scale = torch.ones((b, 1, 1), dtype=xy1.dtype, device=xy1.device, requires_grad=True)
    rotation = torch.zeros((b, 1, 1), dtype=xy1.dtype, device=xy1.device, requires_grad=True)

    param_groups = []
    if not disable_shift:
        param_groups.append({"params": [translation], "lr": lr_translation})
    if not disable_scale:
        param_groups.append({"params": [scale], "lr": lr_scale_rotation})
    if not disable_rotate:
        param_groups.append({"params": [rotation], "lr": lr_scale_rotation})

    if len(param_groups) == 0:
        if batch:
            shift = torch.zeros((b, 2), dtype=xy1.dtype, device=xy1.device)
            scale_out = torch.ones((b, 1), dtype=xy1.dtype, device=xy1.device)
            angle = torch.zeros((b, 1), dtype=xy1.dtype, device=xy1.device)
            return shift, scale_out, angle, center.view(b, 2)
        shift = torch.zeros((2,), dtype=xy1.dtype)
        scale_out = torch.tensor(1.0, dtype=xy1.dtype)
        angle = torch.tensor(0.0, dtype=xy1.dtype)
        return shift, scale_out, angle, center.view(2)

    optimizer = torch.optim.Adam(param_groups, betas=(0.5, 0.9))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=max(1, int(iteration)),
        eta_min=lr_scale_rotation * 1e-3,
    )

    xy1c = xy1 - center
    xy2c = xy2 - center
    norm_scale = torch.nan_to_num(xy1c).abs().amax(dim=[1, 2], keepdim=True)
    norm_scale = torch.clamp(norm_scale, min=1e-6)
    xy1n = xy1c / norm_scale
    xy2n = xy2c / norm_scale

    for i in range(max(1, int(iteration))):
        optimizer.zero_grad()
        xy = xy1n

        rcos = torch.cos(rotation)
        rsin = torch.sin(rotation)
        xy = torch.cat(
            [xy[:, :, :1] * rcos - xy[:, :, 1:] * rsin, xy[:, :, :1] * rsin + xy[:, :, 1:] * rcos],
            dim=2,
        )

        xy = xy * scale
        xy = xy + translation

        if (sigma_min is not None or sigma is not None) and i > 0:
            sigma_i = sigma
            if sigma_min is not None:
                sigma_i = cosine_annealing(sigma_min, sigma_max, i, iteration)

            loss_all = F.l1_loss(xy, xy2n, reduction="none")
            loss_tmp = loss_all.detach().clone()
            valid_mask_3d = mask.unsqueeze(-1).expand_as(loss_tmp)
            loss_tmp[torch.logical_not(valid_mask_3d)] = torch.nan
            mean = torch.nanmean(loss_tmp, dim=[1, 2], keepdim=True)
            stdv = torch.sqrt(torch.nanmean((loss_tmp - mean).pow(2), dim=[1, 2], keepdim=True))
            stdv = torch.nan_to_num(stdv, nan=1.0)
            stdv = torch.clamp(stdv, min=1e-6)
            outlier_mask = ((loss_tmp - mean) / stdv) < sigma_i
            joint_mask = torch.logical_and(valid_mask_3d, outlier_mask)

            if joint_mask.any():
                loss = loss_all[joint_mask].mean()
            else:
                loss = F.l1_loss(xy[valid_mask_3d], xy2n[valid_mask_3d])
        else:
            valid_mask_3d = mask.unsqueeze(-1).expand_as(xy)
            loss = F.l1_loss(xy[valid_mask_3d], xy2n[valid_mask_3d])

        loss.backward()
        optimizer.step()
        scheduler.step()

    shift = (translation.detach() * norm_scale).reshape(b, 2)
    scale_out = scale.detach().reshape(b, 1)
    angle = rotation.detach().reshape(b, 1)

    if batch:
        center_out = center.view(b, 2)
        return shift, scale_out, angle, center_out

    return shift[0], scale_out[0, 0], angle[0, 0], center.view(2)
