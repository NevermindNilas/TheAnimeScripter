"""Grid homography estimation and motion clustering (ProjectionUtils.py, pruned)."""

import math

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple

from . import config as cfg


class MedianPool2d(nn.Module):
    """Median pool (usable as median filter when stride=1) module."""

    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super().__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding

    def forward(self, x):
        x = F.pad(x, self._padding(x), mode="reflect")
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x


def HomoCalc(grids, new_grids_loc):
    """
    @param: grids the location of origin grid vertices [2, H, W]
    @param: new_grids_loc the location of desired grid vertices [2, H, W]

    @return: homo_t homograph projection matrix for each grid [3, 3, H-1, W-1]
    """

    _, H, W = grids.shape

    new_grids = new_grids_loc.unsqueeze(0)

    grids = grids.unsqueeze(0)

    one = torch.ones_like(grids[:, 0:1, :-1, :-1], device=grids.device)
    zero = torch.zeros_like(grids[:, 1:2, :-1, :-1], device=grids.device)

    # fmt: off
    A = torch.cat([
        torch.stack([grids[:, 0:1, :-1, :-1], grids[:, 1:2, :-1, :-1], one, zero, zero, zero,
                     -1 * grids[:, 0:1, :-1, :-1] * new_grids[:, 0:1, :-1, :-1], -1 * grids[:, 1:2, :-1, :-1] * new_grids[:, 0:1, :-1, :-1]], 2),
        torch.stack([grids[:, 0:1, 1:, :-1], grids[:, 1:2, 1:, :-1], one, zero, zero, zero,
                     -1 * grids[:, 0:1, 1:, :-1] * new_grids[:, 0:1, 1:, :-1], -1 * grids[:, 1:2, 1:, :-1] * new_grids[:, 0:1, 1:, :-1]], 2),
        torch.stack([grids[:, 0:1, :-1, 1:], grids[:, 1:2, :-1, 1:], one, zero, zero, zero,
                     -1 * grids[:, 0:1, :-1, 1:] * new_grids[:, 0:1, :-1, 1:], -1 * grids[:, 1:2, :-1, 1:] * new_grids[:, 0:1, :-1, 1:]], 2),
        torch.stack([grids[:, 0:1, 1:, 1:], grids[:, 1:2, 1:, 1:], one, zero, zero, zero,
                     -1 * grids[:, 0:1, 1:, 1:] * new_grids[:, 0:1, 1:, 1:], -1 * grids[:, 1:2, 1:, 1:] * new_grids[:, 0:1, 1:, 1:]], 2),
        torch.stack([zero, zero, zero, grids[:, 0:1, :-1, :-1], grids[:, 1:2, :-1, :-1], one,
                     -1 * grids[:, 0:1, :-1, :-1] * new_grids[:, 1:2, :-1, :-1], -1 * grids[:, 1:2, :-1, :-1] * new_grids[:, 1:2, :-1, :-1]], 2),
        torch.stack([zero, zero, zero, grids[:, 0:1, 1:, :-1], grids[:, 1:2, 1:, :-1], one,
                     -1 * grids[:, 0:1, 1:, :-1] * new_grids[:, 1:2, 1:, :-1], -1 * grids[:, 1:2, 1:, :-1] * new_grids[:, 1:2, 1:, :-1]], 2),
        torch.stack([zero, zero, zero, grids[:, 0:1, :-1, 1:], grids[:, 1:2, :-1, 1:], one,
                     -1 * grids[:, 0:1, :-1, 1:] * new_grids[:, 1:2, :-1, 1:], -1 * grids[:, 1:2, :-1, 1:] * new_grids[:, 1:2, :-1, 1:]], 2),
        torch.stack([zero, zero, zero, grids[:, 0:1, 1:, 1:], grids[:, 1:2, 1:, 1:], one,
                     -1 * grids[:, 0:1, 1:, 1:] * new_grids[:, 1:2, 1:, 1:], -1 * grids[:, 1:2, 1:, 1:] * new_grids[:, 1:2, 1:, 1:]], 2),
    ], 1).view(8, 8, -1).permute(2, 0, 1)
    B_ = torch.stack([
        new_grids[:, 0, :-1, :-1],
        new_grids[:, 0, 1:, :-1],
        new_grids[:, 0, :-1, 1:],
        new_grids[:, 0, 1:, 1:],
        new_grids[:, 1, :-1, :-1],
        new_grids[:, 1, 1:, :-1],
        new_grids[:, 1, :-1, 1:],
        new_grids[:, 1, 1:, 1:],
    ], 1).view(8, -1).permute(1, 0)  # A @ H = B ==> H = A^-1 @ B
    # fmt: on
    try:
        A_inverse = torch.inverse(A)
    except RuntimeError:
        # A degenerate cell makes the whole batched inverse raise; upstream
        # fell back to a per-cell python loop here. Least-squares pseudo-inverse
        # keeps the batch path and only differs on the degenerate cells.
        A_inverse = torch.linalg.pinv(A)
    H_recovered = torch.bmm(A_inverse, B_.unsqueeze(2))

    H_ = torch.cat(
        [
            H_recovered,
            torch.ones_like(H_recovered[:, 0:1, :], device=H_recovered.device),
        ],
        1,
    ).view(H_recovered.shape[0], 3, 3)

    H_ = H_.permute(1, 2, 0)
    homo_t = H_.reshape(3, 3, H - 1, W - 1)

    return homo_t


def HomoProj(homo, pts):
    """
    @param: homo [3, 3, G_H-1, G_W-1]
    @param: pts  [N, 2(W, H)] - [:, 0] for width and [:, 1] for height

    @return: projected pts [N, 2(W, H)]
    """

    pts_location_x = (pts[:, 0:1] // cfg.PIXELS).long()
    pts_location_y = (pts[:, 1:2] // cfg.PIXELS).long()

    # if the grid is outside of the image
    maxWidth = cfg.WIDTH // cfg.PIXELS - 1
    maxHeight = cfg.HEIGHT // cfg.PIXELS - 1
    index = (pts_location_x[:, 0] >= maxWidth).nonzero().long()
    pts_location_x[index, :] = maxWidth - 1
    index = (pts_location_y[:, 0] >= maxHeight).nonzero().long()
    pts_location_y[index, :] = maxHeight - 1

    homo = homo.to(pts.device)

    # fmt: off
    x_dominator = pts[:, 0] * homo[0, 0, pts_location_y[:, 0], pts_location_x[:, 0]] + pts[:, 1] * \
        homo[0, 1, pts_location_y[:, 0], pts_location_x[:, 0]] + homo[0, 2, pts_location_y[:, 0], pts_location_x[:, 0]]
    y_dominator = pts[:, 0] * homo[1, 0, pts_location_y[:, 0], pts_location_x[:, 0]] + pts[:, 1] * \
        homo[1, 1, pts_location_y[:, 0], pts_location_x[:, 0]] + homo[1, 2, pts_location_y[:, 0], pts_location_x[:, 0]]
    noiminator = pts[:, 0] * homo[2, 0, pts_location_y[:, 0], pts_location_x[:, 0]] + pts[:, 1] * \
        homo[2, 1, pts_location_y[:, 0], pts_location_x[:, 0]] + homo[2, 2, pts_location_y[:, 0], pts_location_x[:, 0]]
    # fmt: on

    new_kp_x = x_dominator / noiminator
    new_kp_y = y_dominator / noiminator

    return torch.stack([new_kp_x, new_kp_y], 1)


def MotionDistanceMeasure(motion1, motion2):
    """
    @params motion1 np.array(2) (w, h)
    @params motion2 np.array(2) (w, h)

    @return bool: True for far and False for close
    """

    mangnitue_motion1 = np.sqrt(np.sum(motion1**2))
    mangnitue_motion2 = np.sqrt(np.sum(motion2**2))
    diff_mangnitude = np.abs(mangnitue_motion1 - mangnitue_motion2)

    def rot(x):
        return math.atan2(x[1], x[0]) / math.pi * 180

    rot_motion1 = rot(motion1)
    rot_motion2 = rot(motion2)
    diff_rot = np.abs(rot_motion1 - rot_motion2)
    if diff_rot > 180:
        diff_rot = 360 - diff_rot

    return (diff_mangnitude >= cfg.THRESHOLD_MANG) or (diff_rot >= cfg.THRESHOLD_ROT)


def _kmeans2Cluster(motion_numpy):
    """Deterministic k=2 Lloyd's k-means over keypoint motion vectors.

    Upstream used sklearn KMeans(n_clusters=2, random_state=2), which is not a
    TAS dependency; ~512 2D points into 2 clusters does not need one. Init is
    mean + farthest point (deterministic). Label order is irrelevant: the
    caller relabels so cluster 0 is the majority.
    """
    points = motion_numpy.astype(np.float64)
    c0 = points.mean(0)
    c1 = points[np.argmax(((points - c0) ** 2).sum(1))]
    labels = None
    for _ in range(100):
        d0 = ((points - c0) ** 2).sum(1)
        d1 = ((points - c1) ** 2).sum(1)
        newLabels = (d1 < d0).astype(np.int64)
        if labels is not None and np.array_equal(newLabels, labels):
            break
        labels = newLabels
        if (labels == 0).any():
            c0 = points[labels == 0].mean(0)
        if (labels == 1).any():
            c1 = points[labels == 1].mean(0)
    return labels


def multiHomoEstimate(motion, kp):
    """
    @param: motion [4, N]
    @param: kp     [2, N]
    """

    new_kp = torch.cat([kp[1:2, :], kp[0:1, :]], 0) + motion[2:, :]
    new_points_numpy = new_kp.cpu().detach().numpy().transpose(1, 0)
    old_points = torch.stack([kp[1, :], kp[0, :]], 1).to(motion.device)
    old_points_numpy = (
        torch.cat([kp[1:2, :], kp[0:1, :]], 0).cpu().detach().numpy().transpose(1, 0)
    )
    motion_numpy = new_points_numpy - old_points_numpy

    pred_Y = _kmeans2Cluster(motion_numpy)
    if np.sum(pred_Y) > cfg.TOPK / 2:
        pred_Y = 1 - pred_Y
    cluster1_old_points = old_points_numpy[(pred_Y == 0).nonzero()[0], :]
    cluster1_new_points = new_points_numpy[(pred_Y == 0).nonzero()[0], :]

    # pre-warping with global homography
    Homo, _ = cv2.findHomography(cluster1_old_points, cluster1_new_points, cv2.RANSAC)

    if Homo is None:
        Homo = np.eye(3)

    dominator = (
        Homo[2, 0] * old_points_numpy[:, 0]
        + Homo[2, 1] * old_points_numpy[:, 1]
        + Homo[2, 2]
    )
    new_points_projected = (
        torch.from_numpy(
            np.stack(
                [
                    (
                        Homo[0, 0] * old_points_numpy[:, 0]
                        + Homo[0, 1] * old_points_numpy[:, 1]
                        + Homo[0, 2]
                    )
                    / dominator,
                    (
                        Homo[1, 0] * old_points_numpy[:, 0]
                        + Homo[1, 1] * old_points_numpy[:, 1]
                        + Homo[1, 2]
                    )
                    / dominator,
                ],
                1,
            ).astype(np.float32)
        )
        .to(old_points.device)
        .permute(1, 0)
    )

    index = (pred_Y == 1).nonzero()[0]
    attribute = np.zeros_like(new_points_numpy[:, 0:1])  # N', 1
    cluster2_old_points = old_points_numpy[index, :]
    cluster2_new_points = new_points_numpy[index, :]
    attribute[index, :] = np.expand_dims(np.ones_like(index), 1)

    cluster1_motion = cluster1_new_points - cluster1_old_points
    clsuter2_motion = cluster2_new_points - cluster2_old_points
    cluster1_meanMotion = np.mean(cluster1_motion, 0)
    cluster2_meanMotion = np.mean(clsuter2_motion, 0)
    distanceMeasure = MotionDistanceMeasure(cluster1_meanMotion, cluster2_meanMotion)

    threhold = (np.sum(pred_Y) > cfg.THRESHOLDPOINT) and distanceMeasure

    if threhold:
        Homo_2, _ = cv2.findHomography(
            cluster2_old_points, cluster2_new_points, cv2.RANSAC
        )
        if Homo_2 is None:
            Homo_2 = Homo

        meshes_x, meshes_y = np.meshgrid(
            np.arange(0, cfg.WIDTH, cfg.PIXELS), np.arange(0, cfg.HEIGHT, cfg.PIXELS)
        )

        # Use first cluster to do projection
        x_dominator = Homo[0, 0] * meshes_x + Homo[0, 1] * meshes_y + Homo[0, 2]
        y_dominator = Homo[1, 0] * meshes_x + Homo[1, 1] * meshes_y + Homo[1, 2]
        noiminator = Homo[2, 0] * meshes_x + Homo[2, 1] * meshes_y + Homo[2, 2]

        projected_1 = np.reshape(
            np.stack([x_dominator / noiminator, y_dominator / noiminator], 2), (-1, 2)
        )

        # Use second cluster to do projection
        x_dominator = Homo_2[0, 0] * meshes_x + Homo_2[0, 1] * meshes_y + Homo_2[0, 2]
        y_dominator = Homo_2[1, 0] * meshes_x + Homo_2[1, 1] * meshes_y + Homo_2[1, 2]
        noiminator = Homo_2[2, 0] * meshes_x + Homo_2[2, 1] * meshes_y + Homo_2[2, 2]

        projected_2 = np.reshape(
            np.stack([x_dominator / noiminator, y_dominator / noiminator], 2), (-1, 2)
        )

        # Determine use which projected position
        distance_x = np.expand_dims(new_points_numpy[:, 0], 0) - np.reshape(
            meshes_x, (-1, 1)
        )
        distance_y = np.expand_dims(new_points_numpy[:, 1], 0) - np.reshape(
            meshes_y, (-1, 1)
        )
        distance = distance_x**2 + distance_y**2  # N, N'
        distance_mask = distance < (cfg.RADIUS**2)  # N, N'
        distance_mask_value = distance_mask.astype(np.float32) * attribute.transpose(
            1, 0
        )  # N, N'
        distance = np.sum(distance_mask_value, 1) / (np.sum(distance_mask, 1) + 1e-9)

        project_pos = np.reshape(
            np.expand_dims(distance, 1) * projected_2
            + np.expand_dims((1 - distance), 1) * projected_1,
            (cfg.HEIGHT // cfg.PIXELS, cfg.WIDTH // cfg.PIXELS, 2),
        )

        meshes_projected = (
            torch.from_numpy(project_pos.astype(np.float32))
            .to(old_points.device)
            .permute(2, 0, 1)
        )

        # calculate reference location for each keypoint
        meshes_x, meshes_y = torch.meshgrid(
            torch.arange(0, cfg.WIDTH, cfg.PIXELS),
            torch.arange(0, cfg.HEIGHT, cfg.PIXELS),
            indexing="ij",
        )
        meshes_x = meshes_x.float().permute(1, 0).to(old_points.device)
        meshes_y = meshes_y.float().permute(1, 0).to(old_points.device)
        meshes = torch.stack([meshes_x, meshes_y], 0)

        x_motions = meshes[0, :, :] - meshes_projected[0, :, :]
        y_motions = meshes[1, :, :] - meshes_projected[1, :, :]

        homo_cal = HomoCalc(meshes, meshes_projected)
        project_pts = HomoProj(homo_cal, old_points)
        new_points_projected = project_pts.to(old_points.device).permute(1, 0)

    else:
        Homo = torch.from_numpy(Homo.astype(np.float32)).to(old_points.device)
        meshes_x, meshes_y = torch.meshgrid(
            torch.arange(0, cfg.WIDTH, cfg.PIXELS),
            torch.arange(0, cfg.HEIGHT, cfg.PIXELS),
            indexing="ij",
        )
        meshes_x = meshes_x.float().permute(1, 0).to(old_points.device)
        meshes_y = meshes_y.float().permute(1, 0).to(old_points.device)
        meshes_z = torch.ones_like(meshes_x).to(old_points.device)
        meshes = torch.stack([meshes_x, meshes_y, meshes_z], 0)
        meshes_projected = torch.mm(Homo, meshes.view(3, -1)).view(*meshes.shape)

        x_motions = (
            meshes[0, :, :] - meshes_projected[0, :, :] / (meshes_projected[2, :, :])
        )
        y_motions = (
            meshes[1, :, :] - meshes_projected[1, :, :] / (meshes_projected[2, :, :])
        )

    grids = (
        torch.stack(
            torch.meshgrid(
                torch.arange(0, cfg.WIDTH, cfg.PIXELS),
                torch.arange(0, cfg.HEIGHT, cfg.PIXELS),
                indexing="ij",
            ),
            0,
        )
        .to(motion.device)
        .permute(0, 2, 1)
        .reshape(2, -1)
        .permute(1, 0)
    )

    grids = grids.unsqueeze(2).float()  # N', 2, 1
    projected_motion = (
        torch.stack([x_motions, y_motions], 2).view(-1, 2, 1).to(motion.device)
    )

    redisual_kp_motion = new_points_projected - torch.cat([kp[1:2, :], kp[0:1, :]], 0)

    motion[:2, :] = motion[:2, :] + motion[2:, :]
    motion = motion.unsqueeze(0).repeat(grids.shape[0], 1, 1)  # N', 4, N
    motion[:, :2, :] = (motion[:, :2, :] - grids) / cfg.WIDTH
    origin_motion = motion[:, 2:, :] / cfg.FLOWC
    motion[:, 2:, :] = (redisual_kp_motion.unsqueeze(0) - motion[:, 2:, :]) / cfg.FLOWC

    return motion, projected_motion / cfg.FLOWC, origin_motion
