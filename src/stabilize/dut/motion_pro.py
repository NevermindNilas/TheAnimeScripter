"""MotionPro: propagates sparse keypoint motion to the mesh grid (MotionPro.py)."""

import torch
import torch.nn as nn

from . import config as cfg
from .projection import MedianPool2d, multiHomoEstimate


class MotionPro(nn.Module):
    def __init__(
        self,
        inplanes=2,
        embeddingSize=64,
        hiddenSize=128,
        number_points=512,
        kernel=5,
    ):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Conv1d(inplanes, embeddingSize, 1),
            nn.ReLU(),
        )
        self.embedding_motion = nn.Sequential(
            nn.Conv1d(inplanes, embeddingSize, 1),
            nn.ReLU(),
        )

        self.pad = kernel // 2
        self.conv1 = nn.Conv1d(embeddingSize, embeddingSize, 1)
        self.conv2 = nn.Conv1d(embeddingSize, embeddingSize // 2, 1)
        self.conv3 = nn.Conv1d(embeddingSize // 2, 1, 1)

        self.weighted = nn.Softmax(dim=2)

        self.relu = nn.ReLU()
        self.leakyRelu = nn.LeakyReLU(0.1)

        self.m_conv1 = nn.Conv1d(embeddingSize, 2 * embeddingSize, 1)
        self.m_conv2 = nn.Conv1d(2 * embeddingSize, 2 * embeddingSize, 1)
        self.m_conv3 = nn.Conv1d(2 * embeddingSize, embeddingSize, 1)

        self.fuse_conv1 = nn.Conv1d(
            embeddingSize + embeddingSize // 2, embeddingSize, 1
        )
        self.fuse_conv2 = nn.Conv1d(embeddingSize, embeddingSize, 1)

        self.decoder = nn.Linear(embeddingSize, 2, bias=False)

        self.homoEstimate = multiHomoEstimate
        self.meidanPool = MedianPool2d(5, same=True)

    def forward(self, motion):
        """
        @param: motion contains distance info and motion info of keypoints

        @return: predicted motion for each grid vertex
        """
        distance_info = motion[:, 0:2, :]
        motion_info = motion[0:1, 2:4, :]

        embedding_distance = self.embedding(distance_info)
        embedding_distance = self.leakyRelu(self.conv1(embedding_distance))
        embedding_distance = self.leakyRelu(self.conv2(embedding_distance))
        distance_weighted = self.weighted(self.conv3(embedding_distance))

        embedding_motion = self.embedding_motion(motion_info)
        embedding_motion = self.leakyRelu(self.m_conv1(embedding_motion))
        embedding_motion = self.leakyRelu(self.m_conv2(embedding_motion))
        embedding_motion = self.leakyRelu(self.m_conv3(embedding_motion))
        embedding_motion = embedding_motion.repeat(distance_info.shape[0], 1, 1)

        embedding_motion = torch.cat([embedding_motion, embedding_distance], 1)
        embedding_motion = self.leakyRelu(self.fuse_conv1(embedding_motion))
        embedding_motion = self.leakyRelu(self.fuse_conv2(embedding_motion))

        embedding_motion = torch.sum(embedding_motion * distance_weighted, 2)

        out_motion = self.decoder(embedding_motion)

        return out_motion

    def inference(self, x_flow, y_flow, kp):
        """
        @param x_flow [B, 1, H, W]
        @param y_flow [B, 1, H, W]
        @param kp     [N, 4] / [N, 2]

        @return GridMotion [1, 2, G_H, G_W] in model-resolution pixels
        """
        if kp.shape[1] == 4:
            kp = kp[:, 2:]
        index = kp.long()
        origin_motion = torch.cat([x_flow, y_flow], 1)
        extracted_motion = origin_motion[0, :, index[:, 0], index[:, 1]]
        kp = kp.permute(1, 0).float()
        concat_motion = torch.cat([kp[1:2, :], kp[0:1, :], extracted_motion], 0)

        motion, gridsMotion, _ = self.homoEstimate(concat_motion, kp)
        GridMotion = (self.forward(motion) + gridsMotion.squeeze(-1)) * cfg.FLOWC
        GridMotion = GridMotion.view(
            cfg.HEIGHT // cfg.PIXELS, cfg.WIDTH // cfg.PIXELS, 2
        )
        GridMotion = GridMotion.permute(2, 0, 1).unsqueeze(0)
        GridMotion = self.meidanPool(GridMotion)
        return GridMotion

    def loadWeights(self, path):
        pretrained = torch.load(path, map_location="cpu", weights_only=False)
        model_dict = self.state_dict()
        pretrained = {k: v for k, v in pretrained.items() if k in model_dict}
        assert len(pretrained) > 0
        model_dict.update(pretrained)
        assert len(model_dict) == len(pretrained), "MotionPro checkpoint mismatch"
        self.load_state_dict(model_dict)
