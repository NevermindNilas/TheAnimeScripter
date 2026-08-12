"""Deep trajectory smoother + iterative Jacobi kernel smoothing
(Smoother.py + IterativeSmooth.py, device-agnostic)."""

import numpy as np
import torch
import torch.nn as nn


def gauss(t, r=0, window_size=3):
    if np.abs(r - t) > window_size:
        return 0
    else:
        return np.exp((-9 * (r - t) ** 2) / window_size**2)


def generateSmooth(originPath, kernel=None, repeat=20):
    # B, 1, T, H, W; B, 6, T, H, W
    smooth = originPath

    temp_smooth_3 = originPath[:, :, 3:-3, :, :]

    if kernel is None:
        kernel = torch.Tensor([gauss(i) for i in range(-3, 4)]).to(originPath.device)
        kernel = torch.cat([kernel[:3], kernel[4:]])
        kernel = kernel.unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        kernel = kernel.repeat(*originPath.shape)

    abskernel = torch.abs(kernel)
    lambda_t = 100

    for _ in range(repeat):
        temp_smooth = torch.zeros_like(smooth, device=smooth.device)
        temp_smooth_0 = smooth[:, :, 0:-6, :, :] * kernel[:, 0:1, 3:-3, :, :] * lambda_t
        temp_smooth_1 = smooth[:, :, 1:-5, :, :] * kernel[:, 1:2, 3:-3, :, :] * lambda_t
        temp_smooth_2 = smooth[:, :, 2:-4, :, :] * kernel[:, 2:3, 3:-3, :, :] * lambda_t

        temp_smooth_4 = smooth[:, :, 4:-2, :, :] * kernel[:, 3:4, 3:-3, :, :] * lambda_t
        temp_smooth_5 = smooth[:, :, 5:-1, :, :] * kernel[:, 4:5, 3:-3, :, :] * lambda_t
        temp_smooth_6 = smooth[:, :, 6:, :, :] * kernel[:, 5:6, 3:-3, :, :] * lambda_t

        temp_smooth[:, :, 3:-3, :, :] = (
            temp_smooth_0
            + temp_smooth_1
            + temp_smooth_2
            + temp_smooth_3
            + temp_smooth_4
            + temp_smooth_5
            + temp_smooth_6
        ) / (1 + lambda_t * torch.sum(abskernel[:, :, 3:-3, :, :], dim=1, keepdim=True))

        # fmt: off
        # 0
        temp = smooth[:, :, 1:4, :, :]
        temp_smooth[:, :, 0, :, :] = (torch.sum(kernel[:, 3:, 0, :, :].unsqueeze(
            1) * temp, 2) * lambda_t + originPath[:, :, 0, :, :]) / (1 + lambda_t * torch.sum(abskernel[:, 3:, 0, :, :].unsqueeze(1), 2))
        # 1
        temp = torch.cat([smooth[:, :, :1, :, :], smooth[:, :, 2:5, :, :]], 2)
        temp_smooth[:, :, 1, :, :] = (torch.sum(kernel[:, 2:, 1, :, :].unsqueeze(
            1) * temp, 2) * lambda_t + originPath[:, :, 1, :, :]) / (1 + lambda_t * torch.sum(abskernel[:, 2:, 1, :, :].unsqueeze(1), 2))
        # 2
        temp = torch.cat([smooth[:, :, :2, :, :], smooth[:, :, 3:6, :, :]], 2)
        temp_smooth[:, :, 2, :, :] = (torch.sum(kernel[:, 1:, 2, :, :].unsqueeze(
            1) * temp, 2) * lambda_t + originPath[:, :, 2, :, :]) / (1 + lambda_t * torch.sum(abskernel[:, 1:, 2, :, :].unsqueeze(1), 2))
        # -1
        temp = smooth[:, :, -4:-1]
        temp_smooth[:, :, -1, :, :] = (torch.sum(kernel[:, :3, -1, :, :].unsqueeze(1) * temp, 2) * lambda_t +
                                       originPath[:, :, -1, :, :]) / (1 + lambda_t * torch.sum(abskernel[:, :3, -1, :, :].unsqueeze(1), 2))
        # -2
        temp = torch.cat([smooth[:, :, -5:-2, :, :], smooth[:, :, -1:, :, :]], 2)
        temp_smooth[:, :, -2, :, :] = (torch.sum(kernel[:, :4, -2, :, :].unsqueeze(1) * temp, 2) * lambda_t +
                                       originPath[:, :, -2, :, :]) / (1 + lambda_t * torch.sum(abskernel[:, :4, -2, :, :].unsqueeze(1), 2))
        # -3
        temp = torch.cat([smooth[:, :, -6:-3, :, :], smooth[:, :, -2:, :, :]], 2)
        temp_smooth[:, :, -3, :, :] = (torch.sum(kernel[:, :5, -3, :, :].unsqueeze(1) * temp, 2) * lambda_t +
                                       originPath[:, :, -3, :, :]) / (1 + lambda_t * torch.sum(abskernel[:, :5, -3, :, :].unsqueeze(1), 2))
        # fmt: on

        smooth = temp_smooth

    return smooth


class Smoother(nn.Module):
    def __init__(self, inplanes=2, embeddingSize=64, hiddenSize=64, kernel=5):
        super().__init__()
        self.embedding = nn.Sequential(nn.Linear(inplanes, embeddingSize), nn.ReLU())
        self.pad = kernel // 2
        self.conv1 = nn.Conv3d(
            embeddingSize, embeddingSize, (kernel, 3, 3), padding=(self.pad, 1, 1)
        )
        self.conv3 = nn.Conv3d(
            embeddingSize, embeddingSize, (kernel, 3, 3), padding=(self.pad, 1, 1)
        )
        self.conv2 = nn.Conv3d(
            embeddingSize, embeddingSize, (kernel, 3, 3), padding=(self.pad, 1, 1)
        )
        self.decoder = nn.Linear(embeddingSize, 12, bias=True)
        self.scale = nn.Linear(embeddingSize, 1, bias=True)
        self.activation = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, trajectory):
        """
        @param trajectory: Unstable trajectory with shape [B, 2, T, H, W]

        @return kernel: dynamic smooth kernel with shape [B, 12, T, H, W]
        """

        trajectory = trajectory.permute(0, 2, 3, 4, 1)
        embedding_trajectory = self.embedding(trajectory).permute(0, 4, 1, 2, 3)
        hidden = embedding_trajectory
        hidden = self.relu(self.conv1(hidden))
        hidden = self.relu(self.conv3(hidden))
        hidden = self.relu(self.conv2(hidden))
        kernel = self.activation(
            self.decoder(hidden.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        )
        kernel = (
            self.scale(hidden.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3) * kernel
        )
        return kernel

    def kernelSmooth(self, kernel, path, repeat=50):
        smooth_x = generateSmooth(
            path[:, 0:1, :, :, :], kernel[:, 0:6, :, :, :], repeat
        )
        smooth_y = generateSmooth(
            path[:, 1:2, :, :, :], kernel[:, 6:12, :, :, :], repeat
        )
        return smooth_x, smooth_y

    def smoothPath(self, path, repeat=50, minV=None, maxV=None):
        """
        @param path [1, 2, T, G_H, G_W] cumulative grid trajectory (any device)
        @param minV/maxV optional normalization bounds. The kernel-prediction
            net sees the normalized path, so callers smoothing a long video in
            temporal chunks must pass the WHOLE video's min/max here — chunk-
            local normalization would feed each chunk a differently-scaled
            trajectory and produce different kernels chunk-wide.
        @return smoothed path, same shape

        generateSmooth's boundary handling needs T >= 12; shorter clips are
        returned unsmoothed (nothing meaningful to stabilize).
        """
        if path.shape[2] < 12:
            return path
        min_v = path.amin() if minV is None else minV
        path = path - min_v
        max_v = (path.amax() if maxV is None else maxV - min_v) + 1e-5
        path = path / max_v
        kernel = self.forward(path)
        smooth_x, smooth_y = self.kernelSmooth(kernel, path, repeat)
        return torch.cat([smooth_x, smooth_y], 1) * max_v + min_v

    def loadWeights(self, path):
        pretrained = torch.load(path, map_location="cpu", weights_only=False)
        model_dict = self.state_dict()
        pretrained = {k: v for k, v in pretrained.items() if k in model_dict}
        assert len(pretrained) > 0
        assert len(model_dict) == len(pretrained), "Smoother checkpoint mismatch"
        self.load_state_dict(pretrained)
