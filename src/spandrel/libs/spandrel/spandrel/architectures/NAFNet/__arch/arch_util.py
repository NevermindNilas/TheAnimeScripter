import torch
import torch.nn as nn
from torch import Tensor


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight: Tensor = nn.Parameter(torch.ones(channels))
        self.bias: Tensor = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        # TorchScript-compatible LayerNorm implementation
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + self.eps).sqrt()
        # Apply weight and bias scaling using unsqueeze operations
        weight_expanded = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        bias_expanded = self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        y = y * weight_expanded + bias_expanded
        return y
