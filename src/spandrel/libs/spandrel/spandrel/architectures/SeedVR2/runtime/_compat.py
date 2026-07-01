import math
from dataclasses import dataclass

import torch
from torch import nn


def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    half = embedding_dim // 2
    exponent = -math.log(10000) * torch.arange(
        half, device=timesteps.device, dtype=torch.float32
    )
    exponent = exponent / max(half - 1, 1)
    emb = timesteps.float()[:, None] * torch.exp(exponent)[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if embedding_dim % 2:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if elementwise_affine else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        y = y.to(x.dtype)
        return y if self.weight is None else y * self.weight.to(y.dtype)


class DiagonalGaussianDistribution:
    def __init__(self, parameters: torch.Tensor):
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = self.logvar.clamp(-30.0, 20.0)

    def sample(self) -> torch.Tensor:
        return self.mean + torch.randn_like(self.mean) * torch.exp(0.5 * self.logvar)

    def mode(self) -> torch.Tensor:
        return self.mean


@dataclass
class AutoencoderKLOutput:
    latent_dist: DiagonalGaussianDistribution


@dataclass
class DecoderOutput:
    sample: torch.Tensor


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary_emb(freqs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    while freqs.ndim < t.ndim:
        freqs = freqs.unsqueeze(0)
    freqs = freqs.to(device=t.device, dtype=torch.float32)
    cos = freqs.cos().to(t.dtype)
    sin = freqs.sin().to(t.dtype)
    return (t * cos) + (_rotate_half(t) * sin)


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        freqs_for: str = "pixel",
        max_freq: int = 256,
        theta: int = 10000,
    ):
        super().__init__()
        if freqs_for == "pixel":
            inv = torch.linspace(1.0, max_freq / 2, dim // 2)
        else:
            inv = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("freqs", inv, persistent=False)

    def get_axial_freqs(self, *dims: int) -> torch.Tensor:
        parts = []
        for d in dims:
            pos = torch.arange(d, device=self.freqs.device).float()
            f = pos[:, None] * self.freqs[None, :]
            parts.append(torch.cat([f, f], dim=-1))
        expanded = []
        for axis, f in enumerate(parts):
            view = [1] * len(dims) + [f.shape[-1]]
            view[axis] = dims[axis]
            expanded.append(f.reshape(view).expand(*dims, f.shape[-1]))
        return torch.cat(expanded, dim=-1)
