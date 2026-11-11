"""Sparse upsample module."""
from typing import Literal, Tuple

import torch
from torch import Tensor
from torch import nn

from ..ops.neighbor_cache import NeighborCacheT
from ..ops.sample.upsample import sparse_upsample
from ..ops.utils import _broadcast_dim_arg


__all__ = [
    "SparseUpsample",
    "SparseUpsample3d",
]


class SparseUpsample(nn.Module):
    """N-D sparse upsample by an integer per-spatial-dim ``scale_factor``.

    Wraps :func:`flex_gemm.ops.sparse_upsample`. Stateless except for the
    stored ``scale_factor`` / ``mode`` / ``padding_mode``.

    Args:
        scale_factor: per-spatial-dim upscale factor. ``len(scale_factor)``
            defines the spatial dimensionality.
        mode: ``"nearest"`` or ``"bilinear"``.
        padding_mode: only consulted for ``mode="bilinear"``; see
            :func:`flex_gemm.ops.sparse_upsample`.
    """

    def __init__(
        self,
        scale_factor: tuple[int, ...],
        mode: Literal["nearest", "bilinear"] = "nearest",
        padding_mode: Literal["zeros", "normalize"] = "normalize",
    ) -> None:
        super().__init__()
        self.scale_factor = tuple(int(s) for s in scale_factor)
        self.mode = mode
        self.padding_mode = padding_mode

    def forward(
        self,
        feats: Tensor,
        coords: Tensor,
        shape: torch.Size,
        output_coords: Tensor | None = None,
        output_shape: torch.Size | None = None,
        neighbor_cache: NeighborCacheT | None = None,
    ) -> Tuple[Tensor, Tensor, torch.Size, NeighborCacheT]:
        return sparse_upsample(
            feats, coords, shape, self.scale_factor,
            mode=self.mode, padding_mode=self.padding_mode,
            output_coords=output_coords, output_shape=output_shape,
            neighbor_cache=neighbor_cache,
        )

    def extra_repr(self) -> str:
        s = f"scale_factor={self.scale_factor}, mode={self.mode!r}"
        if self.mode == "bilinear":
            s += f", padding_mode={self.padding_mode!r}"
        return s


# ---------------------------------------------------------------------------
# Fixed-spatial-dim aliases. ``scale_factor`` accepts either a scalar ``int``
# (broadcast to length ``D``) or a length-``D`` tuple.
# ---------------------------------------------------------------------------




class SparseUpsample3d(SparseUpsample):
    """3-D alias of :class:`SparseUpsample`."""

    def __init__(
        self,
        scale_factor: int | tuple[int, int, int],
        mode: Literal["nearest", "bilinear"] = "nearest",
        padding_mode: Literal["zeros", "normalize"] = "normalize",
    ) -> None:
        scale_factor = _broadcast_dim_arg(scale_factor, 3, "scale_factor")
        super().__init__(scale_factor, mode, padding_mode)


