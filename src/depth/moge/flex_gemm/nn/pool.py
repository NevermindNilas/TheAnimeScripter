"""Sparse pooling modules.

:class:`SubmanifoldPool` / :class:`SparsePool` wrap the corresponding op-layer
functions; ``2d`` / ``3d`` / ``4d`` aliases broadcast scalar ``kernel_size`` /
``stride`` / ``padding``.
"""
from typing import Literal, Tuple

import torch
from torch import Tensor
from torch import nn

from ..ops.neighbor_cache import NeighborCache
from ..ops.pool.sparse_pool import sparse_pool
from ..ops.utils import _broadcast_dim_arg


__all__ = [
    "SparsePool",
    "SparsePool3d",
]


_Reduce = Literal["sum", "mean", "max", "min", "prod"]


# ===========================================================================
# Submanifold pool (output coords == input coords)
# ===========================================================================










# ===========================================================================
# Strided / general sparse pool
# ===========================================================================


class SparsePool(nn.Module):
    """N-D strided sparse pooling.

    Wraps :func:`flex_gemm.ops.sparse_pool`.

    Args:
        kernel_size: per-spatial-dim kernel extents. ``len(kernel_size)``
            defines the spatial dimensionality.
        stride: per-spatial-dim stride. Defaults to ``kernel_size``
            (non-overlapping pool).
        padding: per-spatial-dim padding. Defaults to all-0.
        reduce: one of ``sum`` / ``mean`` / ``max`` / ``min`` / ``prod``.
    """

    def __init__(
        self,
        kernel_size: tuple[int, ...],
        stride: tuple[int, ...] | None = None,
        padding: tuple[int, ...] | None = None,
        reduce: _Reduce = "mean",
    ) -> None:
        super().__init__()
        kernel_size = tuple(int(k) for k in kernel_size)
        D = len(kernel_size)
        stride  = tuple(int(s) for s in stride)  if stride  is not None else kernel_size
        padding = tuple(int(p) for p in padding) if padding is not None else (0,) * D
        assert len(stride) == D and len(padding) == D, (
            "kernel_size / stride / padding must all share length D"
        )

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.reduce = reduce

    def forward(
        self,
        feats: Tensor,
        coords: Tensor,
        shape: torch.Size,
        output_coords: Tensor | None = None,
        output_shape: torch.Size | None = None,
        neighbor_cache: NeighborCache | None = None,
    ) -> Tuple[Tensor, Tensor, torch.Size, NeighborCache]:
        return sparse_pool(
            feats, coords, shape, self.kernel_size, self.stride, self.padding,
            self.reduce, output_coords, output_shape, neighbor_cache,
        )

    def extra_repr(self) -> str:
        return (
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, reduce={self.reduce!r}"
        )


# ---------------------------------------------------------------------------
# Fixed-spatial-dim aliases. ``kernel_size`` / ``stride`` / ``padding``
# accept either a scalar ``int`` (broadcast to length ``D``) or a length-``D``
# tuple. ``coords.shape[1]`` may exceed ``D`` in :meth:`forward`; the leading
# columns are batch dims.
# ---------------------------------------------------------------------------




class SparsePool3d(SparsePool):
    """3-D alias of :class:`SparsePool`."""

    def __init__(
        self,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] | None = None,
        padding: int | tuple[int, int, int] = 0,
        reduce: _Reduce = "mean",
    ) -> None:
        kernel_size = _broadcast_dim_arg(kernel_size, 3, "kernel_size")
        stride      = _broadcast_dim_arg(stride,      3, "stride")
        padding     = _broadcast_dim_arg(padding,     3, "padding")
        super().__init__(kernel_size, stride, padding, reduce)


