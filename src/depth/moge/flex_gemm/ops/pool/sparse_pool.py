from typing import *

import torch
from torch import Tensor

from ... import kernels
from ..neighbor_cache import NeighborCache, build_neighbor_cache
from ..index_segment_reduce import index_segment_reduce
from ..utils import _broadcast_dim_arg, split_sparse_shape


__all__ = [
    "sparse_pool",
]


_REDUCE_MODES = ("sum", "mean", "max", "min", "prod")


def sparse_pool(
    feats: Tensor,
    input_coords: Tensor,
    shape: torch.Size,
    kernel_size: tuple[int, ...],
    stride: tuple[int, ...] | None = None,
    padding: tuple[int, ...] | None = None,
    reduce: Literal["sum", "mean", "max", "min", "prod"] = "mean",
    output_coords: Tensor | None = None,
    output_shape: torch.Size | None = None,
    neighbor_cache: NeighborCache | None = None,
) -> Tuple[Tensor, Tensor, torch.Size, NeighborCache]:
    """Strided sparse pooling (general path).

    Computes ``output[coord_out] = reduce_{v} input[coord_out * stride - padding + v]``
    over the set of input coordinates that actually exist, where ``v`` ranges
    over the dense kernel volume.

    Args:
        feats (Tensor): [M, C] input features.
        input_coords (Tensor): ``(M, B + Ds)`` input coordinates.
        shape (torch.Size): input dense shape
            ``(*batch_dims, S1, ..., SDs, C)`` — channel-last convention.
        kernel_size: tuple of length Ds.
        stride / padding: tuple of length Ds, or ``None``. Defaults:
            ``stride = kernel_size``, ``padding = (0,) * Ds`` (non-overlapping pool).
        reduce: one of ``sum`` / ``mean`` / ``max`` / ``min`` / ``prod``.
        output_coords / output_shape: passthrough to :func:`build_neighbor_cache`.
        neighbor_cache: if provided, must be consistent with the call (verified via
            :meth:`NeighborCache.assert_match`); its ``output_coords`` /
            ``output_shape`` are used.

    Returns:
        (output_feats, output_coords, output_shape, neighbor_cache).

    Note:
        Specialization for ``stride == kernel_size, padding == 0`` (perfect
        partition: every input maps to exactly one output) is available via
        :func:`_sparse_pool_perfect_partition`. Pending benchmark to decide
        whether the specialization is worth keeping.
    """
    if reduce not in _REDUCE_MODES:
        raise ValueError(f"reduce must be one of {_REDUCE_MODES}, got {reduce!r}")
    assert input_coords.is_contiguous(), "Coords should be contiguous"

    kernel_size = tuple(kernel_size)
    D_spatial = len(kernel_size)

    if stride is None:
        # Default to non-overlapping pool: stride == kernel_size.
        stride = kernel_size
    else:
        stride = tuple(stride)

    if padding is None:
        padding = (0,) * D_spatial
    else:
        padding = tuple(padding)

    assert len(stride) == D_spatial and len(padding) == D_spatial, (
        "kernel_size / stride / padding must all have the same length."
    )

    if neighbor_cache is None:
        # Cache lives in sparse-shape land; op truncates the channel-last
        # ``shape`` / ``output_shape`` before delegating.
        sparse_dim = input_coords.shape[1]
        sparse_in_shape  = split_sparse_shape(shape,        sparse_dim)
        sparse_out_shape = split_sparse_shape(output_shape, sparse_dim)
        neighbor_cache = build_neighbor_cache(
            input_coords, output_coords,
            submanifold=False,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            input_sparse_shape=sparse_in_shape,
            output_sparse_shape=sparse_out_shape,
        )
        output_coords = neighbor_cache.output_coords
        sparse_out_shape = neighbor_cache.output_sparse_shape
    else:
        assert output_coords is not None, (
            "When passing a precomputed neighbor_cache, output_coords must also be provided."
        )
        neighbor_cache.assert_match(
            input_coords=input_coords,
            output_coords=output_coords,
            is_transposed=False,
            kernel_size=kernel_size,
            stride=stride,
        )
        sparse_out_shape = neighbor_cache.output_sparse_shape

    output_feats = index_segment_reduce(
        feats,
        neighbor_cache.fwd_seg_indices,
        neighbor_cache.fwd_seg_offsets,
        reduce,
    )
    # Reassemble full channel-last output shape from sparse prefix + C.
    output_shape = torch.Size([*sparse_out_shape, *output_feats.shape[1:]])
    return output_feats, output_coords, output_shape, neighbor_cache


# ---------------------------------------------------------------------------
# Fixed-spatial-dim aliases.
#
# Compared with :func:`sparse_pool`, dim-related args (``kernel_size`` /
# ``stride`` / ``padding``) accept either a scalar ``int`` (broadcast to
# length ``D``) or a length-``D`` sequence. ``input_coords.shape[1]`` may
# exceed ``D``; the leading columns are batch dims.
# ---------------------------------------------------------------------------









