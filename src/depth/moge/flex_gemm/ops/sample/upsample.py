from typing import Literal, overload

import torch
from torch import Tensor

from ..index_select_add import index_select_add
from ..neighbor_cache import NeighborCacheT, build_neighbor_cache
from ..utils import _broadcast_dim_arg, split_sparse_shape


__all__ = [
    "sparse_upsample",
]


# --- nearest: no padding_mode ------------------------------------------------
@overload
def sparse_upsample(
    feats: Tensor,
    coords: Tensor,
    shape: torch.Size,
    scale_factor: tuple[int, ...],
    *,
    mode: Literal["nearest"] = ...,
    output_coords: Tensor | None = ...,
    output_shape: torch.Size | None = ...,
    neighbor_cache: NeighborCacheT | None = ...,
) -> tuple[Tensor, Tensor, torch.Size, NeighborCacheT]: ...


# --- bilinear: padding_mode + align_corners are meaningful -------------------
@overload
def sparse_upsample(
    feats: Tensor,
    coords: Tensor,
    shape: torch.Size,
    scale_factor: tuple[int, ...],
    *,
    mode: Literal["bilinear"],
    padding_mode: Literal["zeros", "normalize"] = ...,
    align_corners: bool = ...,
    output_coords: Tensor | None = ...,
    output_shape: torch.Size | None = ...,
    neighbor_cache: NeighborCacheT | None = ...,
) -> tuple[Tensor, Tensor, torch.Size, NeighborCacheT]: ...


def sparse_upsample(
    feats: Tensor,
    coords: Tensor,
    shape: torch.Size,
    scale_factor: tuple[int, ...],
    *,
    mode: Literal["nearest", "bilinear"] = "nearest",
    padding_mode: Literal["zeros", "normalize"] = "normalize",
    align_corners: bool = False,
    output_coords: Tensor | None = None,
    output_shape: torch.Size | None = None,
    neighbor_cache: NeighborCacheT | None = None,
) -> tuple[Tensor, Tensor, torch.Size, NeighborCacheT]:
    """Sparse upsample by an integer per-dim ``scale_factor``.

    The set of high-resolution output coordinates is the conv-transpose image
    of the input coords under ``kernel_size = stride = scale_factor`` with
    ``padding = 0``: each low-res coord ``c_in`` lights up the block of
    ``prod(scale_factor)`` high-res coords ``c_out = c_in * scale_factor +
    delta``. ``build_neighbor_cache(transpose=True)`` is used solely to
    materialise those output coords; the actual feature interpolation is
    then delegated to the (dense) sparse-grid-sample kernels:

    * ``mode="nearest"``  → :func:`index_select_add` over the cache's COO
      edges (each output voxel copies its parent input voxel; zero hash
      lookups).
    * ``mode="bilinear"`` → :func:`_sparse_grid_sample_linear` against the
      materialised output coords; falls back to a hash lookup because the
      ``2^D`` corners of each output voxel are not all covered by the
      conv-transpose cache.

    Args:
        feats: ``(M, C)`` low-resolution features.
        coords: ``(M, B + Ds)`` low-resolution integer coordinates (contiguous).
        shape: low-resolution dense shape
            ``(*batch_dims, S1, ..., SDs, C)`` — channel-last convention.
        scale_factor: per-spatial-dim upscale factor. Must be a tuple aligned
            with the spatial dims (``len(scale_factor) == Ds``).
        mode: ``"nearest"`` or ``"bilinear"`` (multilinear) interpolation.
        padding_mode: ``"zeros"`` / ``"normalize"`` — only consulted in
            ``"bilinear"`` mode (see :func:`sparse_grid_sample`).
        align_corners: only meaningful in ``"bilinear"`` mode. Mirrors the
            convention of :func:`torch.nn.functional.interpolate` (in the
            input's voxel-center coordinate space, where voxel ``i`` is
            centered at integer location ``i``):

            * ``False`` (default) — output voxel ``j`` maps to input coord
              ``x = (j + 0.5) / s - 0.5``. Shape-independent; for ``s=1``
              this is the identity ``x = j``.
            * ``True`` — output voxel ``j`` maps to ``x = j * (W - 1) / (s*W - 1)``
              per spatial dim, where ``W`` is the low-res spatial extent.
              Requires ``s*W > 1`` for the transform to be well-defined.

            Must be ``False`` for ``mode='nearest'``.
        output_coords: optional precomputed high-res coordinates.
        output_shape: optional high-res full channel-last shape (sparse_shape + dense C); 
            Used to trim customized out-of-bound ``output_coords``.
        neighbor_cache: optional precomputed :class:`NeighborCacheT`.

    Returns:
        ``(output_feats, output_coords, output_shape, neighbor_cache)``.
    """
    assert coords.is_contiguous(), "Coords should be contiguous"
    assert isinstance(scale_factor, tuple), (
        "scale_factor must be a tuple; len(scale_factor) defines the spatial "
        "dimensionality (no batch-dim assumption)."
    )
    scale_factor = tuple(int(s) for s in scale_factor)
    D_spatial = len(scale_factor)
    assert D_spatial >= 1, "scale_factor must have at least one spatial dim"
    assert all(s >= 1 for s in scale_factor), \
        f"scale_factor must be positive ints, got {scale_factor!r}"
    if mode not in ("nearest", "bilinear"):
        raise ValueError(f"Unsupported mode: {mode!r}")
    if padding_mode not in ("zeros", "normalize"):
        raise ValueError(f"Unsupported padding_mode: {padding_mode!r}")
    if mode == "nearest" and align_corners:
        raise ValueError("align_corners=True is only valid for mode='bilinear'.")

    # ------------------------------------------------------------------
    # Build / verify neighbor_cache purely to obtain ``output_coords``.
    # The cache's edge tables themselves are discarded — interpolation
    # happens against the dense grid-sample kernels below.
    # ------------------------------------------------------------------
    sparse_dim = coords.shape[1]
    sparse_in_shape = split_sparse_shape(shape, sparse_dim)
    if neighbor_cache is None:
        # ``padding=0`` is required (not just default) so the cache's
        # centered-kernel offset resolves to ``(r-1)//2``, making the
        # conv-transpose relation ``c_out = c_in*r + offset + delta``
        # span exactly ``c_in*r + {0, .., r-1}`` and tile the high-res
        # grid without gaps. Omitting it leaves offset=0, which clips
        # the low / high boundary coords.
        sparse_out_shape = split_sparse_shape(output_shape, sparse_dim)
        neighbor_cache = build_neighbor_cache(
            coords, output_coords,
            submanifold=False,
            kernel_size=scale_factor,
            stride=scale_factor,
            padding=(0,) * D_spatial,
            input_sparse_shape=sparse_in_shape,
            output_sparse_shape=sparse_out_shape,
            transpose=True,
        )
    else:
        neighbor_cache.assert_match(
            input_coords=coords,
            output_coords=output_coords,
            is_transposed=True,
            kernel_size=scale_factor,
            stride=scale_factor,
        )
    output_coords = neighbor_cache.output_coords
    sparse_out_shape = neighbor_cache.output_sparse_shape

    # ------------------------------------------------------------------
    # Interpolate.
    #
    # nearest: the cache already encodes the (input → output) parent
    # relation as COO edges (each output voxel has exactly one parent
    # under kernel_size=stride=scale_factor / padding=0). Reuse those
    # edges directly via index_select_add — no redundant hashmap lookup.
    #
    # bilinear: the 2^Ds corners of each output voxel are *not* all
    # parents under the conv-transpose adjacency (only one is), so the
    # cache is insufficient and we fall back to a fresh hashmap-backed
    # linear lookup.
    # ------------------------------------------------------------------
    device = output_coords.device
    if mode == "nearest":
        output_feats = index_select_add(
            feats,
            neighbor_cache.edge_in,
            neighbor_cache.edge_out,
            neighbor_cache.num_output_coords,
        )
    else:
        raise NotImplementedError("bilinear sparse upsample was removed from the vendored FlexGEMM")

    output_shape = torch.Size([*sparse_out_shape, *output_feats.shape[1:]])

    return output_feats, output_coords, output_shape, neighbor_cache


# ---------------------------------------------------------------------------
# Fixed-spatial-dim aliases.
#
# Compared with :func:`sparse_upsample`, ``scale_factor`` accepts either a
# scalar ``int`` (broadcast to length ``D``) or a length-``D`` sequence.
# ``coords.shape[1]`` may exceed ``D``; the leading columns are batch dims.
#
# ``@overload`` declarations don't transfer through ``functools.wraps`` (they
# live in ``typing._overload_registry`` per fully-qualified name), so we
# re-declare both overloads (nearest / bilinear mode) per alias.
# ---------------------------------------------------------------------------




# --- 2-D ---------------------------------------------------------------------


# --- 3-D ---------------------------------------------------------------------


# --- 4-D ---------------------------------------------------------------------



