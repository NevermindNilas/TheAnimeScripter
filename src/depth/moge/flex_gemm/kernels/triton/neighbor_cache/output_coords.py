from typing import *
import itertools
import math

import torch
from torch import Tensor
import triton
import triton.language as tl

from ..hashmap import (
    _vec_load,
    _vec_hash_32bit,
    _reduce_any,
    _reduce_all,
    _vec_pack_little_endian_to_int32,
    hashmap_unique,
    HASHMAP_LOAD_FACTOR,
)
from .neighbor_map import (
    _make_conv_delta_inline,
    _make_4d_vec_inline,
    _hashmap_prepare_offs_masks_inline,
)
from ..utils import floor_divmod, pad_to_size_along_dim, _floor_divmod_inline


__all__ = [
    "get_output_coords_kernel_size_dilation",
    "get_output_coords_strict_downsample",
]










def get_output_coords_kernel_size_dilation(
    input_coords: Tensor,
    kernel_size: tuple[int, ...],
    stride: tuple[int, ...] | None = None,
    dilation: tuple[int, ...] | None = None,
    offset: tuple[int, ...] | None = None,
    boundary: tuple[tuple[int, int], ...] | None = None,
    transposed: bool = False,
) -> tuple[Tensor, Tensor | None]:
    """Compute output coords for strided sparse convolution using Triton GPU kernels.

    Non-transposed mode (``transposed=False``, default — sparse conv forward):
        For each input coordinate ``coord_in`` and each kernel delta ``delta``:
            ``candidate_out = (coord_in - offset - delta) // stride``
        is a valid output coordinate when divisibility holds for every spatial
        dimension and ``candidate_out`` lies within ``boundary``.

    Transposed mode (``transposed=True`` — sparse conv-transpose):
        For each input coord and each kernel delta:
            ``candidate_out = coord_in * stride + offset + delta``
        is always divisibility-valid; only the boundary check applies.

    The unique set of valid candidate output coordinates is returned together
    with the backward neighbor map ``(N, V) -> M``.

    Args:
        input_coords: (N, D) int8 / int16 / int32 tensor of input voxel coordinates.
        kernel_size: tuple of ints (length ≤ D), the convolution kernel size.
        stride: tuple of ints (length ≤ D) or None (defaults to all-1).
        dilation: tuple of ints (length ≤ D) or None (defaults to all-1).
        offset: tuple of ints (length ≤ D) or None (defaults to all-0).
        boundary: tuple of (min, max) pairs (length ≤ D) or None.
            When provided, output coords are filtered to ``boundary[d][0] <= x < boundary[d][1]``.
            When None, no boundary filtering is applied.
            
    Returns:
        output_coords: (M, D) tensor — unique output coordinates.
        bwd_neighbor_map: (N, V) int32 tensor or ``None`` — ``bwd[n, v] = m`` means
            input coord ``n`` maps to output coord ``m`` via kernel index ``v``; -1 if none.
    """
    assert input_coords.dtype in (torch.int8, torch.int16, torch.int32), (
        f"input_coords must be int8, int16 or int32, got {input_coords.dtype}."
    )
    N, orig_D = input_coords.shape
    coord_dtype = input_coords.dtype
    device = input_coords.device

    # Normalize all params to length orig_D (prefix-pad with neutral values)
    kernel_size = (1,) * (orig_D - len(kernel_size)) + tuple(kernel_size)
    dilation    = (1,) * (orig_D - len(dilation))    + tuple(dilation)    if dilation is not None else (1,) * orig_D
    stride      = (1,) * (orig_D - len(stride))      + tuple(stride)      if stride   is not None else (1,) * orig_D
    offset      = (0,) * (orig_D - len(offset))      + tuple(offset)      if offset   is not None else (0,) * orig_D
    if boundary is not None:
        # Prefix dims are batch / extra dims not constrained by boundary, so use infinite bounds for them.
        iinfo = torch.iinfo(input_coords.dtype)
        boundary = ((iinfo.min, iinfo.max),) * (orig_D - len(boundary)) + tuple(boundary)
    
    # Fast path 1: downsample
    if all(k == s for k, s in zip(kernel_size, stride)) and all(d == 1 for d in dilation) and transposed == False:
        # When kernel_size == stride and dilation == 1 and not transposed, strict downsample: each input coord maps to at most 1 output coord    
        return get_output_coords_strict_downsample(
            input_coords=input_coords,
            kernel_size=kernel_size,
            offset=offset,
            boundary=boundary,
        )

    # Fast path 2: fuse expansion and uniqueness
    raise NotImplementedError("only strict-downsample output coords are kept in the vendored FlexGEMM")


# =============================================================================
# Strict-downsample output coords  (stride == kernel_size, dilation == 1).
#
# In this regime each input coord maps to *exactly one* (output_coord,
# kernel_index) pair AND there is a closed-form per-dim formula — no kernel
# meshgrid, no lookup tables required:
#
#     Let R[d] = (k[d] - 1) // 2  (number of "negative" kernel positions),
#         s[d] = c[d] - o[d] + R[d]                  (per-dim shifted input)
#
#     candidate[d]    = floor(s[d] / k[d])           (= output coord)
#     delta_plus_R[d] = s[d] - candidate[d] * k[d]   (= residue ∈ [0, k))
#     edge_kernel    = Σ_d delta_plus_R[d] * kstride[d]
#     edge_in[i]     = i                             (no boundary)
#
# Derivation: from coord_in = coord_out * k + offset + delta with the
# meshgrid kernel positions ``delta ∈ [-R, k - R)`` (matching
# ``_make_conv_delta_inline``), the unique valid ``delta`` per input is
#
#     delta = ((c - o + R) mod k) - R
#
# and so
#
#     coord_out = (c - o - delta) / k = floor((c - o + R) / k).
# =============================================================================


@triton.jit
def _strict_downsample_output_coords_kernel(
    input_coords_ptr,                            # (N, D) coord dtype
    shift_stride_ptr,                            # (2*D,) coord dtype = shift[D] ++ stride[D] (stride == kernel_size)
    boundary_ptr: tl.pointer_type | None,        # (2*D,) coord dtype, or None
    hashmap_ptr,                                 # (hashmap_size,) int32, init -1
    candidates_ptr,                              # (N, D) coord dtype — canonical lanes only
    edge_kernel_ptr,                             # (N,)   int32
    indices_ptr,                                 # (N,)   int32 — canonical input idx for each valid lane
    is_canonical_ptr,                            # (N,)   int8
    valid_mask_ptr: tl.pointer_type | None,      # (N,)   int8, or None (only when boundary)
    hashmap_size: int,
    N: int,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Fused strict-downsample expand + unique.

    Each input ``i`` produces a single candidate via the closed-form
    ``floor((coord - offset + R) / k)``. Valid lanes CAS-insert their input
    index into a shared hashmap; on collision we reload ``coords_in[prev_idx]``
    and reapply the same formula to reconstruct the colliding lane's
    candidate, then full-vector match to resolve duplicates. Only canonical
    lanes (``found_idx == i``) write to ``candidates_ptr``.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    offs_D = tl.arange(0, D)

    coord = tl.load(
        input_coords_ptr + offs[:, None] * D + offs_D[None, :],
        mask=mask[:, None], other=0,
    )                                                                            # (BLOCK, D)
    shift_vec  = tl.load(shift_stride_ptr + offs_D)                              # (D,)  R - offset
    stride_vec = tl.load(shift_stride_ptr + D + offs_D)                          # (D,)  kernel_size

    # shifted = coord - offset + R  (the "rounding-to-nearest-cell" form).
    shifted = coord + shift_vec[None, :]

    # Floor-divmod consistent with Python (torch's rounding_mode='floor');
    # Triton's ``//`` is truncated-toward-zero for signed integers.
    candidate, delta_plus_R = _floor_divmod_inline(shifted, stride_vec)

    # Mixed-radix encoding weights kstride[d] = prod(ks[d+1:]).
    # With ks padded to length D using 1s, prod(ks) = V and so
    #     kstride[d] = V / cumprod_inclusive(ks)[d].
    # Padded dims have ks=1 (kstride=1) but delta_plus_R=0 there, so they
    # contribute nothing to edge_kernel.
    ks_i32      = stride_vec.to(tl.int32)
    kstride_vec = tl.cumprod(ks_i32, axis=0, reverse=True) // ks_i32                # (D,)
    edge_kernel = tl.sum(delta_plus_R.to(tl.int32) * kstride_vec[None, :], axis=1)  # (BLOCK,)
    tl.store(edge_kernel_ptr + offs, edge_kernel, mask=mask)

    # ---- Validity ----
    if valid_mask_ptr is not None:
        bmin = tl.load(boundary_ptr + offs_D)
        bmax = tl.load(boundary_ptr + D + offs_D)
        in_bounds = (
            (tl.min((candidate >= bmin[None, :]).to(tl.int8), axis=1) > 0) &
            (tl.min((candidate <  bmax[None, :]).to(tl.int8), axis=1) > 0)
        )
        valid = mask & in_bounds
        tl.store(valid_mask_ptr + offs, valid.to(tl.int8), mask=mask)
    else:
        valid = mask

    # ---- Hash + CAS-loop (mirrors _hashmap_unique_kernel_32bit) ----
    # ``out_idx = i`` directly (V = 1 in strict-downsample); on a collision the
    # stored ``prev_idx`` is another input index, so reloading
    # ``input_coords[prev_idx]`` lets us reapply the per-dim formula and
    # reproduce that lane's candidate without an extra GMEM round-trip.
    hash_val = _vec_hash_32bit(candidate, D=D)

    SLOT_BIT_MASK = tl.cast(hashmap_size - 1, tl.int32)
    TAG_BIT_MASK  = (~SLOT_BIT_MASK) & 0x7FFF_FFFF

    my_tag    = hash_val & TAG_BIT_MASK
    store_val = my_tag | offs

    found_idx   = offs
    active      = valid
    target_slot = hash_val & SLOT_BIT_MASK

    probes = 0
    while _reduce_any(active) & (probes < hashmap_size):
        # Inactive lanes feed ``expected = -2`` so their CAS is a no-op.
        prev = tl.atomic_cas(hashmap_ptr + target_slot, tl.where(active, -1, -2), store_val)

        # CAS succeeded (prev == -1): this lane owns the slot and its index is
        # canonical. CAS failed: probe the existing entry for a duplicate.
        active &= (prev >= 0)

        prev_tag = prev & TAG_BIT_MASK
        prev_idx = prev & SLOT_BIT_MASK
        tag_match = active & (prev_tag == my_tag)

        # Reproduce prev's candidate: same shift/stride, just a different coord.
        prev_coord = tl.load(
            input_coords_ptr + prev_idx[:, None] * D + offs_D[None, :],
            mask=tag_match[:, None], other=0,
        )
        prev_shifted = prev_coord + shift_vec[None, :]
        prev_candidate, _ = _floor_divmod_inline(prev_shifted, stride_vec)

        full_match = tag_match & _reduce_all(prev_candidate == candidate, axis=-1)
        found_idx  = tl.where(full_match, prev_idx, found_idx)
        active    &= ~full_match

        target_slot = (target_slot + 1) & SLOT_BIT_MASK
        probes += 1

    tl.device_assert(tl.max(active) == 0, "strict_downsample: hashmap full -- caller mis-sized the table")

    # ---- Write per-lane outputs ----
    is_canonical_val = valid & (found_idx == offs)

    # ``indices`` is meaningful only for valid lanes.
    tl.store(indices_ptr + offs, found_idx, mask=valid)
    tl.store(is_canonical_ptr + offs, is_canonical_val.to(tl.int8), mask=mask)

    # Store candidate only for canonical lanes -- skips writes for duplicates
    # and out-of-boundary lanes.
    tl.store(
        candidates_ptr + offs[:, None] * D + offs_D[None, :],
        candidate, mask=is_canonical_val[:, None],
    )


def get_output_coords_strict_downsample(
    input_coords: Tensor,
    kernel_size: tuple[int, ...],
    offset: tuple[int, ...] | None = None,
    boundary: tuple[tuple[int, int], ...] | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Specialised non-transposed output coords for **strict downsampling**:
    ``stride == kernel_size`` and ``dilation == 1``.

    Closed-form per-dim formula (no delta meshgrid, no lookup table):
    with ``R[d] = (k[d] - 1) // 2``,

        candidate[d]    = floor((c[d] - offset[d] + R[d]) / k[d])
        delta_plus_R[d] = (c[d] - offset[d] + R[d]) mod k[d]   ∈ [0, k[d])
        edge_kernel     = Σ_d delta_plus_R[d] * kstride[d]

    where ``kstride[d] = prod(k[d+1:])`` is the mixed-radix encoding stride
    of the meshgrid kernel index (matching ``_make_conv_delta_inline`` in
    ``neighbor_map.py``).

    Args:
        input_coords: ``(N, D)`` int8 / int16 / int32 tensor.
        kernel_size:  tuple of ints (length ≤ D). ``stride`` is implicitly
                      equal to ``kernel_size`` and ``dilation`` is implicitly all-1.
        offset:       tuple of ints (length ≤ D) or None (defaults to all-0).
        boundary:     tuple of ``(min, max)`` pairs (length ≤ D) or None.

    Returns:
        output_coords: ``(M, D)`` tensor of unique valid output coords.
        edge_in:       ``(E,)`` int32 — row index into ``input_coords``.
        edge_out:      ``(E,)`` int32 — row index into ``output_coords``.
        edge_kernel:   ``(E,)`` int32 — meshgrid kernel index ``v ∈ [0, V)``.
    """
    assert input_coords.dtype in (torch.int8, torch.int16, torch.int32), (
        f"input_coords must be int8, int16 or int32, got {input_coords.dtype}."
    )
    N, orig_D = input_coords.shape
    coord_dtype = input_coords.dtype
    device = input_coords.device

    # Normalize to length orig_D (prefix-pad with neutral values).
    kernel_size = (1,) * (orig_D - len(kernel_size)) + tuple(kernel_size)
    offset      = ((0,) * (orig_D - len(offset)) + tuple(offset)
                   if offset is not None else (0,) * orig_D)
    V = math.prod(kernel_size)

    empty = lambda: (
        torch.empty((0, orig_D), dtype=coord_dtype, device=device),
        torch.empty((0,), dtype=torch.int32, device=device),
        torch.empty((0,), dtype=torch.int32, device=device),
        torch.empty((0,), dtype=torch.int32, device=device),
    )
    if N == 0 or V == 0:
        return empty()

    # Pad spatial dim to next power of 2 (>= 4) for vectorised loads. Padded
    # dims use stride=1, shift=0 — so they contribute 0 to ``edge_kernel`` and
    # pass through the floor-div as 0 → 0.
    D: int = max(4, triton.next_power_of_2(orig_D))
    shift_padded  = [(k - 1) // 2 - o for k, o in zip(kernel_size, offset)] + [0] * (D - orig_D)
    stride_padded = list(kernel_size) + [1] * (D - orig_D)

    # Pack shift ++ stride (++ optional boundary) into a single host->device
    # transfer. ``kstride`` is no longer materialised on host: the kernel
    # derives it from ``stride`` via ``tl.cumprod``.
    flat = shift_padded + stride_padded
    if boundary is not None:
        iinfo = torch.iinfo(coord_dtype)
        boundary_full = ((iinfo.min, iinfo.max),) * (orig_D - len(boundary)) + tuple(boundary)
        boundary_full = boundary_full + ((0, 1),) * (D - orig_D)
        flat += [b[0] for b in boundary_full] + [b[1] for b in boundary_full]
    meta = torch.tensor(flat, dtype=coord_dtype, device=device)
    shift_stride_dev = meta[: 2 * D]
    boundary_dev     = meta[2 * D : 4 * D] if boundary is not None else None

    # ---- Pad spatial dim of input_coords ----
    input_coords_D = pad_to_size_along_dim(input_coords, dim=1, size=D, value=0, side='right').contiguous()

    candidates  = torch.empty((N, D), dtype=coord_dtype, device=device)
    edge_kernel = torch.empty((N,),   dtype=torch.int32, device=device)
    indices      = torch.empty((N,),   dtype=torch.int32, device=device)
    is_canonical = torch.empty((N,),   dtype=torch.int8,  device=device)
    if boundary is not None:
        valid_mask  = torch.empty((N,), dtype=torch.int8, device=device)
    else:
        valid_mask = None

    # Hashmap sized so ``i < hashmap_size`` (we encode the input index in the
    # low ``log2(hashmap_size)`` bits of each slot).
    hashmap_size = triton.next_power_of_2(int(N / HASHMAP_LOAD_FACTOR))
    assert hashmap_size <= (1 << 30), (
        "Hash map size exceeds 2^30, which is the limit for our 32-bit implementation."
    )
    hashmap = torch.full((hashmap_size,), -1, dtype=torch.int32, device=device)

    BLOCK = 256
    grid = (triton.cdiv(N, BLOCK),)
    _strict_downsample_output_coords_kernel[grid](
        input_coords_ptr=input_coords_D,
        shift_stride_ptr=shift_stride_dev,
        boundary_ptr=boundary_dev,
        hashmap_ptr=hashmap,
        candidates_ptr=candidates,
        edge_kernel_ptr=edge_kernel,
        indices_ptr=indices,
        is_canonical_ptr=is_canonical,
        valid_mask_ptr=valid_mask,
        hashmap_size=hashmap_size,
        N=N,
        D=D,
        BLOCK=BLOCK,
    )

    # ---- Gather canonical / valid lanes ----
    if valid_mask is not None:
        valid_indices = valid_mask.nonzero(as_tuple=True)[0].to(torch.int32)
        if valid_indices.numel() == 0:
            return empty()

    canonical_indices = is_canonical.nonzero(as_tuple=True)[0].to(torch.int32)
    num_uniques = canonical_indices.shape[0]
    output_coords = candidates[:, :orig_D][canonical_indices].contiguous()

    # inverse_full[canonical_idx] = dense unique position; edge_out then maps
    # each valid lane to its canonical lane's unique position.
    inverse_full = torch.empty(N, dtype=torch.int32, device=device)
    inverse_full[canonical_indices] = torch.arange(num_uniques, device=device, dtype=torch.int32)

    if valid_mask is not None:
        edge_in     = valid_indices
        edge_out    = inverse_full[indices[valid_indices]]
        edge_kernel = edge_kernel[valid_indices]
    else:
        edge_in     = torch.arange(N, dtype=torch.int32, device=device)
        edge_out    = inverse_full[indices]
    return output_coords, edge_in, edge_out, edge_kernel


# =============================================================================
# Fused expand-+-unique 4D kernel.
#
# Combines ``_expand_output_coords_4d_triton_kernel`` and ``hashmap_unique``
# into a single Triton kernel:
#
#   1. Each ``(n, v)`` lane computes its candidate output coord exactly as the
#      stand-alone expand kernel does.
#   2. Valid lanes hash the candidate and CAS-insert ``out_idx = n*V + v`` into
#      a shared hashmap (sized for ``N*V`` entries).
#   3. On CAS failure the slot holds some other lane's ``prev_idx``; we decode
#      ``prev_n, prev_v = floor_divmod(prev_idx, V)`` and *recompute* that
#      lane's candidate from ``coords_in[prev_n]`` and kernel index ``prev_v``
#      — no extra global memory traffic for the candidate buffer is needed.
#      A full-vector match against our own candidate resolves duplicates.
#   4. After the hashmap resolves, lanes that ended up canonical (``found_idx
#      == out_idx``) write their candidate to ``out_candidates_ptr``. Non-
#      canonical lanes skip the store entirely, saving N*V*D bandwidth.
#
# The host then gathers canonical positions to build ``output_coords`` and uses
# the indirection array to build ``edge_out`` for the valid lanes.
# =============================================================================




