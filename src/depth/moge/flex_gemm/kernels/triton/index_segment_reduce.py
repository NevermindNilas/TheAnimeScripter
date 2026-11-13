"""
index_segment_reduce: fused index_select + segment_reduce
"""


from typing import *

import torch
import triton
import triton.language as tl
from torch import Tensor


__all__ = [
    "index_segment_reduce_sum_mean",
]


# Reduce-mode constants for the constexpr selector in the kernel.
# Inside @triton.jit we compare REDUCE_MODE against literal ints (triton
# disallows referencing Python globals from kernels). ``max`` / ``min`` have
# their own kernel (``_index_segment_reduce_extrema_kernel``) because backward
# needs the per-channel arg-extremum.
_REDUCE_SUM = 0
_REDUCE_MEAN = 1


@triton.jit
def _index_segment_reduce_sum_mean_kernel(
    data_ptr,            # (N, C) input
    indices_ptr,         # (L,)    each entry is a row of data
    offsets_ptr,         # (M+1,)
    out_ptr,             # (M, C)
    C: int,
    stride_data_n: int,
    stride_data_c: int,
    stride_out_m: int,
    stride_out_c: int,
    BLOCK_C: tl.constexpr,
    REDUCE_MODE: tl.constexpr,  # 0 = sum, 1 = mean
):
    """One program per (segment m, channel block). Streams through the segment's
    rows directly out of ``data`` (no intermediate ``gathered`` tensor)."""
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)

    start = tl.load(offsets_ptr + pid_m)
    end   = tl.load(offsets_ptr + pid_m + 1)
    length = end - start

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

    # Stream over segment rows. ``length`` varies per program; this dynamic-trip
    # loop is the price we pay for direct gather inside the reduce kernel.
    for i in range(length):
        row = tl.load(indices_ptr + start + i)
        row_ptr = data_ptr + row * stride_data_n + offs_c * stride_data_c
        v = tl.load(row_ptr, mask=mask_c, other=0.0).to(tl.float32)
        acc += v

    if REDUCE_MODE == 1:  # MEAN
        # Guard against empty segments (length==0) — leave acc at 0.
        denom = tl.where(length > 0, length, 1).to(tl.float32)
        acc /= denom

    out_row_ptr = out_ptr + pid_m * stride_out_m + offs_c * stride_out_c
    tl.store(out_row_ptr, acc, mask=mask_c)


def index_segment_reduce_sum_mean(
    data: Tensor,
    indices: Tensor,
    offsets: Tensor,
    reduce: str,
) -> Tensor:
    """Forward launcher for the ``sum`` / ``mean`` segment reductions.

    For each output row m, computes
    ``reduce({ data[indices[i]] : offsets[m] <= i < offsets[m+1] })``,
    materializing no ``(L, C)`` intermediate tensor.

    Used by :class:`flex_gemm.ops.pool.index_segment_reduce._IndexSegmentReduceFn`
    for the forward pass; ``max`` / ``min`` are handled separately by
    :func:`index_segment_reduce_extrema` so that backward has the per-channel
    arg-extremum it needs.

    Empty segments produce ``0``.
    """
    assert data.dim() == 2, "data must be 2D (N, C)"
    assert indices.dim() == 1 and offsets.dim() == 1
    assert offsets.shape[0] >= 1

    reduce_mode = {"sum": _REDUCE_SUM, "mean": _REDUCE_MEAN}[reduce]

    N, C = data.shape
    M = offsets.shape[0] - 1

    out = torch.empty((M, C), dtype=data.dtype, device=data.device)
    if M == 0:
        return out

    # Smallest power of 2 >= C, capped at 1024 to bound register pressure.
    BLOCK_C = min(triton.next_power_of_2(C), 1024)

    grid = (M, triton.cdiv(C, BLOCK_C))
    _index_segment_reduce_sum_mean_kernel[grid](
        data_ptr=data,
        indices_ptr=indices,
        offsets_ptr=offsets,
        out_ptr=out,
        C=C,
        stride_data_n=data.stride(0),
        stride_data_c=data.stride(1),
        stride_out_m=out.stride(0),
        stride_out_c=out.stride(1),
        BLOCK_C=BLOCK_C,
        REDUCE_MODE=reduce_mode,
    )
    return out


# -----------------------------------------------------------------------------
# Backward support for index_segment_reduce
# -----------------------------------------------------------------------------












