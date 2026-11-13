
from typing import *

import torch
import triton
import triton.language as tl
from torch import Tensor

from .scatter_rank import scatter_rank_

__all__ = [
    "scatter_to_segment",
]



@triton.jit
def _scatter_to_segments_kernel(
    indices_ptr: tl.const,
    ranks_ptr: tl.const,
    offsets_ptr: tl.const,
    seg_indices_ptr: tl.pointer_type,
    N: int,
    BLOCK_SIZE: tl.constexpr,
):
    """Place each input index ``i`` at position ``offsets[index[i]] + ranks[i]``
    in ``seg_indices``.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    out = tl.load(indices_ptr + offs, mask=mask, other=0)
    rank = tl.load(ranks_ptr + offs, mask=mask, other=0)
    base = tl.load(offsets_ptr + out, mask=mask, other=0)
    pos = base + rank
    tl.store(seg_indices_ptr + pos, offs, mask=mask)


def scatter_to_segment(
    indices: Tensor,
    size: int,
) -> tuple[Tensor, Tensor]:
    """Convert indices for scatter to segments for reduce.

    Args:
        indices: (N,) integer tensor, denotes the output segment of each input. Must satisfy ``0 <= indices[i] < size``.
        size: the number of output segments.

    Returns:
        seg_indices: (N,) tensor (same dtype as ``indices``); concatenation of
            per-output input indices.
        seg_offsets: (M+1,) tensor (same dtype as ``indices``);
            ``seg_indices[seg_offsets[m]:seg_offsets[m+1]]`` are the inputs
            mapping to output m.

    NOTE: the order of inputs within each segment is not guaranteed (race-order due to atomics).
    """
    N = indices.shape[0]
    device = indices.device
    dtype = indices.dtype

    seg_offsets = torch.zeros((size + 1,), dtype=dtype, device=device)
    seg_indices = torch.empty((N,), dtype=dtype, device=device)

    ranks = scatter_rank_(seg_offsets[1:], indices)
    seg_offsets.cumsum_(dim=0)

    if N == 0:
        return seg_indices, seg_offsets

    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    _scatter_to_segments_kernel[grid](
        indices_ptr=indices,
        ranks_ptr=ranks,
        offsets_ptr=seg_offsets,
        seg_indices_ptr=seg_indices,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return seg_indices, seg_offsets


