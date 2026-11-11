from typing import *

import torch
import triton
import triton.language as tl
from torch import Tensor


__all__ = [
    "scatter_rank_",
]


@triton.jit
def _scatter_rank_kernel(
    indices_ptr: tl.const,
    counts_ptr: tl.pointer_type,
    ranks_ptr: tl.pointer_type,
    N: int,
    BLOCK_SIZE: tl.constexpr,
):
    """For each input i, atomically increment ``counts[inverse[i]]`` and record
    the pre-increment value as ``ranks[i]``. After the kernel, ``counts[m]`` is
    the number of inputs mapping to output m, and ``ranks[i]`` is the unique
    rank in ``[0, counts[inverse[i]])`` of input i within its output's race.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    out = tl.load(indices_ptr + offs, mask=mask, other=0)
    # atomic_add returns the value before the increment, which is the
    # 0-based rank of this input within its output segment.
    rank = tl.atomic_add(counts_ptr + out, 1, mask=mask)
    tl.store(ranks_ptr + offs, rank, mask=mask)

    
def scatter_rank_(
    counts: Tensor,
    indices: Tensor,
) -> tuple[Tensor, Tensor]:
    """Atomic-add based count + per-input rank computation.

    Args:
        counts: (size,) tensor (same dtype as ``indices``); must be zero-initialized.
        indices: (N,) integer tensor, denotes the output segment of each input. Must satisfy ``0 <= indices[i] < size``.

    Returns:
        counts: (size,) tensor (same dtype as ``indices``); ``counts[m]`` =
            number of inputs with ``index == m``.
        ranks: (N,) tensor (same dtype as ``indices``); ``ranks[i]`` in
            ``[0, counts[index[i]])`` is a unique rank of input i within its
            output's segment (race-order).
    """

    N = indices.shape[0]
    device, dtype = indices.device, indices.dtype
    ranks = torch.empty((N,), dtype=dtype, device=device)

    if N == 0:
        return counts, ranks

    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    _scatter_rank_kernel[grid](
        indices_ptr=indices,
        counts_ptr=counts,
        ranks_ptr=ranks,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return ranks
