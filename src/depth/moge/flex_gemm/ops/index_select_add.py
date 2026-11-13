from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.autograd import Function


class _IndexSelectAddFn(Function):
    """Sparse nearest-neighbour gather.

    Takes the raw ``(M,)`` int32 lookup result (``-1`` for misses) and
    materialises a zero-padded ``(M, C)`` output. Internally we resolve the
    miss mask to a positions tensor *once*, so the backward avoids the
    repeated boolean mask-selects (``indices[valid]``, ``grad_out[valid]``)
    that bool-indexing would incur.
    """

    @staticmethod
    def forward(ctx, feats: Tensor, src_index: Tensor, dst_index: Tensor, M: int) -> Tensor:
        N, C = feats.shape
        out = torch.zeros((M, C), device=feats.device, dtype=feats.dtype)
        if dst_index.numel():
            out.index_add_(0, dst_index, feats.index_select(0, src_index))
        ctx.save_for_backward(dst_index, src_index)
        ctx.N, ctx.C = N, C
        return out



def index_select_add(feat: Tensor, src: Tensor, dst: Tensor, M: int,) -> Tensor:
    return _IndexSelectAddFn.apply(feat, src, dst, M)