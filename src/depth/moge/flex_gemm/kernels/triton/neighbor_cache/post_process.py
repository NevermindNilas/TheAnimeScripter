from typing import *
import itertools
import math

import torch
from torch import Tensor
import triton
import triton.language as tl


__all__ = [
    "neighbor_map_gray_code_sort",
    "neighbor_map_valid_kernel",
]



@triton.jit
def _mask_gray_binary_triton_kernel(
    mask_ptr: tl.pointer_type,
    gray_ptr: tl.pointer_type,
    binary_ptr: tl.pointer_type,
    N: int,
    V: int,
    stride_n: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    offs_v = tl.arange(0, BLOCK_V)
    mask_v = offs_v < V

    mask_vals = tl.load(
        mask_ptr + offs_n[:, None] * stride_n + offs_v[None, :],
        mask=mask_n[:, None] & mask_v[None, :],
        other=0
    )
    valid = mask_vals > 0

    bit_weights = tl.full((BLOCK_V,), 1, dtype=tl.uint32) << offs_v
    gray = tl.sum(tl.where(valid, bit_weights[None, :], 0), axis=1)

    binary = gray
    binary ^= binary >> 1
    binary ^= binary >> 2
    binary ^= binary >> 4
    binary ^= binary >> 8
    binary ^= binary >> 16

    tl.store(gray_ptr + offs_n, gray, mask=mask_n)
    tl.store(binary_ptr + offs_n, binary, mask=mask_n)


def neighbor_map_gray_code_sort(
    neighbor_mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Post-process the neighbor map for masked implicit GEMM.

    Returns:
        gray_code: (N,) uint32 tensor of per-row kernel masks (bitset up to 32).
        sorted_idx: (N,) int64 tensor sorting rows by binary code.
    """
    if neighbor_mask.dim() != 2:
        raise ValueError("neighbor_map must be a 2D tensor")

    neighbor_mask = neighbor_mask.contiguous()
    N, V = neighbor_mask.shape

    if V > 32:
        raise ValueError(f"Masked implicit GEMM with more than 32 neighbors is not supported. Got V={V}.")

    if neighbor_mask.numel() == 0:
        gray_code = torch.empty((N,), dtype=torch.uint32, device=neighbor_mask.device)
        sorted_idx = torch.empty((N,), dtype=torch.int64, device=neighbor_mask.device)
        return gray_code, sorted_idx

    gray_code = torch.empty((N,), dtype=torch.uint32, device=neighbor_mask.device)
    binary_code = torch.empty((N,), dtype=torch.long, device=neighbor_mask.device)
    BLOCK_N = 64
    BLOCK_V = 32
    grid = (triton.cdiv(N, BLOCK_N),)
    _mask_gray_binary_triton_kernel[grid](
        mask_ptr=neighbor_mask,
        gray_ptr=gray_code,
        binary_ptr=binary_code,
        N=N,
        V=V,
        stride_n=neighbor_mask.stride(0),
        BLOCK_N=BLOCK_N,
        BLOCK_V=BLOCK_V,
    )
    
    sorted_idx = torch.argsort(binary_code)

    return gray_code, sorted_idx




@triton.jit
def _reduce_gray_code_triton_kernel(
    gray_code_ptr: tl.const,
    sorted_idx_ptr: tl.const,
    reduced_code_ptr: tl.pointer_type,
    seglen_ptr: tl.pointer_type,
    N: int,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_n = offs < N
    sorted_offs = tl.load(sorted_idx_ptr + offs, mask=mask_n, other=0)
    gray_code = tl.load(gray_code_ptr + sorted_offs, mask=mask_n, other=0).to(tl.uint32)

    reduced_code = tl.reduce_or(gray_code, axis=0) 
    seglen = tl.sum((reduced_code >> tl.arange(0, 32)) & 1, axis=0).to(tl.int32) # popcount of acc. Inline ASM does not improve speed.

    tl.store(reduced_code_ptr + pid, reduced_code)
    tl.store(seglen_ptr + pid + 1, seglen)


@triton.jit
def _scatter_reduced_code_kernel(
    reduced_code_ptr: tl.const,
    seg_ptr: tl.const,
    out_ptr: tl.pointer_type,
    num_blocks: int,
    BLOCK_BITS: tl.constexpr,
):
    pid = tl.program_id(0)
    mask = pid < num_blocks
    code = tl.load(reduced_code_ptr + pid, mask=mask, other=0).to(tl.uint32)
    seg_start = tl.load(seg_ptr + pid, mask=mask, other=0).to(tl.int32)
    bits = tl.arange(0, BLOCK_BITS)
    bit_set = (code >> bits.to(tl.uint32)) & 1
    write_pos = tl.cumsum(bit_set.to(tl.int32), axis=0) - 1
    do_store = (bit_set != 0) & mask
    pos = seg_start + write_pos
    tl.store(out_ptr + pos, bits, mask=do_store)


def neighbor_map_valid_kernel(
    gray_code: Tensor,
    sorted_idx: Tensor,
    block_size: int,
) -> tuple[Tensor, Tensor]:
    """
    Build valid kernel indices for masked implicit GEMM.

    Returns:
        valid_kernel_idx: (L,) int32 tensor containing valid kernel indices.
        valid_kernel_seg: (num_blocks + 1,) int32 tensor containing segment boundaries.
    """
    if gray_code.dim() != 1 or sorted_idx.dim() != 1:
        raise ValueError("gray_code and sorted_idx must be 1D tensors")
    if block_size <= 0 or (block_size & (block_size - 1)) != 0:
        raise ValueError("block_size must be a positive power of 2")
    if gray_code.dtype not in (torch.int32, torch.uint32):
        raise ValueError("gray_code must be int32 or uint32")
    assert gray_code.is_contiguous() and sorted_idx.is_contiguous(), "gray_code and sorted_idx must be contiguous"

    N = gray_code.numel()

    num_blocks: int = triton.cdiv(N, block_size)
    valid_kernel_seg = torch.zeros((num_blocks + 1,), dtype=torch.long, device=gray_code.device)

    if N == 0 or num_blocks == 0:
        valid_kernel_idx = torch.empty((0,), dtype=torch.long, device=gray_code.device)
        return valid_kernel_idx, valid_kernel_seg

    reduced_code = torch.empty((num_blocks,), dtype=torch.long, device=gray_code.device)
    grid = (num_blocks,)
    _reduce_gray_code_triton_kernel[grid](
        gray_code_ptr=gray_code,
        sorted_idx_ptr=sorted_idx,
        reduced_code_ptr=reduced_code,
        seglen_ptr=valid_kernel_seg,
        N=N,
        BLOCK_SIZE=block_size,
        num_warps=4 if block_size >= 128 else 2,
    )

    valid_kernel_seg.cumsum_(dim=0)
    total_valid = valid_kernel_seg[-1].item()
    valid_kernel_idx = torch.empty((total_valid,), dtype=torch.long, device=gray_code.device)
    if total_valid == 0:
        return valid_kernel_idx, valid_kernel_seg
    
    _scatter_reduced_code_kernel[grid](
        reduced_code_ptr=reduced_code,
        seg_ptr=valid_kernel_seg,
        out_ptr=valid_kernel_idx,
        num_blocks=num_blocks,
        BLOCK_BITS=32,
        num_warps=1,
    )

    return valid_kernel_idx, valid_kernel_seg

