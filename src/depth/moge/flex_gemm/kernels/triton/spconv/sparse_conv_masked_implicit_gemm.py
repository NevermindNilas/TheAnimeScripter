from typing import *
import math
import torch
import triton
import triton.language as tl
from ....autotuner import triton_autotune
from ..utils import autotune_size_bucket
from . import config
from .... import config as _global_config
from .sparse_conv_implicit_gemm import sparse_conv_implicit_gemm_kernel


@triton_autotune(
    configs=config.autotune_config,
    key=['LOGN', 'LOGM', 'Ci', 'Co', 'V', 'allow_tf32'],
)
@triton.heuristics({
    'valid_kernel': lambda args: args['valid_kernel'](args['B1']),
    'valid_kernel_seg': lambda args: args['valid_kernel_seg'](args['B1']),
    'HAS_BIAS': lambda args: args['bias'] is not None,
})
@triton.jit
def sparse_conv_masked_implicit_gemm_kernel(
    input,
    weight,
    bias,
    neighbor,
    sorted_idx,
    output,
    # Tensor dimensions
    M, LOGN, LOGM, Ci, Co, V: tl.constexpr,
    # Meta-parameters
    B1: tl.constexpr,   # Block size for M dimension
    B2: tl.constexpr,   # Block size for Co dimension
    BK: tl.constexpr,   # Block size for K dimension (V * Ci)
    HAS_BIAS: tl.constexpr,  # Whether bias is present
    allow_tf32: tl.constexpr,  # Allow TF32 precision for matmuls
    # Huristic parameters
    valid_kernel,
    valid_kernel_seg,
    # Specialize
    TRANSPOSE_WEIGHT: tl.constexpr = False,  # Whether to transpose the weight matrix
    FLIP_WEIGHT: tl.constexpr = False,  # Whether to flip the weight matrix along V dimension
):
    """
    Indice convolution forward kernel using masked implicit GEMM.
    
    Args:
        input (pointer): A pointer to the input tensor of shape (N, Ci)
        weight (pointer): A pointer to the weight tensor of shape (Co, V, Ci)
        bias (pointer): A pointer to the bias tensor of shape (Co)
        neighbor (pointer): A pointer to the neighbor tensor of shape (M, V)
        sorted_idx (pointer): A pointer to the sorted index tensor of shape (M,)
        valid_kernel (pointer): A pointer to the valid neighbor index tensor of shape (L,)
        valid_kernel_seg (pointer): A pointer to the valid neighbor index segment tensor of shape (BLOCK_M + 1,)
        output (pointer): A pointer to the output tensor of shape (M, Co)
    """
    block_id = tl.program_id(axis=0)
    block_dim_co = tl.cdiv(Co, B2)
    block_id_co = block_id % block_dim_co
    block_id_m = block_id // block_dim_co
    
    # Create pointers for submatrices of A and B.
    num_k = tl.cdiv(Ci, BK)  # Number of blocks in K dimension
    valid_kernel_start = tl.load(valid_kernel_seg + block_id_m)
    valid_kernel_seglen = tl.load(valid_kernel_seg + block_id_m + 1) - valid_kernel_start
    offset_m = block_id_m * B1 + tl.arange(0, B1)
    m_mask = offset_m < M
    offset_sorted_m = tl.load(sorted_idx + offset_m, mask=m_mask, other=0)  # (B1,)
    offset_co = (block_id_co * B2 + tl.arange(0, B2)) % Co                  # (B2,)
    offset_k = tl.arange(0, BK)                                             # (BK,)
    
    # Create a block of the output matrix C.
    accumulator = tl.zeros((B1, B2), dtype=tl.float32)
    
    # Iterate along V*Ci dimension.
    for k in range(num_k * valid_kernel_seglen):
        v = k // num_k
        bk = k % num_k
        v = tl.load(valid_kernel + valid_kernel_start + v)
        # Calculate pointers to weight matrix.
        weight_v = V - 1 - v if FLIP_WEIGHT else v
        if not TRANSPOSE_WEIGHT:
            weight_ptr = weight + (offset_co[None, :] * V * Ci) + (weight_v * Ci) + (bk * BK + offset_k[:, None])      # (BK, B2)
        else:
            weight_ptr = weight + (offset_co[None, :]) + (weight_v * Co) + ((bk * BK + offset_k[:, None]) * V * Co)    # (BK, B2)
        # Calculate pointers to input matrix.
        neighbor_offset = tl.load(neighbor + offset_sorted_m * V + v)                             # (B1,)
        input_ptr = input + bk * BK + (neighbor_offset[:, None].to(tl.int64) * Ci + offset_k[None, :])         # (B1, BK)
        # Load the next block of input and weight.
        neigh_mask = neighbor_offset != -1
        k_mask = offset_k < Ci - bk * BK
        input_block = tl.load(input_ptr, mask=neigh_mask[:, None] & k_mask[None, :], other=0.0)
        weight_block = tl.load(weight_ptr, mask=k_mask[:, None], other=0.0)
        # Accumulate along the K dimension.
        accumulator = tl.dot(input_block, weight_block, accumulator,
                             input_precision='tf32' if allow_tf32 else 'ieee')                      # (B1, B2)

    # Add bias on the fp32 accumulator (before down-casting) so that bias
    # contributes at full precision even under fp16 / bf16 AMP.
    if HAS_BIAS:
        co_mask = block_id_co * B2 + tl.arange(0, B2) < Co
        bias_block = tl.load(bias + offset_co, mask=co_mask, other=0.0)
        accumulator += bias_block[None, :]

    c = accumulator.to(input.type.element_ty)

    # Write back the block of the output matrix with masks.
    out_offset_m = offset_sorted_m
    out_offset_co = block_id_co * B2 + tl.arange(0, B2)
    out_ptr = output + (out_offset_m[:, None].to(tl.int64) * Co + out_offset_co[None, :])
    out_mask = m_mask[:, None] & (out_offset_co[None, :] < Co)
    tl.store(out_ptr, c, mask=out_mask)
    



def sparse_conv_fwd_masked_implicit_gemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    fwd_neighbor_map: torch.Tensor,
    fwd_sorted_idx: torch.Tensor,
    fwd_valid_kernel: Callable[[int], torch.Tensor],
    fwd_valid_kernel_seg: Callable[[int], torch.Tensor],
    allow_tf32: Optional[bool] = None,
) -> torch.Tensor:
    if allow_tf32 is None:
        allow_tf32 = _global_config.SPCONV_ALLOW_TF32
    assert input.shape[1] == weight.shape[2], "Incompatible dimensions"
    assert input.is_contiguous(), "Matrix input must be contiguous"
    assert weight.is_contiguous(), "Matrix weight must be contiguous"
    assert fwd_neighbor_map.is_contiguous(), "Matrix neighbor must be contiguous"
    N, M, Ci, Co, V = input.shape[0], fwd_neighbor_map.shape[0], input.shape[1], weight.shape[0], weight.shape[1]
    LOGN = autotune_size_bucket(N)
    LOGM = autotune_size_bucket(M)
    # Allocate output matrix output.
    output = torch.empty((M, Co), device=input.device, dtype=input.dtype)
    # Launch the kernel.
    grid = lambda META: (triton.cdiv(Co, META['B2']) * triton.cdiv(M, META['B1']),)
    sparse_conv_masked_implicit_gemm_kernel[grid](
        input, weight, bias, fwd_neighbor_map, fwd_sorted_idx, output,
        M, LOGN, LOGM, Ci, Co, V,
        valid_kernel=fwd_valid_kernel,
        valid_kernel_seg=fwd_valid_kernel_seg,
        allow_tf32=allow_tf32,
    )
    return output




