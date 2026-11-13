from typing import *
import itertools
import math

import torch
from torch import Tensor
import triton
import triton.language as tl

from ..hashmap import (
    hashmap_build, 
    _hashmap_lookup_inline_32bit, 
    _vec_load, 
)
from ..utils import pad_to_size_along_dim


__all__ = [
    "build_neighbor_map_from_kernel_size_dilation",
]


# ================ helper inline functions ========================
@triton.jit
def _hashmap_find_store_neighbor_map_inline(
    hashmap_ptr: tl.pointer_type,
    hashmap_size: int,
    coords_in_ptr: tl.pointer_type,
    coords_out_ptr: tl.pointer_type,
    delta_vec: tl.tensor,
    offs_M: tl.tensor,
    mask_M: tl.tensor,
    offs_V: tl.tensor,
    mask_V: tl.tensor,
    coord_stride_vec: tl.tensor | None,
    coord_offset_vec: tl.tensor | None,
    neighbor_map_ptr: tl.pointer_type,
    V: int,
    D: tl.constexpr,
    SYMMETRIC: tl.constexpr = False,
):  
    """Inline triton JIT function to find neighbor indices and store to neighbor map.
    
    """
    mask_MV = mask_M[:, None] & mask_V[None, :]
    coord_vec = _vec_load(coords_out_ptr + offs_M * D, mask_M, D)
    if coord_stride_vec is not None:
        coord_vec *= coord_stride_vec 
    if coord_offset_vec is not None:
        coord_vec += coord_offset_vec
    neighbor_coord_vec = coord_vec[:, None, :] + delta_vec[None, :, :].to(coord_vec.dtype)
    found_idx = _hashmap_lookup_inline_32bit(
        hashmap_ptr, hashmap_size,
        coords_in_ptr, 
        neighbor_coord_vec,
        mask=mask_MV,
        D=D
    )
    tl.store(
        neighbor_map_ptr + offs_M[:, None] * V + offs_V[None, :], 
        found_idx, 
        mask=mask_MV
    )
    if SYMMETRIC:
        symmetric_mask = (found_idx >= 0) & mask_MV & (offs_V <= V // 2)
        tl.store(
            neighbor_map_ptr + found_idx * V + (V - 1 - offs_V[None, :]),
            offs_M[:, None],
            mask=symmetric_mask,
        )


@triton.jit
def _hashmap_prepare_offs_masks_inline(
    pid_m: int,
    pid_v: int,
    M: int,
    V: int,
    BLOCK_M: tl.constexpr,
    BLOCK_V: tl.constexpr,
) -> tuple[tl.tensor, tl.tensor, tl.tensor, tl.tensor]:
    offs_M = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_M = offs_M < M
    offs_V = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_V = offs_V < V
    return offs_M, mask_M, offs_V, mask_V



@triton.jit
def _make_conv_delta_inline(
    idx: tl.tensor,
    kernel_size_vec: tl.tensor,
    kernel_dilation_vec: tl.tensor,
    dtype=tl.int32
) -> tl.tensor:
    idx = idx.to(dtype)
    kernel_size_vec = kernel_size_vec.to(dtype)
    kernel_dilation_vec = kernel_dilation_vec.to(dtype)

    kernel_stride_vec = tl.cumprod(kernel_size_vec, 0, reverse=True) // kernel_size_vec
    delta = ((idx[:, None] // kernel_stride_vec) % kernel_size_vec - ((kernel_size_vec - 1) >> 1)) * kernel_dilation_vec
    return delta


# ===== Arbitrary kernel delta ======


# =========== 4D ============
@triton.jit
def _make_4d_vec_inline(x0, x1, x2, x3, dtype=tl.int32) -> tl.tensor:
    idx = tl.arange(0, 4)
    vec = tl.full((4,), x0, dtype=dtype)
    vec = tl.where(idx == 1, x1, vec)
    vec = tl.where(idx == 2, x2, vec)
    vec = tl.where(idx == 3, x3, vec)
    return vec


@triton.jit
def _hashmap_build_neighbor_map_kernel_size_dilation_4d_triton_kernel(
    hashmap_ptr: tl.tensor,
    hashmap_size: int,
    coords_in_ptr: tl.const,
    coords_out_ptr: tl.const,
    neighbor_map_ptr: tl.pointer_type,
    coord_stride_offset_ptr: tl.pointer_type | None,
    K0: int, K1: int, K2: int, K3: int,
    KD0: int, KD1: int, KD2: int, KD3: int,
    M: int,
    BLOCK_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    INT16_DELTA: tl.constexpr,
    SYMMETRIC: tl.constexpr,
):  
    D: tl.constexpr = 4
    V = K0 * K1 * K2 * K3
    pid_M = tl.program_id(0) 
    pid_V = tl.program_id(1)

    offs_D = tl.arange(0, D)
    offs_M, mask_M, offs_V, mask_V = _hashmap_prepare_offs_masks_inline(
        pid_M, pid_V, M, V, BLOCK_M, BLOCK_V
    )

    # Make constexpr neighbor deltas.
    delta_dtype = tl.int16 if INT16_DELTA else tl.int32
    kernel_size_vec = _make_4d_vec_inline(K0, K1, K2, K3, dtype=delta_dtype)
    kernel_dilation_vec = _make_4d_vec_inline(KD0, KD1, KD2, KD3, dtype=delta_dtype)
    delta_vec = _make_conv_delta_inline(
        offs_V, 
        kernel_size_vec=kernel_size_vec,
        kernel_dilation_vec=kernel_dilation_vec,
        dtype=delta_dtype
    ) # (BLOCK_V, D)
    
    if coord_stride_offset_ptr is not None:
        coord_stride_vec = tl.load(coord_stride_offset_ptr + offs_D)
        coord_offset_vec = tl.load(coord_stride_offset_ptr + D + offs_D)
    else:
        coord_stride_vec = None
        coord_offset_vec = None

    _hashmap_find_store_neighbor_map_inline(
        hashmap_ptr=hashmap_ptr,
        hashmap_size=hashmap_size,
        coords_in_ptr=coords_in_ptr,
        coords_out_ptr=coords_out_ptr,
        delta_vec=delta_vec,
        offs_M=offs_M,
        mask_M=mask_M,
        offs_V=offs_V,
        mask_V=mask_V,
        coord_stride_vec=coord_stride_vec,
        coord_offset_vec=coord_offset_vec,
        neighbor_map_ptr=neighbor_map_ptr,
        V=V,
        D=4,
        SYMMETRIC=SYMMETRIC,
    )


# =========== ND ===========



def build_neighbor_map_from_kernel_size_dilation(
    input_coords: Tensor,
    output_coords: Tensor | None,
    kernel_size: tuple[int, ...],
    stride: tuple[int, ...] | None = None,
    dilation: tuple[int, ...] | None = None,
    offset: tuple[int, ...] | None = None,
    hashmap: Tensor | None = None,
    symmetric: bool = None,
) -> Tensor:
    """Build neighbor map for constexpr kernel defined by `kernel_size` and `dilation`.
    Supports up to 8D.
    
    Args:
        input_coords: (N, D) int8 / int16 / int32 tensor. Input coordinates. 
            Prefix dimensions will be viewed as batch dimensions.
        output_coords: (M, D) int8 / int16 / int32 tensor. If None, output_coords will be the same as input_coords.
        kernel_size: (<=D,) tuple of integers, the size of the convolution kernel.
        dilation: (<=D,) tuple of integers, the dilation of the convolution kernel.
            If kernel_size or dilation is shorter than D, the first unspecified coordinate dimensions will be treated as batch dimensions.
        stride: (<=D,) tuple of integers, the stride of the convolution kernel.
            If stride is shorter than D, the first unspecified coordinate dimensions will be treated as batch dimensions.
        offset: (<=D,) tuple of integers, the offset of the convolution kernel.
            If offset is shorter than D, the first unspecified coordinate dimensions will be treated as batch dimensions.
        hashmap: (N,) int32 tensor, mapping from flat key to index in coords. If None, it will be built from coords.
    
    Returns:
        neighbor_map: (M, V) int32 tensor, the neighbor map. Each element is the index of the neighbor in coords, or -1 if not found.
    """
    assert input_coords.dtype in (torch.int8, torch.int16, torch.int32), (
        f"input_coords must be int8, int16 or int32. Got {input_coords.dtype}."
    )
    orig_D = input_coords.shape[1]
    if output_coords is not None:
        assert output_coords.dtype == input_coords.dtype, f"output_coords must have the same dtype as input_coords. Got {output_coords.dtype} and {input_coords.dtype}."
        assert output_coords.shape[1] == input_coords.shape[1], f"output_coords must have the same number of dimensions as input_coords. Got {output_coords.shape[1]} and {input_coords.shape[1]}."
    assert len(kernel_size) <= orig_D and (stride is None or len(stride) <= orig_D) and (dilation is None or len(dilation) <= orig_D) and (offset is None or len(offset) <= orig_D), (
        f"kernel_size, stride, dilation and offset must have length less than or equal to the number of coordinate dimensions."
        f"Got kernel_size with length {len(kernel_size)}, stride with length {len(stride) if stride is not None else 'None'}, dilation with length {len(dilation) if dilation is not None else 'None'}, offset with length {len(offset) if offset is not None else 'None'}"
        f"but coords has {orig_D} dimensions."
    )
    device = input_coords.device
    
    # Pad prefix batch dimensions (Prepend to left) to align kernel to coords
    kernel_size = (1,) * (orig_D - len(kernel_size)) + tuple(kernel_size)
    dilation = (1,) * (orig_D - len(dilation)) + tuple(dilation) if dilation is not None else (1,) * orig_D
    stride = (1,) * (orig_D - len(stride)) + tuple(stride) if stride is not None else (1,) * orig_D
    offset = (0,) * (orig_D - len(offset)) + tuple(offset) if offset is not None else (0,) * orig_D
    if symmetric is None:
        symmetric = output_coords is None and all(k % 2 == 1 for k in kernel_size) and all(o == 0 for o in offset) and all(s == 1 for s in stride)

    # Pad dimensions to next power of 2 (Append to right)
    D = max(4, triton.next_power_of_2(orig_D))
    input_coords = pad_to_size_along_dim(input_coords, dim=1, size=D, value=0, side='right')
    output_coords = input_coords if output_coords is None else pad_to_size_along_dim(output_coords, dim=1, size=D, side='right')
    M = output_coords.shape[0]
    kernel_size_D = tuple(kernel_size) + (1,) * (D - orig_D)
    kernel_dilation_D = tuple(dilation) + (1,) * (D - orig_D)
    stride_D = tuple(stride) + (1,) * (D - orig_D)
    offset_D = tuple(offset) + (0,) * (D - orig_D)
    V = math.prod(kernel_size_D)

    # Build hashmap for input coords if not provided.
    if hashmap is None:
        hashmap = hashmap_build(input_coords)
    
    INT16_DELTA = V < 32768
    if any(s != 1 for s in stride_D) or any(o != 0 for o in offset_D):
        coord_stride_offset_tensor = torch.tensor(list(stride_D) + list(offset_D), dtype=input_coords.dtype, device=device)
    else:
        coord_stride_offset_tensor = None
    
    # Build neighbor map
    #   NOTE: If symmetric, need to prefill -1 since the kernel may overlook. 
    #   Otherwise all -1 will covered by the kernel, so no need to prefill (saving a little bit of time).
    if symmetric: 
        neighbor_map = torch.full((M, V), -1, dtype=torch.int32, device=device)
    else:
        neighbor_map = torch.empty((M, V), dtype=torch.int32, device=device)

    if symmetric:
        # For symmetric, only search half of the kernel since the other half is symmetric.
        BLOCK_V = min(32, triton.next_power_of_2((V + 1) // 2))
        BLOCK_M = max(1, 256 // BLOCK_V)
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv((V + 1) // 2, BLOCK_V))
    else:
        BLOCK_V = min(32, triton.next_power_of_2(V))
        BLOCK_M = max(1, 256 // BLOCK_V)
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(V, BLOCK_V))
    
    if D == 4:
        _hashmap_build_neighbor_map_kernel_size_dilation_4d_triton_kernel[grid](
            hashmap_ptr=hashmap,
            hashmap_size=hashmap.shape[0],
            coords_in_ptr=input_coords,
            coords_out_ptr=output_coords,
            neighbor_map_ptr=neighbor_map,
            coord_stride_offset_ptr=coord_stride_offset_tensor,
            K0=kernel_size_D[0], K1=kernel_size_D[1], K2=kernel_size_D[2], K3=kernel_size_D[3],
            KD0=kernel_dilation_D[0], KD1=kernel_dilation_D[1], KD2=kernel_dilation_D[2], KD3=kernel_dilation_D[3],
            M=M,
            BLOCK_V=BLOCK_V,
            BLOCK_M=BLOCK_M,
            INT16_DELTA=INT16_DELTA,
            SYMMETRIC=symmetric,
        )
    else:
        raise NotImplementedError("only 4-D (batch + 3 spatial) neighbor maps are kept in the vendored FlexGEMM")
    return neighbor_map




# ========================================================================================
# ============================= backward neighbor map ====================================
# ========================================================================================





