from typing import *
from numbers import Number
import itertools

import torch
import triton
import triton.language as tl
from torch import Tensor


def get_gpu_name():
    return torch.cuda.get_device_name()


def get_platform_name():
    if torch.cuda.is_available():
        if getattr(torch.version, 'hip', None) is not None:
            return 'hip'
        return 'cuda'
    return 'unknown'
    



def autotune_size_bucket(n: int) -> int:
    """Coarse log-scale bucket used as an autotune cache key for size-like
    arguments (e.g. M, N over which the kernel iterates).
    """
    if n <= 1:
        return 0
    return min(max((int(n).bit_length() - 1) // 2, 4), 8)
    

def get_autotune_config(
    default: List[triton.Config] = None,
    platform: Dict[str, List[triton.Config]] = None,
    device: Dict[str, List[triton.Config]] = None,
) -> List[triton.Config]:
    """
    Get the autotune configuration for the current platform and device.
    """
    if device is not None:
        gpu_name = get_gpu_name()
        for key, value in device.items():
            if key.lower() in gpu_name.lower():
                return value
    
    if platform is not None:
        platform_name = get_platform_name()
        for key, value in platform.items():
            if key.lower() in platform_name.lower():
                return value
    
    if default is None:
        raise ValueError("No autotune configuration found for the current platform and device.")
    return default






def pad_to_size_along_dim(x: Tensor, dim: int | tuple[int, ...], size: int | tuple[int, ...], value: Number = 0., side: Literal['left', 'right'] = 'right') -> Tensor:
    "Pad the specified dimension of the tensor to the next power of two with zeros."
    if isinstance(dim, int):
        dim = (dim,)
    if isinstance(size, int):
        size = (size,)
    if len(dim) == 1 and len(size) > 1:
        size = size * len(dim)
    if len(dim) > 1 and len(size) == 1:
        size = size * len(dim)
    assert len(dim) == len(size), f"dim and size must have the same length. Got {len(dim)} and {len(size)} respectively."
    
    pad_size = [0] * x.dim()
    for d, s in zip(dim, size):
        pad_size[d] = max(0, s - x.shape[d])
    if any(p > 0 for p in pad_size):
        x = torch.nn.functional.pad(
            x, 
            tuple(itertools.chain.from_iterable((0, p) if side == 'right' else (p, 0) for p in reversed(pad_size))), 
            value=value
        )
    return x

# -----------------------------------------------------------------------------
# Fused integer floor-division + remainder.
# -----------------------------------------------------------------------------

@triton.jit
def _floor_divmod_inline(x: tl.tensor, d: tl.tensor):
    # Triton's ``//`` is truncated toward zero for signed ints; convert to
    # Python / torch ``rounding_mode='floor'`` semantics.
    trunc_q = x // d
    trunc_r = x - trunc_q * d
    need_adj = (trunc_r != 0) & ((x < 0) ^ (d < 0))
    adj = need_adj.to(x.dtype)
    q = trunc_q - adj
    r = trunc_r + adj * d
    return q, r


@triton.jit
def _floor_divmod_kernel(
    x_ptr,            # (N,) input integers
    q_ptr,            # (N,) output  q = floor(x / d)
    r_ptr,            # (N,) output  r = x - q * d   ∈ [0, d)  (when d > 0)
    d,                # scalar divisor (matches x's dtype)
    N: int,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    q, r = _floor_divmod_inline(x, d)
    tl.store(q_ptr + offs, q, mask=mask)
    tl.store(r_ptr + offs, r, mask=mask)


def floor_divmod(x: Tensor, d: int) -> Tuple[Tensor, Tensor]:
    """Fused integer floor-division and remainder for a 1-D integer tensor.

    Equivalent to::

        q = torch.div(x, d, rounding_mode='floor')
        r = x - q * d                          # == torch.remainder(x, d)

    but computed in a single Triton kernel (one load + two stores per element)
    instead of the multi-pass torch implementation.

    Args:
        x: 1-D integer tensor on CUDA.
        d: non-zero Python int divisor.

    Returns:
        ``(q, r)`` — both with the same dtype, shape, and device as ``x``.
    """
    assert x.is_cuda, "floor_divmod requires a CUDA tensor"
    assert x.dtype in (torch.int8, torch.int16, torch.int32, torch.int64), (
        f"floor_divmod requires an integer tensor, got {x.dtype}"
    )
    assert d != 0, "floor_divmod divisor must be non-zero"
    x = x.contiguous()
    N = x.numel()
    q = torch.empty_like(x)
    r = torch.empty_like(x)
    if N == 0:
        return q, r
    BLOCK = 256
    grid = (triton.cdiv(N, BLOCK),)
    _floor_divmod_kernel[grid](x, q, r, d, N, BLOCK=BLOCK)
    return q, r






# -----------------------------------------------------------------------------
# Mixed-radix coordinate (de)serialization.
#
# A D-dim integer coordinate ``c`` lying in ``[0, shape[0]) x ... x [0, shape[D-1])``
# is encoded as a single int32 ``key = sum(c[i] * strides[i])`` with row-major
# strides ``strides[i] = prod(shape[i+1:])``. Caller must ensure
# ``prod(shape) < 2**31`` so the result fits in int32.
#
# The encoding is bijective on the valid coord domain, so neighbor-cache callers
# can store a single int32 key per coord (instead of a D-vector) and rely on
# the hashmap's D_32 == 1 fast path. Out-of-range coords (e.g. produced by
# adding a delta) may alias other valid keys, so callers must bounds-check
# before/after lookup.
#
# The inline / kernel helpers always take ``shape_vec`` / ``shape_ptr`` and
# derive strides on-device via ``tl.cumprod``: this avoids materializing a
# second small tensor on the host and sidesteps the name collision with the
# conv-style ``stride`` parameter elsewhere in the stack.
# -----------------------------------------------------------------------------















