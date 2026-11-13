from typing import *
import itertools

import torch
from torch import Tensor
from torch.autograd import Function
try:
    # PyTorch >= 2.4 unified API.
    from torch.amp import custom_fwd as _custom_fwd, custom_bwd as _custom_bwd
    def custom_fwd(fn):
        return _custom_fwd(device_type="cuda")(fn)
    def custom_bwd(fn):
        return _custom_bwd(device_type="cuda")(fn)
except ImportError:  # pragma: no cover - legacy fallback
    from torch.cuda.amp import custom_fwd, custom_bwd  # type: ignore[no-redef]
from ... import config
from ... import kernels
from ..neighbor_cache import NeighborCache


def _amp_cast(ctx, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
    """Resolve the compute dtype under ``torch.autocast`` and cast inputs.

    ``custom_fwd`` attaches ``ctx._fwd_used_autocast`` / ``ctx._dtype`` that
    record whether the *caller* was inside an autocast region and which dtype
    was active. Inside the decorated ``forward`` autocast is already disabled,
    so we must rely on those captured attributes (not
    ``torch.is_autocast_enabled()``).

    Returns ``(input, weight, bias)`` cast to a common compute dtype:
      * autocast active  -> ``ctx._dtype`` (fp16 or bf16, set by the user)
      * autocast off     -> ``input.dtype`` (no-op, just enforces weight/bias)

    ``bias`` is **intentionally not cast** — the Triton fwd kernels add bias
    to the fp32 accumulator in the GEMM epilogue, so keeping bias in its
    original (typically fp32) dtype preserves precision. Passing fp32 bias
    when input/weight are fp16 only costs a negligible HBM read per output
    tile and is the same epilogue contract used by cuBLAS Lt and PyTorch's
    ``nn.Linear`` under autocast.

    Crucially, ``requires_grad`` from the *original* forward arguments is
    propagated onto the cast tensors. Inside ``Function.forward`` autograd is
    disabled, so ``.to(other_dtype)`` returns a leaf-like tensor with
    ``requires_grad=False``; without re-attaching the flag, the existing
    ``if X.requires_grad:`` branches in ``backward`` (which inspect saved
    tensors) would all become False under AMP, silently producing ``None``
    grads.
    """
    if getattr(ctx, "_fwd_used_autocast", False):
        compute_dtype = ctx._dtype
    else:
        compute_dtype = input.dtype

    inp_rg = input.requires_grad
    w_rg = weight.requires_grad
    b_rg = bias is not None and bias.requires_grad

    input = input.to(compute_dtype)
    weight = weight.to(compute_dtype)
    # bias intentionally kept at its original dtype (see docstring).

    if inp_rg:
        input.requires_grad_(True)
    if w_rg:
        weight.requires_grad_(True)
    if bias is not None and b_rg:
        bias.requires_grad_(True)

    return input, weight, bias





class SparseConvImplicitGemmFunction(Function):
    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        input: Tensor,
        neighbor_cache: NeighborCache,
        weight: Tensor,
        bias: Optional[Tensor] = None,
        allow_tf32: Optional[bool] = None,
    ) -> Tuple[Tensor, NeighborCache]:
        input, weight, bias = _amp_cast(ctx, input, weight, bias)
        input = input.contiguous()
        assert input.is_contiguous(), "Input features should be contiguous"
        Co, V, Ci = weight.shape
        assert input.shape[-1] == Ci, f"Input channels ({input.shape[-1]}) should match weight channels ({Ci})"

        output  = kernels.triton.sparse_conv_fwd_implicit_gemm(
            input,
            weight,
            bias,
            neighbor_cache.fwd_map,
            allow_tf32=allow_tf32,
        )

        ctx.save_for_backward(input, weight, bias)
        ctx.neighbor_cache = neighbor_cache
        ctx.allow_tf32 = allow_tf32
        return output, neighbor_cache






class SparseConvMaskedImplicitGemmFunction(Function):
    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        input: torch.Tensor,
        neighbor_cache: NeighborCache,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        allow_tf32: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, NeighborCache]:
        input, weight, bias = _amp_cast(ctx, input, weight, bias)
        input = input.contiguous()
        assert input.is_contiguous(), "Input features should be contiguous"
        Co, V, Ci = weight.shape
        assert input.shape[-1] == Ci, f"Input channels ({input.shape[-1]}) should match weight channels ({Ci})"

        output = kernels.triton.sparse_conv_fwd_masked_implicit_gemm(
            input,
            weight,
            bias,
            neighbor_cache.fwd_map,
            neighbor_cache.fwd_sorted_idx,
            neighbor_cache.fwd_valid_kernel_callback,
            neighbor_cache.fwd_valid_kernel_seg_callback,
            allow_tf32=allow_tf32,
        )

        ctx.save_for_backward(input, weight, bias)
        ctx.neighbor_cache = neighbor_cache
        ctx.allow_tf32 = allow_tf32
        return output, neighbor_cache






def _select_function(algorithm: Literal["explicit_gemm", "implicit_gemm", "implicit_gemm_splitk", "masked_implicit_gemm", "masked_implicit_gemm_splitk"] | None = None) -> Type[Function]:
    if algorithm is None:
        raise ValueError("pass algorithm= explicitly (no default in the vendored FlexGEMM)")
        
    if algorithm == "implicit_gemm":
        return SparseConvImplicitGemmFunction
    if algorithm == "masked_implicit_gemm":
        return SparseConvMaskedImplicitGemmFunction
    raise ValueError(f"Invalid algorithm {algorithm}")

