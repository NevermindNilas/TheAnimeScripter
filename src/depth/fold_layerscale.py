"""Fold LayerScale gamma into preceding Linear weights/biases.

Block does: x = x + ls1(attn(norm1(x)));  x = x + ls2(mlp(norm2(x)))
ls(y) = y * gamma → fold gamma into attn.proj and mlp.fc2 / mlp.w3.
After fold ls1/ls2 = Identity. Math identical in fp32, near-identical in fp16.
"""

import torch
from torch import nn

from .dinov2_layers.layer_scale import LayerScale
from .dinov2_layers.mlp import Mlp
from .dinov2_layers.swiglu_ffn import SwiGLUFFN

try:
    from xformers.ops import SwiGLU as _XSwiGLU

    _SWIGLU_TYPES = (SwiGLUFFN, _XSwiGLU)
except ImportError:
    _SWIGLU_TYPES = (SwiGLUFFN,)


@torch.no_grad()
def _fold_into_linear(linear: nn.Linear, gamma: torch.Tensor) -> None:
    linear.weight.mul_(gamma.unsqueeze(1).to(linear.weight.dtype))
    if linear.bias is not None:
        linear.bias.mul_(gamma.to(linear.bias.dtype))


@torch.no_grad()
def fold_layerscale_(model: nn.Module) -> nn.Module:
    folded = 0
    for module in model.modules():
        if not (hasattr(module, "attn") and hasattr(module, "mlp")):
            continue

        ls1 = getattr(module, "ls1", None)
        if isinstance(ls1, LayerScale) and hasattr(module.attn, "proj"):
            _fold_into_linear(module.attn.proj, ls1.gamma.data)
            module.ls1 = nn.Identity()
            folded += 1

        ls2 = getattr(module, "ls2", None)
        if isinstance(ls2, LayerScale):
            mlp = module.mlp
            last = None
            if isinstance(mlp, _SWIGLU_TYPES):
                last = getattr(mlp, "w3", None)
            elif isinstance(mlp, Mlp):
                last = getattr(mlp, "fc2", None)
            if isinstance(last, nn.Linear):
                _fold_into_linear(last, ls2.gamma.data)
                module.ls2 = nn.Identity()
                folded += 1

    return model
