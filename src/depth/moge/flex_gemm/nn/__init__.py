"""``flex_gemm.nn`` — :class:`torch.nn.Module` wrappers around the op layer.

Each module pairs a dim-generic base class with ``2d`` / ``3d`` / ``4d``
fixed-spatial-dim aliases that accept scalar dim args (broadcast to the
appropriate length). See the corresponding base-class docstring for details.
"""
from .pool import SparsePool, SparsePool3d
from .upsample import SparseUpsample, SparseUpsample3d


__all__ = [
    "SparsePool",
    "SparsePool3d",
    "SparseUpsample",
    "SparseUpsample3d",
]
