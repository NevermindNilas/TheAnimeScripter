import itertools
from numbers import Number
from typing import Literal

import torch
from torch import Tensor


def _broadcast_dim_arg(x, D: int, name: str):
    """Broadcast a per-spatial-dim argument.

    ``None`` passes through. A scalar ``int`` is repeated ``D`` times. A
    sequence is validated to have exactly length ``D`` and returned as a
    tuple. Used by the fixed-spatial-dim op aliases (``*2d`` / ``*3d`` /
    ``*4d``) to let callers pass scalars for ``kernel_size`` / ``stride`` /
    ``padding`` / ``dilation`` / ``offset`` / ``scale_factor`` etc.
    """
    if x is None:
        return None
    if isinstance(x, int):
        return (x,) * D
    t = tuple(x)
    assert len(t) == D, (
        f"{name} must be a scalar int or a length-{D} sequence; got {x!r}"
    )
    return t


def split_sparse_shape(
    shape: torch.Size | None, sparse_dim: int,
) -> torch.Size | None:
    """Slice the leading ``sparse_dim`` entries of a channel-last full shape.

    FlexGEMM's library-wide convention is sparse-first, dense-last
    (mirroring :func:`torch.sparse_coo_tensor`): the op-layer ``shape`` is
    ``(*sparse_shape, *dense_shape)`` with ``len(sparse_shape) == coords.shape[1]``.
    The neighbor cache only ever stores the sparse prefix, so ops call this
    helper before delegating to :func:`build_neighbor_cache`.

    ``None`` passes through unchanged (the cache treats a missing shape as
    "derive lazily").
    """
    if shape is None:
        return None
    assert sparse_dim <= len(shape), (
        f"split_sparse_shape: sparse_dim={sparse_dim} exceeds "
        f"len(shape)={len(shape)}"
    )
    return torch.Size(tuple(shape[:sparse_dim]))






def pad_to_size_along_dim(x: Tensor, dim: int | tuple[int, ...], size: int | tuple[int, ...], value: Number = 0.) -> Tensor:
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
            tuple(itertools.chain.from_iterable((0, p) for p in reversed(pad_size))), 
            value=value
        )
    return x








