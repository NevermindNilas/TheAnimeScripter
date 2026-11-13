"""Bounded depth-only adaptation of DA3-Streaming's overlapping chunks.

Uses the existing Apache-2.0 DA3 Small/Base runtime. Chunk alignment is a
robust depth-scale fit on repeated views, not the upstream 3D Sim(3)/SALAD
pipeline. Only predictions of the SAME input frame are blended.

Runs in torch on the device the chunk's depth is on, so on a GPU nothing comes
back to the host: the alignment's decisions (enough shared pixels, a usable
scale) are taken on the device with ``torch.where``, and the one check that
must stop the run (a non-finite map after scaling) is read back only when the
caller writes the chunk (``DepthBlock.verify``).
"""

from itertools import islice
from typing import NamedTuple

import torch

# Ratio pixels sampled per overlap: deterministic strided sampling bounds the
# temporary arrays whatever the resolution.
_ALIGN_SAMPLES = 100000
# Elements per float64 slice of the scale multiply.
_SCALE_ELEMENTS = 1 << 20

_NON_FINITE = "Non-finite depth after chunk scale alignment"


def _nanMedian(values):
    """``np.median`` of the non-NaN entries of a 1-D tensor, NaN when none.

    The mean of the two middle values for an even count (``torch.median`` and
    ``nanmedian`` return the lower one), found by sorting rather than by a
    boolean gather so the count never has to reach the host.
    """
    ordered = torch.sort(values).values  # NaN sorts last
    count = (~torch.isnan(values)).sum()
    last = values.numel() - 1
    # index_select, not ordered[tensor]: indexing with a 0-dim tensor can read
    # it back to the host.
    middle = torch.stack(
        [((count - 1).clamp_min(0) // 2).clamp_max(last), (count // 2).clamp_max(last)]
    )
    lower, upper = ordered.index_select(0, middle)
    return (lower + upper) / 2


def _scaleInPlace(depths, scale):
    """``depths *= scale`` the way numpy's float32 ``*=`` a float64 does it:
    the product in float64, rounded once to float32. In place, as numpy's was;
    sliced along the frame axis to bound the float64 temporaries."""
    perFrame = max(1, depths[0].numel())
    step = max(1, _SCALE_ELEMENTS // perFrame)
    for i in range(0, len(depths), step):
        depths[i : i + step] = (depths[i : i + step].double() * scale).float()
    return depths


def _alignOverlap(previous, current):
    """Align ``current`` to ``previous`` in place; returns (it, non-finite flag).

    The flag is a device bool, set when the scaled chunk is not finite; the
    caller raises on it before writing anything of this chunk.
    """
    overlap = len(previous)
    if tuple(current[:overlap].shape) != tuple(previous.shape):
        raise ValueError("Depth overlap dimensions changed between chunks")

    # Deterministic spatial sampling bounds the temporary ratio array. Median
    # ratios tolerate moving-object errors and outliers in either prediction.
    stride = max(1, (previous.numel() + _ALIGN_SAMPLES - 1) // _ALIGN_SAMPLES)
    old = previous.reshape(-1)[::stride].double()
    new = current[:overlap].reshape(-1)[::stride].double()
    valid = torch.isfinite(old) & torch.isfinite(new) & (old > 0) & (new > 0)
    scale = _nanMedian(torch.where(valid, old / new, torch.nan))
    # Fewer than 32 shared pixels or a useless scale: no reliable shared scale,
    # so the new prediction is used as is, without blending two unrelated
    # coordinate systems (a scale of exactly 1 leaves it bit-identical).
    usable = (valid.sum() >= 32) & torch.isfinite(scale) & (scale > 0)
    scale = torch.where(usable, scale, torch.ones_like(scale))
    aligned = _scaleInPlace(current, scale)
    nonFinite = usable & ~torch.isfinite(aligned).all()

    # The weights are divided on the host: a CUDA tensor divided by a Python
    # scalar is multiplied by its reciprocal, a last-bit difference.
    weight = (torch.arange(1, overlap + 1, dtype=torch.float32) / (overlap + 1)).to(
        aligned.device
    )
    # One repeated view at a time keeps the blend's temporaries to one map.
    for i in range(overlap):
        old, new, w = previous[i], aligned[i], weight[i]
        newValid = new > 0
        blended = torch.where(
            (old > 0) & newValid,
            old * (1 - w) + new * w,
            torch.where(newValid, new, old),
        )
        aligned[i] = torch.where(usable, blended, new)
    return aligned, nonFinite


def alignDepthOverlap(previous, current):
    """Align a new chunk to its predecessor and blend their repeated views.

    Like the numpy version, a float32 ``current`` is updated in place."""
    previous = torch.as_tensor(previous, dtype=torch.float32)
    current = torch.as_tensor(current, dtype=torch.float32, device=previous.device)
    aligned, nonFinite = _alignOverlap(previous, current)
    if bool(nonFinite):
        raise ValueError(_NON_FINITE)
    return aligned


class DepthBlock(NamedTuple):
    """Depth maps [N, H, W] ready to write, in frame order.

    ``nonFinite`` is the device flag of the chunk alignment that produced them
    (None when no alignment ran); ``verify`` raises on it. Reading it waits for
    that chunk's GPU work, so callers read it when they write the block anyway.
    """

    depths: torch.Tensor
    nonFinite: torch.Tensor | None = None

    def verify(self) -> None:
        verifyFinite(self.nonFinite)


def verifyFinite(nonFinite) -> None:
    """Raise on a chunk alignment's non-finite flag (``DepthBlock.nonFinite``).

    A check that holds the flag alone, not the block's chunk of depth maps."""
    if nonFinite is not None and bool(nonFinite):
        raise ValueError(_NON_FINITE)


def streamDepthBlocks(frames, inferChunk, chunkSize=32, overlap=None):
    """Yield every frame's depth map exactly once, as ``DepthBlock`` runs.

    ``inferChunk`` receives ordered views of one scene and returns [N,H,W]:
    a tensor on any device, which the stream then owns and edits in place
    (its own fresh output, never a view of a buffer the driver reuses), or
    anything else ``torch.tensor`` copies. A half-window overlap is retained
    until the next prediction, including at EOF. No extra inference or
    duplicate outputs are needed for an exact fit. At most one RGB chunk is
    held.
    """
    overlap = chunkSize // 2 if overlap is None else overlap
    if chunkSize < 2 or not 0 < overlap <= chunkSize // 2:
        raise ValueError("Require chunkSize >= 2 and 0 < overlap <= chunkSize // 2")
    frames = iter(frames)
    window = list(islice(frames, chunkSize))
    previous = None
    while window:
        depths = inferChunk(window)
        if isinstance(depths, torch.Tensor):
            depths = depths.float()
        else:
            depths = torch.tensor(depths, dtype=torch.float32)
        if depths.ndim != 3 or len(depths) != len(window):
            raise ValueError(
                "Chunk inference must return one [H,W] depth map per frame"
            )
        # Every non-finite or non-positive depth is 0 (a -0.0 may stay -0.0).
        depths.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0).clamp_min_(0.0)
        nonFinite = None
        if previous is not None:
            depths, nonFinite = _alignOverlap(previous, depths)

        if len(window) < chunkSize:
            yield DepthBlock(depths, nonFinite)
            return

        yield DepthBlock(depths[:-overlap], nonFinite)
        previous = depths[-overlap:].clone()
        del depths
        window = window[-overlap:]
        newFrames = list(islice(frames, chunkSize - overlap))
        if not newFrames:
            yield DepthBlock(previous, nonFinite)
            return
        window.extend(newFrames)
        del newFrames


def streamDepthChunks(frames, inferChunk, chunkSize=32, overlap=None):
    """``streamDepthBlocks`` one [H, W] map at a time, each chunk verified first."""
    for block in streamDepthBlocks(frames, inferChunk, chunkSize, overlap):
        block.verify()
        yield from block.depths
