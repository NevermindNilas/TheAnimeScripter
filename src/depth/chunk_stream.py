"""Bounded depth-only adaptation of DA3-Streaming's overlapping chunks.

Uses the existing Apache-2.0 DA3 Small/Base runtime. Chunk alignment is a
robust depth-scale fit on repeated views, not the upstream 3D Sim(3)/SALAD
pipeline. Only predictions of the SAME input frame are blended.
"""

from itertools import islice

import numpy as np


def alignDepthOverlap(previous, current):
    """Align a new chunk to its predecessor and blend their repeated views."""
    overlap = len(previous)
    repeated = current[:overlap]
    if repeated.shape != previous.shape:
        raise ValueError("Depth overlap dimensions changed between chunks")

    # Deterministic spatial sampling bounds the temporary ratio array. Median
    # ratios tolerate moving-object errors and outliers in either prediction.
    stride = max(1, (previous.size + 99999) // 100000)
    old = previous.reshape(-1)[::stride].astype(np.float64)
    new = repeated.reshape(-1)[::stride].astype(np.float64)
    valid = np.isfinite(old) & np.isfinite(new) & (old > 0) & (new > 0)
    if valid.sum() < 32:
        # No reliable shared scale: use the new prediction without blending
        # two unrelated coordinate systems.
        return current
    scale = np.median(old[valid] / new[valid])
    if not np.isfinite(scale) or scale <= 0:
        return current
    current *= scale
    if not np.isfinite(current).all():
        raise ValueError("Non-finite depth after chunk scale alignment")

    weight = np.arange(1, overlap + 1, dtype=np.float32)[:, None, None]
    weight /= overlap + 1
    oldValid = previous > 0
    newValid = repeated > 0
    current[:overlap] = np.where(
        oldValid & newValid,
        previous * (1 - weight) + repeated * weight,
        np.where(newValid, repeated, previous),
    )
    return current


def streamDepthChunks(frames, inferChunk, chunkSize=32, overlap=None):
    """Yield exactly one depth map per frame, with at most one RGB chunk held.

    ``inferChunk`` receives ordered views of one scene and returns [N,H,W].
    A half-window overlap is retained until the next prediction, including at
    EOF. No extra inference or duplicate outputs are needed for an exact fit.
    """
    overlap = chunkSize // 2 if overlap is None else overlap
    if chunkSize < 2 or not 0 < overlap <= chunkSize // 2:
        raise ValueError("Require chunkSize >= 2 and 0 < overlap <= chunkSize // 2")
    frames = iter(frames)
    window = list(islice(frames, chunkSize))
    previous = None
    while window:
        depths = np.array(inferChunk(window), dtype=np.float32, copy=True)
        if depths.ndim != 3 or len(depths) != len(window):
            raise ValueError(
                "Chunk inference must return one [H,W] depth map per frame"
            )
        depths[~np.isfinite(depths) | (depths <= 0)] = 0
        if previous is not None:
            depths = alignDepthOverlap(previous, depths)

        if len(window) < chunkSize:
            yield from depths
            return

        yield from depths[:-overlap]
        previous = depths[-overlap:].copy()
        del depths
        window = window[-overlap:]
        newFrames = list(islice(frames, chunkSize - overlap))
        if not newFrames:
            yield from previous
            return
        window.extend(newFrames)
        del newFrames
