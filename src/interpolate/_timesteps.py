def gapPlan(prevPos, curPos, factorNum, factorDen):
    """Frames to synthesize for the SOURCE interval ``(prevPos, curPos]``.

    Positions count source frames, so a --smooth_dedup gap that swallowed duplicate frames
    is simply a wider interval and gets proportionally more output slots. Every
    interpolation driver emits its intermediates for the interval *ending* at the
    frame it was just handed (it interpolates the cached ``I0`` against it), so
    this is the interval the caller must plan for.

    Returns ``(framesToInsert, timesteps)`` with timesteps in ``(0, 1)``,
    relative to that interval. For an integer factor ``f`` over a unit interval
    it yields exactly ``[1/f, 2/f, ..., (f-1)/f]`` -- the same values
    ``interpolateTimestep`` produces on its own.
    """
    outPrev = (prevPos * factorNum) // factorDen
    outCur = (curPos * factorNum) // factorDen
    # Keep the whole numerator in integers: dividing an absolute source position
    # first and subtracting after loses the exactness of ratios like 1/3 once
    # prevPos is large, and those values must match the driver's own ladder.
    base = prevPos * factorNum
    scale = factorNum * (curPos - prevPos)
    timesteps = [(o * factorDen - base) / scale for o in range(outPrev + 1, outCur)]
    return len(timesteps), timesteps


def cutHoldSplit(prevPos, curPos, factorNum, factorDen):
    """How many of a gap's synthesized slots precede its final source step.

    A gap widened by ``--smooth_dedup`` covers several source frames, and every
    one of them except the last was a duplicate of the frame at ``prevPos``. When
    a scene cut lands on ``curPos``, only that final source step is the cut; the
    slots before it still belong to the outgoing shot and must be filled from the
    previous frame, not the incoming one. Returns 0 for a one-frame-wide gap, so
    a cut without ``--smooth_dedup`` fills entirely from the incoming frame as it
    always has.
    """
    outPrev = (prevPos * factorNum) // factorDen
    lastStep = ((curPos - 1) * factorNum) // factorDen
    return max(0, lastStep - outPrev)


def interpolateTimestep(index, framesToInsert, timesteps=None):
    if timesteps is not None and index < len(timesteps):
        return timesteps[index]
    return (index + 1) * 1 / (framesToInsert + 1)


def fillTimestepBuffer(buffer, cachedValue, timestep):
    if cachedValue != timestep:
        buffer.fill_(timestep)
        return timestep
    return cachedValue
