# Arches whose coarsest IFBlock runs at scale_list[0] = 16 / scale. Everything
# else in _RIFE_V1 starts at 8 / scale.
_RIFE_SCALE16 = frozenset({"rife4.25", "rife4.25-heavy", "rife4.25-lite"})


def _padMultiple(method, scale, dynamicScale):
    """
    Frames are zero-padded up to a multiple of this before inference.

    The coarsest IFBlock downsamples its input by ``scale_list[0]`` and then by
    another 4x inside ``conv0``, so the padded size must divide ``4 * 16 / scale``
    for the 4.25 family and ``4 * 8 / scale`` for the rest. ``dynamicScale``
    re-picks ``scale`` per frame and can go as low as 0.5
    (``dynamic_scale.py`` minScale), so it has to budget for the coarsest scale
    it may choose rather than the one passed in.

    Never returns less than 64: the 8/scale arches only need 32 at scale 1, but
    dropping them to 32 measured no faster (1080p pads to 1088 either way) and
    cost rife4.22 -0.038 dB on ATD-12K.
    """
    base = 16 if method in _RIFE_SCALE16 else 8
    if dynamicScale:
        scale = min(scale, 0.5)
    return max(64, int(4 * base / scale))
