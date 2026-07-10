"""Tests for the RIFE input-padding multiple (src/interpolate/rife.py).

Frames are zero-padded up to a multiple of `_padMultiple()` before inference.
The coarsest IFBlock downsamples by `scale_list[0]` and then by another 4x in
its `conv0`, so the padded size must divide `4 * scale_list[0]`.

Regression coverage for two bugs:
  * The 4.25 family padded to a multiple of 128 at every resolution. It only
    needs 64 when scale == 1, so 1080p was padded to 1152 rows instead of 1088
    -- 64 wasted rows, ~6% of the frame, for no quality benefit.
  * The 128 was silently load-bearing for --dynamic_scale, which re-picks scale
    per frame down to 0.5 (dynamic_scale.py) and so needs 4 * 16 / 0.5 = 128.
    A naive 128 -> 64 change crashes there with a shape mismatch.
"""

import pytest

pytest.importorskip("torch")

from src.interpolate._padding import _RIFE_SCALE16, _padMultiple

SCALE16 = sorted(_RIFE_SCALE16)
SCALE8 = ["rife4.22", "rife4.22-lite", "rife4.6", "rife4.18", "rife_elexor"]


@pytest.mark.parametrize("method", SCALE16)
def testScale16FamilyPadsTo64AtScale1(method):
    """The measured win: 1080p pads to 1088, not 1152."""
    assert _padMultiple(method, scale=1.0, dynamicScale=False) == 64


@pytest.mark.parametrize("method", SCALE16)
def testScale16FamilyPadsTo128WhenDynamicScale(method):
    """dynamicScale can pick scale=0.5 per frame, which needs 4 * 16 / 0.5."""
    assert _padMultiple(method, scale=1.0, dynamicScale=True) == 128


@pytest.mark.parametrize("method", SCALE16)
def testScale16FamilyPadsTo128AtUhdScale(method):
    """RifeCuda drops to scale=0.5 above 1080p; scale_list[0] becomes 32."""
    assert _padMultiple(method, scale=0.5, dynamicScale=False) == 128
    assert _padMultiple(method, scale=0.5, dynamicScale=True) == 128


@pytest.mark.parametrize("method", SCALE8)
@pytest.mark.parametrize("scale", [1.0, 0.5])
@pytest.mark.parametrize("dynamicScale", [False, True])
def testScale8ArchesUnchangedAt64(method, scale, dynamicScale):
    """
    The 8/scale arches need only 32 at scale 1, but 64 is kept: dropping to 32
    measured no faster (1080p pads to 1088 under both) and cost rife4.22
    -0.038 dB on ATD-12K. At scale 0.5 they genuinely need 64.
    """
    assert _padMultiple(method, scale, dynamicScale) == 64


@pytest.mark.parametrize(
    "method,scale,dynamicScale",
    [(m, s, d) for m in SCALE16 + SCALE8 for s in (1.0, 0.5) for d in (False, True)],
)
def testPaddedSizeIsDivisibleByCoarsestStride(method, scale, dynamicScale):
    """
    The invariant the multiple exists to satisfy: for every scale the arch may
    actually run at, the padded frame survives `scale_list[0]` downsampling
    followed by conv0's 4x.
    """
    mul = _padMultiple(method, scale, dynamicScale)
    base = 16 if method in _RIFE_SCALE16 else 8
    effective = min(scale, 0.5) if dynamicScale else scale
    required = 4 * base / effective

    for dim in (720, 1080, 1440, 2160):
        padded = ((dim - 1) // mul + 1) * mul
        assert padded % required == 0, (
            f"{method} scale={effective}: padded {dim}->{padded} "
            f"not divisible by {required}"
        )
        assert padded >= dim
        assert padded - dim < mul
