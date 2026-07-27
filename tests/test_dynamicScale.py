"""Tests for --dynamic_scale (src/rifearches/dynamic_scale.py, rife_fast.py).

Regression coverage for four bugs:
  * `rife_fast._FastIFNet` stored `self.dynamicScale` and never read it. Its
    `scale_list` was frozen at construction, so on the fp16 CUDA path -- the
    default, and the only path `--dynamic_scale` is documented to support --
    the flag did nothing at all, while `RifeCuda` still disabled CUDA-graph
    capture on its behalf. Every non-fast arch honoured the flag.
  * `dynamic_scale.dynamicScale` cached one global SSIM module and bound it to
    the device and dtype of the *first* caller. Its gaussian window is a buffer,
    so a later caller with a different dtype hit
    "expected scalar type Half but found Float". The scorer is now
    `frame_analytics.ssim`, which builds its window per call and so has no
    dtype-bound state to get stuck on -- the mixed-dtype test still pins it.
  * The scale mapping was inverted. `scale_list` is `[base / scale, ...]` and is
    consumed as `F.interpolate(scale_factor=1/scale_list[i])`, so a SMALL scale
    is the COARSE pyramid -- Practical-RIFE's 4K / large-displacement setting.
    The old formula handed that to near-duplicate pairs and the finest pyramid
    to the pairs whose displacement did not fit the receptive field.
  * The pick was not restricted to powers of two, so 1.5 gave fractional block
    resolutions (16/1.5 = 10.67).
"""

import pytest

torch = pytest.importorskip("torch")

from src.rifearches import rife_fast
from src.rifearches.dynamic_scale import DYNAMIC_SCALES, dynamicScale, pickScale
from src.rifearches.IFNet_rife425 import IFNet as ReferenceIFNet

# 128 is the smallest size that survives dynamicScale's coarsest pick (0.5):
# block0 then runs at scale_list[0] = 32 and its conv0 downsamples 4x more.
# It is also <= _ANALYSIS_MAX_SIDE, so these pairs are scored as-is.
SIZE = 128
BASE = [16.0, 8.0, 4.0, 2.0, 1.0]


def _pair(kind, dtype=torch.float32):
    torch.manual_seed(0)
    a = torch.rand(1, 3, SIZE, SIZE, dtype=dtype)
    if kind == "hold":  # near-duplicate -> SSIM ~ 1 -> finest scale
        b = (a + torch.randn_like(a) * 0.001).clamp(0, 1)
    else:  # uncorrelated -> SSIM ~ 0 -> coarsest scale
        b = torch.rand(1, 3, SIZE, SIZE, dtype=dtype)
    return a, b


def _forward(model, a, b):
    ts = torch.full((1, 1, SIZE, SIZE), 0.5, dtype=a.dtype)
    with torch.inference_mode():
        model.f0 = model.f1 = None
        model(a, b, ts)


# --- dynamicScale itself ------------------------------------------------------


def testLargeDisplacementPicksCoarsestAndAHoldPicksFinest():
    """A small scale means MORE downsampling. An unrelated pair (large
    displacement) needs the coarse pyramid; a near-duplicate needs the fine one."""
    assert dynamicScale(*_pair("motion")) == 0.5
    assert dynamicScale(*_pair("hold")) == 2.0


@pytest.mark.parametrize("kind", ["hold", "motion"])
def testPickIsAlwaysOneOfTheDiscreteScales(kind):
    assert dynamicScale(*_pair(kind)) in DYNAMIC_SCALES


def testEveryDiscreteScaleIsAPowerOfTwo():
    """Non-powers of two (the old 1.5) give fractional block resolutions."""
    for s in DYNAMIC_SCALES:
        assert float(s).is_integer() or s == 0.5
        log = torch.log2(torch.tensor(float(s))).item()
        assert abs(log - round(log)) < 1e-9


def testAnalysisDownscaleDoesNotMoveThePick():
    """Scoring is done on a <=256px copy; that must not change the decision."""
    for kind in ("hold", "motion"):
        a, b = _pair(kind)
        big = (
            torch.nn.functional.interpolate(a, scale_factor=4, mode="bilinear"),
            torch.nn.functional.interpolate(b, scale_factor=4, mode="bilinear"),
        )
        assert dynamicScale(*big) == dynamicScale(a, b)


def testMixedDtypesInOneProcess():
    """The global-SSIM bug: fp32 first, then fp16, used to raise."""
    a32, b32 = _pair("hold")
    assert dynamicScale(a32, b32) == 2.0
    assert dynamicScale(a32.half(), b32.half()) == 2.0
    assert dynamicScale(a32, b32) == 2.0  # and back again


def testFp16AgreesWithFp32():
    """The scorer accumulates in fp32/fp64 whatever the input dtype, so a
    half-precision pair must not round its way to a different scale pick."""
    for kind in ("hold", "motion"):
        a, b = _pair(kind)
        assert dynamicScale(a.half(), b.half()) == dynamicScale(a, b)


def testMismatchedInputsRaise():
    a, b = _pair("hold")
    with pytest.raises(ValueError):
        dynamicScale(a, b[:, :, :64])


# --- pickScale: the driver may pre-empt the arch ------------------------------


class _Bare:
    pass


def testPickScaleUsesTheDriverSuppliedScale():
    """RifeCuda/RifeMPS score the UNPADDED pair once per pair and park the
    result; the arch must take it verbatim and not re-score the padded buffers."""
    model = _Bare()
    model.dsScale = 0.5
    a, b = _pair("hold")  # would score 2.0 on its own
    assert pickScale(model, a, b) == 0.5


def testPickScaleFallsBackWhenNoDriverSetIt():
    assert pickScale(_Bare(), *_pair("hold")) == 2.0


# --- the fast arch honours the flag -------------------------------------------


def testFastArchLeavesScaleListAloneWhenFlagIsOff():
    model = rife_fast.IFNet425(False, False, 1.0, 2).eval()
    _forward(model, *_pair("hold"))
    assert model.scale_list == BASE


@pytest.mark.parametrize("kind,expected", [("hold", 2.0), ("motion", 0.5)])
def testFastArchRebuildsScaleListPerFrame(kind, expected):
    model = rife_fast.IFNet425(False, True, 1.0, 2).eval()
    assert model.scale_list == BASE
    _forward(model, *_pair(kind))
    assert model.scale_list == [b / expected for b in BASE]


def testFastArchHonoursADriverSuppliedScale():
    model = rife_fast.IFNet425(False, True, 1.0, 2).eval()
    model.dsScale = 0.5
    _forward(model, *_pair("hold"))  # self-scoring would give 2.0
    assert model.scale_list == [b / 0.5 for b in BASE]


def testFastArchAgreesWithReferenceArchOnTheChosenScale():
    """The fast path is a drop-in for the reference arch, flag included."""
    for kind in ("hold", "motion"):
        fast = rife_fast.IFNet425(False, True, 1.0, 2).eval()
        ref = ReferenceIFNet(False, True, 1.0, 2).eval()
        a, b = _pair(kind)
        _forward(fast, a, b)
        _forward(ref, a, b)
        assert fast.scale_list == ref.scale_list


def testConstructorScaleIsOverriddenNotCompounded():
    """dynamicScale replaces the ctor scale (matching the reference arches);
    it does not divide by it twice."""
    model = rife_fast.IFNet425(False, True, 0.5, 2).eval()
    assert model.scale_list == [b / 0.5 for b in BASE]
    _forward(model, *_pair("hold"))
    assert model.scale_list == [b / 2.0 for b in BASE]
