"""Tests for --dynamic_scale (src/rifearches/dynamic_scale.py, rife_fast.py).

Regression coverage for two bugs:
  * `rife_fast._FastIFNet` stored `self.dynamicScale` and never read it. Its
    `scale_list` was frozen at construction, so on the fp16 CUDA path -- the
    default, and the only path `--dynamic_scale` is documented to support --
    the flag did nothing at all, while `RifeCuda` still disabled CUDA-graph
    capture on its behalf. Every non-fast arch honoured the flag.
  * `dynamic_scale.dynamicScale` cached one global SSIM module and bound it to
    the device and dtype of the *first* caller. Its gaussian window is a buffer,
    so a later caller with a different dtype hit
    "expected scalar type Half but found Float".
"""

import pytest

torch = pytest.importorskip("torch")

from src.rifearches import rife_fast
from src.rifearches.dynamic_scale import _SSIMFUNCTIONS, dynamicScale
from src.rifearches.IFNet_rife425 import IFNet as ReferenceIFNet

# 128 is the smallest size that survives dynamicScale's coarsest pick (0.5):
# block0 then runs at scale_list[0] = 32 and its conv0 downsamples 4x more.
SIZE = 128
BASE = [16.0, 8.0, 4.0, 2.0, 1.0]


def _pair(kind, dtype=torch.float32):
    torch.manual_seed(0)
    a = torch.rand(1, 3, SIZE, SIZE, dtype=dtype)
    if kind == "hold":  # near-duplicate -> SSIM ~ 1 -> coarsest scale
        b = (a + torch.randn_like(a) * 0.001).clamp(0, 1)
    else:  # uncorrelated -> SSIM ~ 0 -> finest scale
        b = torch.rand(1, 3, SIZE, SIZE, dtype=dtype)
    return a, b


def _forward(model, a, b):
    ts = torch.full((1, 1, SIZE, SIZE), 0.5, dtype=a.dtype)
    with torch.inference_mode():
        model.f0 = model.f1 = None
        model(a, b, ts)


# --- dynamicScale itself ------------------------------------------------------


def testHoldPicksCoarsestScaleAndMotionPicksFinest():
    """Counterintuitive but correct: HIGH ssim (a hold) -> 0.5, low ssim -> 2.0."""
    assert dynamicScale(*_pair("hold")) == 0.5
    assert dynamicScale(*_pair("motion")) == 2.0


def testMixedDtypesInOneProcess():
    """The global-SSIM bug: fp32 first, then fp16, used to raise."""
    a32, b32 = _pair("hold")
    assert dynamicScale(a32, b32) == 0.5
    assert dynamicScale(a32.half(), b32.half()) == 0.5
    assert dynamicScale(a32, b32) == 0.5  # and back again


def testSsimModuleIsCachedPerDeviceAndDtype():
    a, b = _pair("hold")
    _SSIMFUNCTIONS.clear()
    dynamicScale(a, b)
    assert len(_SSIMFUNCTIONS) == 1
    dynamicScale(a, b)
    assert len(_SSIMFUNCTIONS) == 1, "same (device, dtype) must reuse its module"
    dynamicScale(a.half(), b.half())
    assert len(_SSIMFUNCTIONS) == 2, "a new dtype needs its own module"


# --- the fast arch honours the flag -------------------------------------------


def testFastArchLeavesScaleListAloneWhenFlagIsOff():
    model = rife_fast.IFNet425(False, False, 1.0, 2).eval()
    _forward(model, *_pair("hold"))
    assert model.scale_list == BASE


@pytest.mark.parametrize("kind,expected", [("hold", 0.5), ("motion", 2.0)])
def testFastArchRebuildsScaleListPerFrame(kind, expected):
    model = rife_fast.IFNet425(False, True, 1.0, 2).eval()
    assert model.scale_list == BASE
    _forward(model, *_pair(kind))
    assert model.scale_list == [b / expected for b in BASE]


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
    _forward(model, *_pair("motion"))
    assert model.scale_list == [b / 2.0 for b in BASE]
