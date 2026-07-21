"""Unit tests for the upscale input-multiple probe and padding helper.

These are weight-free: they exercise ``smallestValidMultiple`` with synthetic
``runOK`` callables that model an arch's divisibility requirement, so they run in
CI without downloading any model. End-to-end validation against the real
RealCUGAN-family ONNX/pth is done manually (see CHANGELOG).
"""

import pytest

# src.upscale._shared imports torch at module load (matmul precision + CUDA
# checker), so the whole module is torch-gated even though these helpers are
# pure integer math. Skip on the torch-less CI runner per repo convention.
pytest.importorskip("torch")

from src.upscale._shared import (  # noqa: E402
    KNOWN_INPUT_MULTIPLES,
    calculatePadding,
    lookupRequiredMultiple,
    smallestValidMultiple,
)


def _archRequiring(multiple):
    """runOK for an arch that only runs when both dims are divisible by `multiple`."""

    def runOK(h, w):
        return h % multiple == 0 and w % multiple == 0

    return runOK


def test_fullyConvolutionalArchProbesToOne():
    # A model that accepts any size (compact/SPAN) must report multiple 1 so
    # callers add zero padding and keep bit-exact parity.
    assert smallestValidMultiple(_archRequiring(1)) == 1


def test_realcuganFamilyProbesToFour():
    # adore/fallin_*/shufflecugan need both dims mod-4 (aniscale2 is Compact/mod-1).
    assert smallestValidMultiple(_archRequiring(4)) == 4


def test_probeDetectsEachSupportedMultiple():
    for m in (1, 2, 4, 8, 16):
        assert smallestValidMultiple(_archRequiring(m)) == m


def test_probeSizesAreOddMultiplesAboveFloor():
    # The probe must test a size that is a multiple of the candidate but NOT of
    # twice it (an odd multiple), otherwise it cannot distinguish e.g. 4 from 8,
    # and it must clear the models' internal reflect-pad minimum.
    seen = {}

    def recordingRunOK(h, w):
        seen.setdefault(h, w)
        return h % 4 == 0 and w % 4 == 0

    smallestValidMultiple(recordingRunOK)
    # every probed side is square and clears the reflect-pad minimum (48)
    assert seen, "probe never ran"
    for side, width in seen.items():
        assert side >= 48
        assert width == side  # square probe


def test_lookupKnownArchReturnsTableValue():
    # RealCUGAN-family archs need mod-4; the compact/convolutional ones need 1.
    assert lookupRequiredMultiple("shufflecugan") == 4
    assert lookupRequiredMultiple("adore") == 4
    assert lookupRequiredMultiple("aniscale2") == 1
    assert lookupRequiredMultiple("span") == 1


def test_lookupStripsBackendSuffix():
    # All backend variants of a method must resolve to the same table entry.
    for suffix in ("", "-tensorrt", "-directml", "-openvino", "-ncnn", "-mps"):
        assert lookupRequiredMultiple(f"shufflecugan{suffix}") == 4
        assert lookupRequiredMultiple(f"span{suffix}") == 1


def test_lookupUnknownOrEmptyReturnsNone():
    # Unlisted methods, custom models, and falsy input must fall back to the
    # probe (signaled by None), never a wrong hardcoded multiple.
    assert lookupRequiredMultiple("some-custom-model") is None
    assert lookupRequiredMultiple("figsr") is None  # deliberately not in table
    assert lookupRequiredMultiple("") is None
    assert lookupRequiredMultiple(None) is None


def test_lookupMatchesProbeForKnownArchs():
    # The table value must equal what the probe would have found for that arch.
    for method, multiple in KNOWN_INPUT_MULTIPLES.items():
        assert lookupRequiredMultiple(method) == multiple
        assert smallestValidMultiple(_archRequiring(multiple)) == multiple


def test_calculatePaddingRightBottomToMultiple():
    # 498x566 -> both need +2 to reach the next multiple of 4 (500x568).
    assert calculatePadding(498, 566, 4) == (0, 2, 0, 2)
    # already aligned -> zero padding.
    assert calculatePadding(500, 568, 4) == (0, 0, 0, 0)
    # multiple 1 is always zero padding.
    assert calculatePadding(498, 566, 1) == (0, 0, 0, 0)
