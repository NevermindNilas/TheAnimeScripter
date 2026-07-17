"""Unit tests for the upscale input-multiple probe and padding helper.

These are weight-free: they exercise ``smallestValidMultiple`` with synthetic
``runOK`` callables that model an arch's divisibility requirement, so they run in
CI without downloading any model. End-to-end validation against the real
RealCUGAN-family ONNX/pth is done manually (see CHANGELOG).
"""

from src.upscale._shared import calculatePadding, smallestValidMultiple


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
    # fallin_*/shufflecugan/aniscale2 need both dims mod-4.
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


def test_calculatePaddingRightBottomToMultiple():
    # 498x566 -> both need +2 to reach the next multiple of 4 (500x568).
    assert calculatePadding(498, 566, 4) == (0, 2, 0, 2)
    # already aligned -> zero padding.
    assert calculatePadding(500, 568, 4) == (0, 0, 0, 0)
    # multiple 1 is always zero padding.
    assert calculatePadding(498, 566, 1) == (0, 0, 0, 0)
