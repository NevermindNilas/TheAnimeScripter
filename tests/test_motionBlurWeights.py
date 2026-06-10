"""Tests for generateWeights in src/motionBlur.py — frame-blend weighting schemes.

generateWeights builds the per-sample weights used to blend interpolated frames
into one motion-blurred frame. Every scheme must return a normalised distribution
(sums to 1) so the blend preserves exposure; these pin shape, symmetry and the
normalisation invariant per scheme.
"""

import pytest

# motionBlur pulls in torch/cv2/nelux at import time; skip cleanly where the
# heavy runtime stack isn't installed (e.g. a torch-free CI job). nelux can be
# installed yet still raise ImportError (FFmpeg DLLs are only put on the search
# path at runtime by argumentsChecker), hence exc_type=ImportError.
pytest.importorskip("torch")
pytest.importorskip("cv2")
pytest.importorskip("nelux", exc_type=ImportError)

from src.motionBlur import generateWeights

ALL_SCHEMES = ["equal", "gaussian_sym", "pyramid", "ascending", "descending"]


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #

def testZeroSamplesEmpty():
    assert generateWeights(0) == []


def testNegativeSamplesEmpty():
    assert generateWeights(-5) == []


def testSingleSampleIsUnit():
    assert generateWeights(1) == [1.0]


# --------------------------------------------------------------------------- #
# Normalisation invariant: exposure must be preserved for every scheme
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("scheme", ALL_SCHEMES)
def testWeightsSumToOne(scheme):
    assert sum(generateWeights(8, scheme)) == pytest.approx(1.0)


@pytest.mark.parametrize("scheme", ALL_SCHEMES)
def testWeightCountMatchesSamples(scheme):
    assert len(generateWeights(7, scheme)) == 7


@pytest.mark.parametrize("scheme", ALL_SCHEMES)
def testWeightsNonNegative(scheme):
    assert all(w >= 0 for w in generateWeights(8, scheme))


# --------------------------------------------------------------------------- #
# Per-scheme shape
# --------------------------------------------------------------------------- #

def testEqualIsUniform():
    w = generateWeights(4, "equal")
    assert w == pytest.approx([0.25] * 4)


def testGaussianIsSymmetricAndPeaksCenter():
    w = generateWeights(5, "gaussian_sym")
    assert w == pytest.approx(w[::-1])           # symmetric
    assert w[2] == max(w)                          # peak at center
    assert w[0] == min(w) and w[-1] == min(w)      # tails lowest


def testAscendingMonotonicIncreasing():
    w = generateWeights(4, "ascending")
    assert all(a < b for a, b in zip(w, w[1:]))
    assert w == pytest.approx([0.1, 0.2, 0.3, 0.4])


def testDescendingMonotonicDecreasing():
    w = generateWeights(4, "descending")
    assert all(a > b for a, b in zip(w, w[1:]))
    assert w == pytest.approx([0.4, 0.3, 0.2, 0.1])


def testAscendingDescendingAreMirrors():
    assert generateWeights(6, "ascending") == pytest.approx(generateWeights(6, "descending")[::-1])


def testUnknownSchemeFallsBackToEqual():
    assert generateWeights(5, "nonexistent") == pytest.approx(generateWeights(5, "equal"))
