"""The --inpoint/--outpoint frame arithmetic.

The trim range used to be computed twice: `BuildBuffer` rounded each endpoint
independently while `getVideoMetadata` floored the product. For any non-integer
`(outpoint - inpoint) * fps` -- nearly every trim on 23.976/29.97 material --
the decoder emitted one more frame than the metadata claimed, and the drivers
that use that number as a hard `range()` bound stopped a frame early. Separately,
the end test was gated on the derived frame index rather than on whether a trim
had been requested, so a sub-frame `--outpoint` collapsed to "no limit".

`src.io.getVideoMetadata` imports torch and nelux inside `getVideoMetadata()`,
not at module scope, so this file needs neither.
"""

import argparse

import pytest

from src.io.getVideoMetadata import (
    isFramePoint,
    isTrimUnset,
    parseTrimArg,
    trimFrameRange,
    trimPointToSeconds,
)

NTSC = 24000 / 1001  # 23.976...
PAL = 25.0
NTSC30 = 30000 / 1001


@pytest.mark.parametrize("fps", [NTSC, PAL, NTSC30, 1199 / 50])
@pytest.mark.parametrize(
    "inPoint,outPoint", [(0, 3), (0, 2), (1, 2), (0.5, 2.5), (0.1, 1.9), (2, 7.3)]
)
def testRangeMatchesRoundedEndpoints(fps, inPoint, outPoint):
    # The decoder's own arithmetic, which is what actually gets emitted.
    start, end = trimFrameRange(fps, inPoint, outPoint)

    assert start == round(inPoint * fps)
    assert end == round(outPoint * fps)


def testNoOutpointMeansDecodeToEof():
    # None, not 0: `endFrame > 0` was the test that made a sub-frame outpoint
    # read as "no limit", so the sentinel has to be distinguishable from a
    # legitimately small frame index.
    assert trimFrameRange(NTSC, 0, 0) == (0, None)
    assert trimFrameRange(NTSC, 1.5, 0) == (round(1.5 * NTSC), None)


@pytest.mark.parametrize("fps", [NTSC, 1.0, 8.0])
def testSubFrameOutpointYieldsExactlyOneFrame(fps):
    # 0.02s at 23.976 fps, and 0.5s on a 1 fps timelapse, both round to frame 0.
    start, end = trimFrameRange(fps, 0, 0.4 / fps)

    assert (start, end) == (0, 1)


@pytest.mark.parametrize("inPoint", [1.0, 5.0, 0.25])
def testSubFrameSpanAtNonZeroInpointYieldsOneFrame(inPoint):
    # The worse half, and the case flooring against 0 rather than against the
    # start of the range would leave broken: end == start emitted no frames at
    # all while the audio was still cut to the requested length.
    start, end = trimFrameRange(NTSC, inPoint, inPoint + 0.001)

    assert end == start + 1


def testFloorNeverFiresWithoutATrim():
    # The obvious way to break the fix: applying the one-frame floor to a run
    # that asked for no --outpoint would truncate every untrimmed render to a
    # single frame.
    for inPoint in (0, 2.5):
        assert trimFrameRange(NTSC, inPoint, 0)[1] is None


def testFramePointsAreExact():
    # "<n>f" bypasses fps rounding entirely, so VFR-adjacent rates agree.
    for fps in (NTSC, PAL, NTSC30):
        assert trimFrameRange(fps, "100f", "500f") == (100, 500)
        assert trimFrameRange(fps, 0, "500f") == (0, 500)
        assert trimFrameRange(fps, "100f", 0)[1] is None


def testMixedUnitsConvertThroughFps():
    # 1s at NTSC is frame 24; ending at frame 100 keeps [24, 100).
    assert trimFrameRange(NTSC, 1.0, "100f") == (round(1.0 * NTSC), 100)


def testFrameOutpointZeroMeansEof():
    assert trimFrameRange(NTSC, "100f", "0f")[1] is None
    assert trimFrameRange(NTSC, "100f", 0)[1] is None


@pytest.mark.parametrize(
    "inPoint,outPoint", [("500f", "100f"), (10, 5), (10.0, "100f")]
)
def testInvalidRangeRaises(inPoint, outPoint):
    # 10s at NTSC is frame 240, so "100f" ends before it starts.
    with pytest.raises(ValueError, match="outpoint must be greater"):
        trimFrameRange(NTSC, inPoint, outPoint)


def testParseTrimArg():
    assert parseTrimArg(None) == 0
    assert parseTrimArg("") == 0
    assert parseTrimArg(0) == 0
    assert parseTrimArg("0f") == 0
    assert parseTrimArg(60) == 60.0
    assert parseTrimArg("60") == 60.0
    assert parseTrimArg("100f") == "100f"
    assert parseTrimArg(" 100F ") == "100f"
    assert isTrimUnset(0) and isTrimUnset("0f") and not isTrimUnset("100f")
    assert isFramePoint("100f") and not isFramePoint(60.0)
    assert trimPointToSeconds(NTSC, "24f") == pytest.approx(24 / NTSC)
    with pytest.raises(argparse.ArgumentTypeError):
        parseTrimArg("-5")
    with pytest.raises(argparse.ArgumentTypeError):
        parseTrimArg("10.5f")
    with pytest.raises(argparse.ArgumentTypeError):
        parseTrimArg("abc")
