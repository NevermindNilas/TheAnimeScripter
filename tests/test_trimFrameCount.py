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

import pytest

from src.io.getVideoMetadata import trimFrameRange

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
