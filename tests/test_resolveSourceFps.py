"""Tests for src.io.getVideoMetadata.resolveSourceFps — VFR fps selection.

A VFR source used to get r_frame_rate (the highest instantaneous rate) as its
fps while the decoder indexed frames with the average rate, so the output
played ~2.3x too fast against full-length audio.
"""

from src.io.getVideoMetadata import resolveSourceFps


def vfrProps():
    # Values from the reproduction sample: real720.mp4 with 3 of every 7
    # frames kept at original PTS.
    return {
        "r_frame_rate_num": 1199,
        "r_frame_rate_den": 50,
        "avg_frame_rate_num": 104313,
        "avg_frame_rate_den": 9950,
        "is_vfr": True,
        "fps": 10.4837,
    }


def testVfrPrefersAverageRate():
    fps, warning = resolveSourceFps(vfrProps())
    assert abs(fps - 104313 / 9950) < 1e-9
    assert warning is not None and "average rate" in warning


def testCfrKeepsExactRFrameRate():
    props = {
        "r_frame_rate_num": 24000,
        "r_frame_rate_den": 1001,
        "avg_frame_rate_num": 24000,
        "avg_frame_rate_den": 1001,
        "is_vfr": False,
        "fps": 23.976,
    }
    fps, warning = resolveSourceFps(props)
    assert fps == 24000 / 1001  # exact ratio, not the rounded fps key
    assert warning is None


def testVfrFlagWithoutDivergenceIsQuiet():
    # Container says VFR but the rates agree: no scary console warning.
    props = {
        "r_frame_rate_num": 24,
        "r_frame_rate_den": 1,
        "avg_frame_rate_num": 24,
        "avg_frame_rate_den": 1,
        "is_vfr": True,
    }
    fps, warning = resolveSourceFps(props)
    assert fps == 24.0
    assert warning is None


def testDivergenceWithoutVfrFlagStillPrefersAverage():
    # Some containers underreport is_vfr; a >1% r-vs-avg gap is trusted.
    props = {
        "r_frame_rate_num": 48,
        "r_frame_rate_den": 1,
        "avg_frame_rate_num": 24,
        "avg_frame_rate_den": 1,
        "is_vfr": False,
    }
    fps, warning = resolveSourceFps(props)
    assert fps == 24.0
    assert warning is not None


def testMissingRatiosFallBackToFpsKey():
    fps, warning = resolveSourceFps({"fps": 30.0})
    assert fps == 30.0
    assert warning is None


def testZeroDenominatorIsSafe():
    fps, _ = resolveSourceFps(
        {"r_frame_rate_num": 24, "r_frame_rate_den": 0, "fps": 25.0}
    )
    assert fps == 25.0
