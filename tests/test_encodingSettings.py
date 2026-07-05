"""Tests for src.io.encodingSettings — FFmpeg encoder flags and pixel formats.

getPixFMT is the bug-prone bit: a chain of mutually-exclusive branches that pick
input/output pixel formats from (encode_method, bitDepth, grayscale, transparent)
and can silently rewrite the encoder (transparency -> prores_segment) or downgrade
bit depth (NVENC H.264 has no 10-bit path). These pin the resolved tuples.
"""

import pytest

from src.io.encodingSettings import getPixFMT, matchEncoder

# --------------------------------------------------------------------------- #
# matchEncoder: name -> ffmpeg flag list
# --------------------------------------------------------------------------- #


def testX264Flags():
    assert matchEncoder("x264") == [
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "15",
    ]


def testPngFlags():
    assert matchEncoder("png") == ["-c:v", "png", "-q:v", "0"]


def testProresAndSegmentShareFlags():
    expected = ["-c:v", "prores_ks", "-profile:v", "4", "-qscale:v", "15"]
    assert matchEncoder("prores") == expected
    assert matchEncoder("prores_segment") == expected


def testTenBitAddsHigh10Profile():
    assert matchEncoder("x264_10bit")[-2:] == ["-profile:v", "high10"]


def testUnknownEncoderReturnsEmpty():
    # No matching case arm -> empty command, caller must handle.
    assert matchEncoder("does_not_exist") == []


# --------------------------------------------------------------------------- #
# getPixFMT: (in, out, encode_method) resolution
# --------------------------------------------------------------------------- #


def testTransparencyOverridesToProresSegment():
    # Transparency wins over everything and rewrites the encoder.
    assert getPixFMT("x264", "8bit", False, True) == (
        "rgba",
        "yuva444p10le",
        "prores_segment",
    )


def testTransparencyDoesNotDoubleRewrite():
    # Already prores_segment -> encoder kept, default formats used (not the rgba override).
    inFmt, outFmt, enc = getPixFMT("prores_segment", "8bit", False, True)
    assert enc == "prores_segment"
    assert (inFmt, outFmt) == ("rgb24", "yuv420p")


@pytest.mark.parametrize(
    "bitDepth,expected",
    [("8bit", ("gray", "yuv420p")), ("16bit", ("gray16le", "yuv444p10le"))],
)
def testGrayscale(bitDepth, expected):
    inFmt, outFmt, _ = getPixFMT("x264", bitDepth, True, False)
    assert (inFmt, outFmt) == expected


def testTenBitMethodEightBitInput():
    assert getPixFMT("x264_10bit", "8bit", False, False) == (
        "rgb24",
        "yuv420p10le",
        "x264_10bit",
    )


def testNvencH264SixteenBitDowngradesOutputToEightBit():
    # NVENC H.264 has no 10-bit encode path; output is forced to 8-bit yuv420p.
    inFmt, outFmt, _ = getPixFMT("nvenc_h264", "16bit", False, False)
    assert (inFmt, outFmt) == ("rgb48le", "yuv420p")


@pytest.mark.parametrize(
    "bitDepth,expected",
    [("8bit", ("rgb24", "yuv420p")), ("16bit", ("rgb48le", "yuv444p10le"))],
)
def testDefaultBranch(bitDepth, expected):
    inFmt, outFmt, _ = getPixFMT("x264", bitDepth, False, False)
    assert (inFmt, outFmt) == expected


def testPngKeepsRgbInAndOut():
    assert getPixFMT("png", "8bit", False, False) == ("rgb24", "rgb24", "png")


def testProresPromotesEightBitOutputTo444p10():
    assert getPixFMT("prores", "8bit", False, False) == (
        "rgb24",
        "yuv444p10le",
        "prores",
    )
