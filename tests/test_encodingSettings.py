"""Tests for src.io.encodingSettings — FFmpeg encoder flags and pixel formats.

getPixFMT is the bug-prone bit: a chain of mutually-exclusive branches that pick
input/output pixel formats from (encode_method, bitDepth, grayscale, transparent)
and can silently rewrite the encoder (transparency -> prores_segment) or downgrade
bit depth (NVENC H.264 has no 10-bit path). These pin the resolved tuples.
"""

import pytest

from src.io.encodingSettings import (
    colorSpaceOptions,
    getPixFMT,
    matchEncoder,
    matchNeluxEncoder,
)

# --------------------------------------------------------------------------- #
# matchEncoder: name -> ffmpeg flag list
# --------------------------------------------------------------------------- #


def testSlowX265LetsPixelFormatSelectProfile():
    assert "-profile:v" not in matchEncoder("slow_x265")
    for bitDepth in ("8bit", "16bit"):
        assert (
            "profile" not in matchNeluxEncoder("slow_x265_nelux", bitDepth)["options"]
        )


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


def testJpegFlags():
    # -qmin 1 must precede -q:v 1: mjpeg's default qmin of 2 silently clamps
    # the requested quality otherwise.
    assert matchEncoder("jpeg") == ["-c:v", "mjpeg", "-qmin", "1", "-q:v", "1"]


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


def testJpegUsesFullRange444():
    assert getPixFMT("jpeg", "8bit", False, False) == ("rgb24", "yuvj444p", "jpeg")


def testJpegSixteenBitDowngradesOutputToEightBit():
    # JPEG has no 16-bit path; frames are accepted at rgb48le but encoded 8-bit.
    assert getPixFMT("jpeg", "16bit", False, False) == ("rgb48le", "yuvj444p", "jpeg")


def testProresPromotesEightBitOutputTo444p10():
    assert getPixFMT("prores", "8bit", False, False) == (
        "rgb24",
        "yuv444p10le",
        "prores",
    )


# --------------------------------------------------------------------------- #
# matchNeluxEncoder: name -> nelux.VideoEncoder kwargs
# --------------------------------------------------------------------------- #


def testUnknownNeluxEncoderReturnsNone():
    # None (not a fallback dict): NeluxWriteBuffer warns loudly on it.
    assert matchNeluxEncoder("does_not_exist") is None
    assert matchNeluxEncoder("png") is None
    assert matchNeluxEncoder("x264") is None  # only *_nelux names map


def testProresNeluxMirrorsQscale15():
    # -qscale:v 15 == flags +qscale with global_quality 15 * FF_QP2LAMBDA(118).
    mapping = matchNeluxEncoder("prores_nelux")
    assert mapping["codec"] == "prores_ks"
    assert mapping["pixel_format"] == "yuv444p10le"
    assert mapping["options"]["flags"] == "+qscale"
    assert mapping["options"]["global_quality"] == str(15 * 118)


def testVp9NeluxCarriesCrfThroughOptions():
    # nelux's cq only maps to x26x/svt/aom/NVENC; libvpx needs raw AVOptions,
    # and b=0 selects CRF-only mode.
    mapping = matchNeluxEncoder("vp9_nelux")
    assert mapping["codec"] == "libvpx-vp9"
    assert mapping["options"] == {"crf": "15", "b": "0"}
    assert "cq" not in mapping


@pytest.mark.parametrize(
    "method,pixFmt",
    [
        ("x264_10bit_nelux", "yuv420p10le"),
        ("x264_animation_10bit_nelux", "yuv420p10le"),
        ("x265_10bit_nelux", "yuv420p10le"),
        ("nvenc_h265_10bit_nelux", "p010le"),
    ],
)
def testTenBitNeluxMethodsSelectTenBitPixelFormats(method, pixFmt):
    mapping = matchNeluxEncoder(method)
    assert mapping["pixel_format"] == pixFmt
    assert "10" in mapping["options"]["profile"]


def testEightBitMappingIsUnchangedByDefault():
    # The default keeps the historical 8-bit pixel_format exactly.
    assert matchNeluxEncoder("x264_nelux")["pixel_format"] == "yuv420p"
    assert matchNeluxEncoder("x264_nelux") == matchNeluxEncoder("x264_nelux", "8bit")


@pytest.mark.parametrize(
    "method,expected",
    [
        ("x264_nelux", "yuv444p10le"),
        ("slow_x264_nelux", "yuv444p10le"),
        ("x264_animation_nelux", "yuv444p10le"),
        ("x265_nelux", "yuv444p10le"),
        ("slow_x265_nelux", "yuv444p10le"),
        ("vp9_nelux", "yuv444p10le"),
        ("lossless_nelux", "yuv444p10le"),
        ("nvenc_h265_nelux", "p010le"),
        ("slow_nvenc_h265_nelux", "p010le"),
        ("nvenc_av1_nelux", "p010le"),
        ("slow_nvenc_av1_nelux", "p010le"),
        # libsvtav1 has no 444 10-bit mode (nelux would fall back to 8-bit
        # yuv420p), so it keeps 420 subsampling at 10-bit depth.
        ("av1_nelux", "yuv420p10le"),
        ("slow_av1_nelux", "yuv420p10le"),
    ],
)
def testSixteenBitPromotesEightBitPixelFormatsToTenBit(method, expected):
    """--bit_depth 16bit must not silently narrow back to 8-bit: the mapping
    promotes the destination so uint16 frames keep their precision."""
    mapping = matchNeluxEncoder(method, "16bit")
    assert mapping["pixel_format"] == expected


@pytest.mark.parametrize(
    "method",
    [
        "x264_10bit_nelux",
        "x264_animation_10bit_nelux",
        "x265_10bit_nelux",
        "nvenc_h265_10bit_nelux",
        "prores_nelux",
    ],
)
def testSixteenBitLeavesDeepPixelFormatsAlone(method):
    # Already 10-bit destinations pass through untouched.
    assert matchNeluxEncoder(method, "16bit") == matchNeluxEncoder(method)


def testSixteenBitLeavesGifAlone():
    # No pixel_format to promote: the palette is 8-bit by nature and the
    # encoder narrows the input, as the FFmpeg twin does.
    assert matchNeluxEncoder("gif_nelux", "16bit") == matchNeluxEncoder("gif_nelux")


def testSixteenBitKeepsNvencH264AtEightBit():
    # No 10-bit H.264 NVENC path exists; mirrors getPixFMT's forced yuv420p.
    mapping = matchNeluxEncoder("nvenc_h264_nelux", "16bit")
    assert mapping["pixel_format"] == "yuv420p"
    slow = matchNeluxEncoder("slow_nvenc_h264_nelux", "16bit")
    assert slow["pixel_format"] == "yuv420p"
    lossless = matchNeluxEncoder("lossless_nvenc_nelux", "16bit")
    assert lossless["pixel_format"] == "yuv420p"


def testSixteenBitKeepsTheQualityKnobs():
    # Promotion touches only the pixel_format, never preset/cq/options.
    plain = matchNeluxEncoder("slow_x264_nelux")
    deep = matchNeluxEncoder("slow_x264_nelux", "16bit")
    for key in ("codec", "preset", "cq", "options"):
        assert deep[key] == plain[key]


def testUnknownNeluxEncoderStillReturnsNoneAtSixteenBit():
    assert matchNeluxEncoder("does_not_exist", "16bit") is None


def testSlowAv1NeluxPassesSvtPresetAsString():
    # A string preset is forwarded to FFmpeg verbatim; the int table would
    # remap 4 -> svt preset 9 (13-n) and quietly encode a different speed tier.
    mapping = matchNeluxEncoder("slow_av1_nelux")
    assert mapping["preset"] == "4"


def testLosslessNeluxTwinsUseQpZero():
    assert matchNeluxEncoder("lossless_nelux")["cq"] == 0
    assert matchNeluxEncoder("lossless_nvenc_nelux")["cq"] == 0


# --------------------------------------------------------------------------- #
# colorSpaceOptions: probed metadata -> nelux colour AVOptions
# --------------------------------------------------------------------------- #


def testDefaultSourceGetsBt709Options():
    # Mirrors BT709_FILTER's setparams: full matrix+primaries+transfer+range.
    assert colorSpaceOptions({}) == {
        "colorspace": "bt709",
        "color_primaries": "bt709",
        "color_trc": "bt709",
        "color_range": "tv",
    }


@pytest.mark.parametrize("field", ["ColorSpace", "PixelFormat", "ColorTRT"])
@pytest.mark.parametrize("value", ["bt2020", "bt2020nc", "bt2020c"])
def testBt2020DetectedOnAnyColourField(field, value):
    # Same detection as colorSpaceFilter: libavutil reports the matrix as
    # bt2020nc/bt2020c and bare primaries as "bt2020", on any of these fields.
    options = colorSpaceOptions({field: value})
    assert options["colorspace"] == "bt2020nc"
    assert options["color_primaries"] == "bt2020"
    assert options["color_range"] == "tv"


def testBt2020CopiesARecognizedSourceTransfer():
    options = colorSpaceOptions({"ColorSpace": "bt2020nc", "ColorTRT": "smpte2084"})
    assert options["color_trc"] == "smpte2084"


def testBt2020OmitsAnUnrecognizedTransfer():
    # An untagged stream is recoverable, a mislabelled one is not -- mirror
    # bt2020Filter and leave color_trc unset rather than guessing.
    options = colorSpaceOptions({"ColorSpace": "bt2020nc", "ColorTRT": "weird"})
    assert "color_trc" not in options
