"""Colour-conversion filter strings, tested without torch.

These live apart from `test_ffmpegColorspace.py` on purpose. That module does a
*module-level* `importorskip("torch")` because it constructs a `WriteBuffer`,
and CI installs only pytest/packaging/barflow -- so the whole file skips there.
That is how a BT.2020 filter string FFmpeg could not even parse shipped, with a
test asserting the broken token. `src/io/encodingSettings.py` imports nothing
heavier than the logger, so everything here runs in CI.
"""

import pytest

from src.io.encodingSettings import (
    BT709_FILTER,
    BT2020_COLOR_VALUES,
    SETPARAMS_TRANSFERS,
    bt2020Filter,
    colorSpaceFilter,
)


def testUnknownMetadataFallsBackToBt709():
    assert colorSpaceFilter({}) == BT709_FILTER
    assert colorSpaceFilter({"ColorSpace": "smpte170m"}) == BT709_FILTER


@pytest.mark.parametrize("value", BT2020_COLOR_VALUES)
@pytest.mark.parametrize("field", ["ColorSpace", "PixelFormat", "ColorTRT"])
def testAnyBt2020FieldSelectsTheZscaleArm(field, value):
    """libavutil reports the BT.2020 *matrix* as bt2020nc/bt2020c and the
    *primaries* as a bare bt2020. Keying only on "bt2020" meant a source
    carrying the matrix but no primaries was converted to, and tagged as,
    BT.709."""
    assert "zscale=matrix=bt2020nc" in colorSpaceFilter({field: value})


@pytest.mark.parametrize("transfer", sorted(SETPARAMS_TRANSFERS))
def testTheSourceTransferIsCopiedNotAssumed(transfer):
    """BT.2020 covers PQ, HLG and SDR, and this arm converts only the matrix.
    Hardcoding smpte2084 relabelled an HLG master as PQ, which tells the player
    to apply the wrong EOTF."""
    assert f"color_trc={transfer}" in bt2020Filter(transfer)


@pytest.mark.parametrize("transfer", ["unknown", "", "nonsense", None])
def testAnUnknownTransferIsLeftUntagged(transfer):
    """An untagged stream is recoverable; a mislabelled one is not."""
    built = bt2020Filter(transfer)
    assert "color_trc=" not in built
    assert "colorspace=bt2020nc" in built and "color_primaries=bt2020" in built


def testBt2020DoesNotCrushToEightBitInsideTheGraph():
    """`format=yuv420p` here reduced a 16-bit HDR frame to 8-bit 4:2:0 before
    the encoder's own -pix_fmt ever ran: a 16-bit ramp came out as 220 distinct
    10-bit codes, every one a multiple of 4."""
    built = bt2020Filter("smpte2084")
    assert "format=yuv444p16le" in built
    assert "format=yuv420p" not in built


def testBothArmsDitherAndTagFully():
    for built in (BT709_FILTER, bt2020Filter("smpte2404")):
        assert "setparams=" in built
        assert "range=tv" in built
    assert "dither=error_diffusion" in bt2020Filter("smpte2084")
