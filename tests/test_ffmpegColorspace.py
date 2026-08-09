"""Tests for WriteBuffer._buildFilterList colorspace handling.

The BT.709 path was migrated from `zscale` (zimg, ~3x CPU) to swscale `scale`,
routed through a 16-bit working format so the downstream depth reduction still
error-diffusion-dithers, plus `setparams` to fully tag the stream. BT.2020 stays
on zscale (no swscale bt2020nc matrix / transfer norm). These pin that contract so
the conversion can't silently regress to the wrong matrix or lose dithering/tags.
"""

import json

import pytest

# ffmpegSettings imports torch/nelux/cv2 at module load; skip cleanly without them.
# nelux can be installed yet still raise ImportError because FFmpeg DLLs are only
# put on the search path during runtime argument preparation.
torch = pytest.importorskip("torch")
pytest.importorskip("nelux", exc_type=ImportError)
pytest.importorskip("cv2")

import src.constants as cs
from src.io.ffmpegSettings import WriteBuffer


def _colorFilter(filterList) -> str:
    """Return the colorspace filter entry (matrix conversion), or "" if absent."""
    hits = [f for f in filterList if "out_color_matrix" in f or "zscale=matrix" in f]
    return hits[0] if hits else ""


def testBt709UsesSwscaleNotZscale(monkeypatch):
    monkeypatch.setattr(cs, "METADATAPATH", "")  # no metadata -> defaults to bt709
    wb = WriteBuffer(output="")
    f = _colorFilter(wb._buildFilterList())
    assert f is not None
    assert "scale=" in f and "out_color_matrix=bt709" in f
    assert "zscale" not in f  # zimg path must be gone for 709


def testBt709DithersViaWideIntermediate(monkeypatch):
    # swscale only dithers on a depth reduction, so the chain must pass through
    # a 16-bit working format before the final -pix_fmt step.
    monkeypatch.setattr(cs, "METADATAPATH", "")
    wb = WriteBuffer(output="")
    f = _colorFilter(wb._buildFilterList())
    assert f is not None
    assert "format=yuv444p16le" in f


def testBt709FullyTagged(monkeypatch):
    monkeypatch.setattr(cs, "METADATAPATH", "")
    wb = WriteBuffer(output="")
    f = _colorFilter(wb._buildFilterList())
    assert f is not None
    assert "setparams=" in f
    for tag in (
        "colorspace=bt709",
        "color_primaries=bt709",
        "color_trc=bt709",
        "range=tv",
    ):
        assert tag in f


def testBt2020KeepsZscale(tmp_path, monkeypatch):
    meta = tmp_path / "meta.json"
    meta.write_text(
        json.dumps({"metadata": {"ColorSpace": "bt2020"}}), encoding="utf-8"
    )
    monkeypatch.setattr(cs, "METADATAPATH", str(meta))
    wb = WriteBuffer(output="")
    f = _colorFilter(wb._buildFilterList())
    assert "zscale=matrix=bt2020" in f
    assert "norm=bt2020" in f  # transfer handling swscale can't do


def testGrayscaleSkipsColorspaceFilter(monkeypatch):
    monkeypatch.setattr(cs, "METADATAPATH", "")
    wb = WriteBuffer(output="", grayscale=True)
    filterList = wb._buildFilterList()
    assert _colorFilter(filterList) == ""
    assert any("gray" in f for f in filterList)


def testTransparentSkipsColorspaceFilter(monkeypatch):
    """The alpha path must reach the encoder at full chroma and 10 bits.

    It used to be pinned to `yuva420p`, which is 8-bit with 2x2-subsampled
    chroma, ahead of a `yuva444p10le` output that cannot recover either -- so
    the ProRes 4444 file `--segment` exists to produce carried 4:2:0 8-bit
    chroma, fringing along the matte edges.
    """
    monkeypatch.setattr(cs, "METADATAPATH", "")
    wb = WriteBuffer(output="", transparent=True)
    filterList = wb._buildFilterList()
    assert _colorFilter(filterList) == ""
    assert any("yuva444p10le" in f for f in filterList)
    assert not any("yuva420p" in f for f in filterList)


def _audioSettings(monkeypatch, output, subtitleCodecs=None):
    """Built audio/subtitle flags for an output path, with a stubbed probe."""
    monkeypatch.setattr(cs, "METADATAPATH", "")
    wb = WriteBuffer(output=output, input="source.mkv")
    monkeypatch.setattr(
        type(wb),
        "_isoBmffSubtitleCodec",
        lambda self: (
            "copy"
            if subtitleCodecs
            and all(c in type(self)._ISO_BMFF_NATIVE_SUBTITLES for c in subtitleCodecs)
            else "mov_text"
        ),
    )
    settings = wb._buildAudioSettings()
    return settings[settings.index("-c:a") + 1], settings[settings.index("-c:s") + 1]


@pytest.mark.parametrize(
    "output,expected",
    [
        ("out.mp4", ("copy", "mov_text")),
        ("out.m4v", ("copy", "mov_text")),
        ("out.mov", ("aac", "mov_text")),
        ("out.mkv", ("copy", "copy")),
        ("out.webm", ("libopus", "webvtt")),
        ("out.avi", ("copy", "copy")),
        # inputOutputHandler matches extensions case-insensitively and copies
        # the input's verbatim, so an uppercase name is ordinary here.
        ("OUT.MP4", ("copy", "mov_text")),
        ("OUT.MOV", ("aac", "mov_text")),
    ],
)
def testPerContainerAudioAndSubtitleCodecs(monkeypatch, output, expected):
    """MP4/M4V used to stream-copy subrip into a container that cannot hold it,
    losing the whole render at header-write."""
    assert _audioSettings(monkeypatch, output, ["subrip"]) == expected


def testIsoBmffNativeSubtitlesAreCopiedNotTranscoded(monkeypatch):
    """FFmpeg has a TTML encoder but no TTML decoder, so forcing mov_text on a
    TTML-in-MP4 source failed the whole mux where a copy had worked."""
    assert _audioSettings(monkeypatch, "out.mp4", ["ttml"]) == ("copy", "copy")
    assert _audioSettings(monkeypatch, "out.mp4", ["mov_text"]) == ("copy", "copy")
    assert _audioSettings(monkeypatch, "out.mp4", ["ttml", "subrip"]) == (
        "copy",
        "mov_text",
    )
