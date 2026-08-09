"""Tests for WriteBuffer._buildFilterList colorspace handling.

The BT.709 path was migrated from `zscale` (zimg, ~3x CPU) to swscale `scale`,
routed through a 16-bit working format so the downstream depth reduction still
error-diffusion-dithers, plus `setparams` to fully tag the stream. BT.2020 stays
on zscale (swscale has no bt2020nc matrix). These pin that contract so
the conversion can't silently regress to the wrong matrix or lose dithering/tags.
"""

import json
import os
import shutil
import subprocess

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


def _bt2020Filter(tmp_path, monkeypatch, transfer=None):
    meta = tmp_path / "meta.json"
    payload = {"ColorSpace": "bt2020"}
    if transfer is not None:
        payload["ColorTRT"] = transfer
    meta.write_text(json.dumps({"metadata": payload}), encoding="utf-8")
    monkeypatch.setattr(cs, "METADATAPATH", str(meta))
    return _colorFilter(WriteBuffer(output="")._buildFilterList())


def testBt2020KeepsZscale(tmp_path, monkeypatch):
    """swscale has no bt2020nc matrix, so this arm stays on zscale -- but it
    used to ask for `matrix=bt2020:norm=bt2020`, and neither token exists
    (the constants are 2020_ncl/2020_cl, and there is no `norm` option). The
    filtergraph could not parse, so every BT.2020/HDR10 source failed its whole
    render. This test pinned `norm=bt2020`, which is how it shipped."""
    f = _bt2020Filter(tmp_path, monkeypatch)
    assert "zscale=matrix=bt2020nc" in f
    assert "norm=" not in f  # not a zscale option; it made the graph unparseable
    assert "matrix=bt2020:" not in f


def testBt2020FilterIsAcceptedByFfmpeg(tmp_path, monkeypatch):
    """A string test cannot tell a valid filtergraph from an unparseable one,
    which is exactly how the broken arm survived. Run it."""
    # Prefer the BUNDLED binary: that is the one production runs, and a build
    # without --enable-libzimg would pass against a system ffmpeg and still
    # fail at runtime.
    bundled = os.path.join("ffmpeg_shared", "ffmpeg.exe")
    ffmpeg = (
        cs.FFMPEGPATH
        or (bundled if os.path.exists(bundled) else None)
        or shutil.which("ffmpeg")
    )
    if not ffmpeg or not os.path.exists(str(ffmpeg)):
        pytest.skip("no ffmpeg binary available")

    result = subprocess.run(
        [
            str(ffmpeg),
            "-hide_banner",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=s=64x64:d=1:r=5",
            "-vf",
            _bt2020Filter(tmp_path, monkeypatch),
            "-frames:v",
            "2",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"filtergraph rejected: {result.stderr.strip()[:300]}"
    )


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


def _probedSubtitleCodec(monkeypatch, tmp_path, name, codecs="ttml", create=True):
    """The real _isoBmffSubtitleCodec, with only ffprobe stubbed out."""
    monkeypatch.setattr(cs, "METADATAPATH", "")
    source = tmp_path / name
    if create:
        source.write_bytes(b"x")
    wb = WriteBuffer(output="out.mp4", input=str(source))

    probed = []

    class _Result:
        stdout = codecs

    def fakeRun(cmd, *a, **k):
        probed.append(cmd)
        return _Result()

    monkeypatch.setattr(subprocess, "run", fakeRun)
    return wb._isoBmffSubtitleCodec(), probed


def testAPercentInTheSourceNameIsStillProbed(monkeypatch, tmp_path):
    """A literal percent is not an image-sequence pattern. Short-circuiting on
    one forced mov_text onto a TTML source, which FFmpeg cannot decode -- the
    0-byte mux failure this method was written to prevent."""
    codec, probed = _probedSubtitleCodec(monkeypatch, tmp_path, "50%_off.mkv")
    assert codec == "copy"
    assert probed, "the probe never ran for a source with a percent in its name"


def testAnImageSequencePatternIsNotProbed(monkeypatch, tmp_path):
    """The pattern is not a path on disk, so the existence check catches it and
    no ffprobe subprocess is spawned."""
    codec, probed = _probedSubtitleCodec(
        monkeypatch, tmp_path, "frames_%05d.png", create=False
    )
    assert codec == "mov_text"
    assert not probed


@pytest.mark.parametrize(
    "transfer", ["smpte2084", "arib-std-b67", "bt2020-10", "bt2020-12"]
)
def testBt2020KeepsTheSourceTransfer(tmp_path, monkeypatch, transfer):
    """BT.2020 covers PQ, HLG and SDR, and only the matrix is converted here.
    Hardcoding smpte2084 relabelled an HLG master as PQ, which tells the player
    to apply the wrong EOTF."""
    f = _bt2020Filter(tmp_path, monkeypatch, transfer)
    assert f"color_trc={transfer}" in f


def testBt2020OmitsTheTransferWhenUnknown(tmp_path, monkeypatch):
    """An untagged stream is recoverable; a mislabelled one is not."""
    f = _bt2020Filter(tmp_path, monkeypatch, "unknown")
    assert "color_trc=" not in f
    assert "colorspace=bt2020nc" in f
