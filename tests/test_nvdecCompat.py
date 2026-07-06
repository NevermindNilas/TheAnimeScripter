"""Tests for src/io/getVideoMetadata.isNvdecCompatible.

The NVDEC decode path in nelux only handles compressed codecs
(H.264/HEVC/VP9/AV1/...) with YUV-family pixel formats. Feeding it an
incompatible source (e.g. an AE-bridge prerender AVI with codec=rawvideo,
pix_fmt=bgr24) deadlocks or fails opaquely inside nelux.VideoReader
construction, so main.py._initVideoMetadata downgrades to CPU decode
when this helper returns False. These tests pin the compat decision so
the gate doesn't silently regress.
"""

import pytest

import src.constants as cs
from src.io.getVideoMetadata import getVideoMetadata, isNvdecCompatible


@pytest.mark.parametrize(
    "codec, pix_fmt",
    [
        ("h264", "yuv420p"),
        ("hevc", "yuv420p10le"),
        ("vp9", "yuv420p"),
        ("av1", "yuv420p10le"),
        ("mpeg2video", "yuv420p"),
        ("mpeg4", "yuv420p"),
        ("vp8", "yuv420p"),
        ("mjpeg", "yuvj420p"),
        ("h264", "nv12"),
        ("hevc", "p010le"),
        ("h264", "yuv444p"),
        ("h264", "yuv444p16le"),
        # Case-insensitive: uppercase input is accepted.
        ("H264", "YUV420P"),
        ("HEVC", "NV12"),
    ],
)
def test_isNvdecCompatible_allows_compressed_yuv(codec, pix_fmt):
    assert isNvdecCompatible(codec, pix_fmt) is True


@pytest.mark.parametrize(
    "codec, pix_fmt",
    [
        # The exact AE-bridge prerender profile that motivated the guard:
        ("rawvideo", "bgr24"),
        # Other raw / lossless intermediate codecs with no cuvid decoder:
        ("rawvideo", "yuv420p"),
        ("ffv1", "yuv420p"),
        ("huffyuv", "yuv420p"),
        ("lagarith", "yuv420p"),
        ("utvideo", "yuv420p"),
        # Compressed codec but packed RGB pix_fmt (NVDEC outputs YUV only):
        ("h264", "rgb24"),
        ("h264", "bgr0"),
        ("hevc", "gbrp"),
        ("hevc", "gbrap10le"),
        ("vp9", "pal8"),
        ("h264", "ya8"),
        ("h264", "ya16"),
    ],
)
def test_isNvdecCompatible_rejects_raw_or_rgb(codec, pix_fmt):
    assert isNvdecCompatible(codec, pix_fmt) is False


def test_isNvdecCompatible_handles_empty_or_unknown():
    # Empty values are treated as "no signal" -> allowed (don't surprise-
    # downgrade when ffprobe failed to populate the field).
    assert isNvdecCompatible("", "") is True
    assert isNvdecCompatible(None, None) is True
    # Unknown codec/pix_fmt is allowed through (conservative: only rule out
    # what NVDEC provably cannot handle).
    assert isNvdecCompatible("somefuturecodec", "somefuturepixfmt") is True


def test_getVideoMetadata_validates_ffprobe_path(monkeypatch, tmp_path):
    inputPath = tmp_path / "clip.mp4"
    ffmpegPath = tmp_path / "ffmpeg.exe"
    inputPath.write_bytes(b"not a real video")
    ffmpegPath.write_bytes(b"")

    monkeypatch.setattr(cs, "FFMPEGPATH", str(ffmpegPath), raising=False)
    monkeypatch.setattr(
        cs, "FFPROBEPATH", str(tmp_path / "missing-ffprobe.exe"), raising=False
    )

    with pytest.raises(FileNotFoundError, match="ffprobe path not found"):
        getVideoMetadata(str(inputPath), 0, 0)
