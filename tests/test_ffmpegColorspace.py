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
# nelux can be installed yet still raise ImportError (FFmpeg DLLs are only put on
# the search path at runtime by argumentsChecker), hence exc_type=ImportError.
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
    for tag in ("colorspace=bt709", "color_primaries=bt709", "color_trc=bt709", "range=tv"):
        assert tag in f


def testBt2020KeepsZscale(tmp_path, monkeypatch):
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"metadata": {"ColorSpace": "bt2020"}}), encoding="utf-8")
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
    monkeypatch.setattr(cs, "METADATAPATH", "")
    wb = WriteBuffer(output="", transparent=True)
    filterList = wb._buildFilterList()
    assert _colorFilter(filterList) == ""
    assert any("yuva420p" in f for f in filterList)
