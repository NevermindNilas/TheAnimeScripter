"""Real encoder checks; skipped in the dependency-free CI environment."""

import pytest

torch = pytest.importorskip("torch")
nelux = pytest.importorskip("nelux", exc_type=ImportError)

from src.io.encodingSettings import matchNeluxEncoder


@pytest.mark.parametrize("bitDepth", ["8bit", "16bit"])
def testSlowX265EncodesWithCompatibleProfile(tmp_path, bitDepth):
    output = tmp_path / "encoded.mp4"
    mapping = matchNeluxEncoder("slow_x265_nelux", bitDepth)
    encoder = nelux.VideoEncoder(str(output), width=64, height=64, fps=24, **mapping)
    try:
        dtype = torch.uint16 if bitDepth == "16bit" else torch.uint8
        encoder.encode_frame(torch.zeros((64, 64, 3), dtype=dtype))
    finally:
        encoder.close()
    reader = nelux.VideoReader(str(output), decode_accelerator="cpu")
    try:
        props = reader.get_properties()
    finally:
        del reader
    assert props["pixel_format"] == mapping["pixel_format"]
    assert output.stat().st_size > 0
