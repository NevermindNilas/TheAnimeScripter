"""`--output_scale` must reach every writer, not just main.py's.

Only `main.py` ever passed `output_scale_width`/`output_scale_height` to a
writer. Every standalone driver -- depth, segment, stabilize (classic and dut),
motion blur, object detection -- builds its writer from a long positional
argument list that never carried them, so the requested resolution was silently
discarded for half the toolkit while `src/cli/validator.py` logged "Output scale
set to WxH". `src/stabilize/dutStabilizer.py` inherited the same omission the
day it was added, which is why the value is resolved once at the writer instead
of threaded through ~20 driver constructors.

`tests/conftest.py` resets the constants between tests.
"""

import pytest

import src.constants as cs

pytest.importorskip("torch")

from src.io.ffmpegSettings import (  # noqa: E402
    NeluxWriteBuffer,
    WriteBuffer,
    _resolveOutputScale,
)


def _scaleFilter(writer):
    # The colour-space entry also starts with `scale=` (`scale=in_range=pc:...`),
    # so match the resize filter by its flags rather than by its prefix.
    return next(
        (f for f in writer._buildFilterList() if ":flags=bilinear" in f),
        None,
    )


def testUnsetByDefault():
    assert _resolveOutputScale(None, None) == (None, None)
    assert _scaleFilter(WriteBuffer(output="out.mp4")) is None


def testDriverWithoutTheKwargPicksUpTheRunWideScale():
    # The regression: this is exactly how a standalone driver builds its writer.
    cs.OUTPUT_SCALE_WIDTH, cs.OUTPUT_SCALE_HEIGHT = 320, 180

    assert _scaleFilter(WriteBuffer(output="out.mp4")) == "scale=320:180:flags=bilinear"


def testScaleFilterPrecedesTheFormatConversion():
    # Order matters: scaling a gray16be or yuva444p10le plane instead of the
    # source frame would resample the wrong thing.
    cs.OUTPUT_SCALE_WIDTH, cs.OUTPUT_SCALE_HEIGHT = 320, 180
    filters = WriteBuffer(output="out.mp4", grayscale=True)._buildFilterList()

    scaleAt = next(i for i, f in enumerate(filters) if ":flags=bilinear" in f)
    formatAt = next(i for i, f in enumerate(filters) if f.startswith("format="))
    assert scaleAt < formatAt


def testExplicitArgumentWins():
    cs.OUTPUT_SCALE_WIDTH, cs.OUTPUT_SCALE_HEIGHT = 320, 180
    writer = WriteBuffer(
        output="out.mp4", output_scale_width=640, output_scale_height=360
    )

    assert _scaleFilter(writer) == "scale=640:360:flags=bilinear"


def testNeluxWriterHonoursTheRunWideScale():
    cs.OUTPUT_SCALE_WIDTH, cs.OUTPUT_SCALE_HEIGHT = 320, 180
    writer = NeluxWriteBuffer(
        output="out.mp4", encode_method="x264_nelux", width=1920, height=1080
    )

    assert (writer.outputWidth, writer.outputHeight) == (320, 180)
    assert writer.encoderKwargs.get("resize") is True


def testPartialPairIsTreatedAsUnset():
    # Both consumers guard with `width and height`, so half a pair must not be
    # allowed to fall through to the global and silently mix the two sources.
    cs.OUTPUT_SCALE_WIDTH, cs.OUTPUT_SCALE_HEIGHT = 320, 180

    assert _resolveOutputScale(640, None) == (640, None)
