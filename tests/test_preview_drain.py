"""Unit tests for WriteBuffer._drainPreview's mjpeg stream parser.

The drain thread reads ffmpeg's mjpeg preview stream and splits it into
individual JPEGs on the EOI (0xFFD9) marker, pushing each to the previewSink.
These tests pin that contract independent of ffmpeg: multiple frames per read,
an EOI straddling a chunk boundary, latest-only delivery, and a partial
trailing frame (no closing EOI) being withheld.
"""

import pytest

ffmpegSettings = pytest.importorskip("src.io.ffmpegSettings")


class RecordingSink:
    def __init__(self):
        self.frames = []

    def update(self, jpeg):
        self.frames.append(jpeg)


class ChunkedStdout:
    """Fake ffmpeg stdout that hands back at most `chunk` bytes per read1()."""

    def __init__(self, data: bytes, chunk: int):
        self.data = data
        self.chunk = chunk
        self.pos = 0

    def read1(self, _n):
        if self.pos >= len(self.data):
            return b""
        end = min(self.pos + self.chunk, len(self.data))
        out = self.data[self.pos : end]
        self.pos = end
        return out


# Fake JPEGs: SOI (FFD8) ... unique body ... EOI (FFD9). Bodies contain no FF.
F1 = b"\xff\xd8AAAA\xff\xd9"
F2 = b"\xff\xd8BBBB\xff\xd9"
F3 = b"\xff\xd8CCCC\xff\xd9"
STREAM = F1 + F2 + F3


def _drain(data, chunk):
    sink = RecordingSink()
    wb = ffmpegSettings.WriteBuffer(output="", previewSink=sink)
    wb._drainPreview(ChunkedStdout(data, chunk))
    return sink.frames


@pytest.mark.parametrize("chunk", [1, 2, 3, 5, 7, len(STREAM), len(STREAM) + 10])
def test_splits_frames_regardless_of_chunking(chunk):
    # Whether all three frames arrive in one read or byte-by-byte (forcing the
    # EOI marker to straddle reads), the parser must recover exactly F1, F2, F3.
    assert _drain(STREAM, chunk) == [F1, F2, F3]


def test_multiple_frames_in_single_chunk():
    assert _drain(STREAM, len(STREAM)) == [F1, F2, F3]


def test_latest_frame_is_last_emitted():
    frames = _drain(STREAM, 1)
    assert frames[-1] == F3


def test_partial_trailing_frame_is_withheld():
    # A frame with no closing EOI must not be emitted until it completes.
    frames = _drain(STREAM + b"\xff\xd8DDDD", 1)
    assert frames == [F1, F2, F3]


def test_empty_stream_emits_nothing():
    assert _drain(b"", 4) == []
