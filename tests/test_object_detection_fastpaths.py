import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cv2")
pytest.importorskip("onnxruntime")

from src.objectDetection.objectDetection import (
    _rescaleBoxes,
    _writeOutputFrame,
    _writeRgbFrame,
)


class _Writer:
    def __init__(self):
        self.frame = None

    def write(self, frame):
        self.frame = frame


class _BoxOwner:
    inputWidth = 10
    inputHeight = 20
    imgWidth = 100
    imgHeight = 200
    _boxScale = None
    _boxScaleShape = None


def testWriteOutputFrameCanEmitHwcUint8ForNelux():
    writer = _Writer()
    bgr = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)

    _writeOutputFrame(writer, bgr, writeHwcUint8=True)

    assert writer.frame.shape == (1, 2, 3)
    assert writer.frame.dtype == torch.uint8
    assert writer.frame.is_contiguous()
    assert writer.frame.tolist() == [[[3, 2, 1], [6, 5, 4]]]


def testWriteRgbFrameCanBypassColorConversionForNoDetections():
    writer = _Writer()
    rgb = np.array([[[3, 2, 1], [6, 5, 4]]], dtype=np.uint8)

    _writeRgbFrame(writer, rgb, writeHwcUint8=True)

    assert writer.frame.shape == (1, 2, 3)
    assert writer.frame.dtype == torch.uint8
    assert writer.frame.tolist() == [[[3, 2, 1], [6, 5, 4]]]


def testWriteRgbFrameCanOwnDecoderBackedFrames():
    writer = _Writer()
    rgb = np.array([[[3, 2, 1]]], dtype=np.uint8)

    _writeRgbFrame(writer, rgb, writeHwcUint8=True, copyHwc=True)
    rgb[:] = 0

    assert writer.frame.tolist() == [[[3, 2, 1]]]


def testWriteOutputFrameKeepsBchwFloatForFfmpegWriter():
    writer = _Writer()
    bgr = np.array([[[0, 127, 255]]], dtype=np.uint8)

    _writeOutputFrame(writer, bgr, writeHwcUint8=False)

    assert writer.frame.shape == (1, 3, 1, 1)
    assert writer.frame.dtype == torch.float32
    assert torch.allclose(
        writer.frame.flatten(),
        torch.tensor([1.0, 127.0 / 255.0, 0.0], dtype=torch.float32),
    )


def testRescaleBoxesUsesCachedScaleInPlace():
    owner = _BoxOwner()
    boxes = np.array([[1, 2, 3, 4]], dtype=np.float32)

    scaled = _rescaleBoxes(owner, boxes)

    assert scaled is boxes
    np.testing.assert_allclose(scaled, [[10, 20, 30, 40]])
    assert owner._boxScaleShape == (100, 200)
