import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
cv2 = pytest.importorskip("cv2")
pytest.importorskip("onnxruntime")

from src.objectDetection.objectDetection import (
    _rescaleBoxes,
    _writeRgbFrame,
)
from src.objectDetection.yolov9_mit import (
    colors,
    colors_rgb,
    draw_box,
    draw_detections,
    draw_masks,
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


def testColorsRgbIsExactChannelReversal():
    # The RGB-draw path relies on colors_rgb being the exact per-channel
    # reversal of colors so overlays land byte-identical to the old
    # draw-in-BGR-then-swap path.
    np.testing.assert_array_equal(colors_rgb, colors[:, ::-1])


def _oldBgrRoundTrip(frameRgb, boxes, classIds, scores):
    frameBgr = cv2.cvtColor(frameRgb, cv2.COLOR_RGB2BGR)
    out = draw_detections(frameBgr, boxes, scores, classIds)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def testDrawInRgbIsByteIdenticalToOldBgrRoundTrip():
    # Core contract of the objdetect RGB-draw optimization: drawing straight
    # onto the RGB frame with the reversed palette must be byte-identical to
    # the removed RGB->BGR->draw->BGR->RGB round trip.
    rng = np.random.default_rng(7)
    frame = rng.integers(0, 256, (64, 96, 3), dtype=np.uint8)
    n = 8
    x1 = rng.integers(0, 60, n)
    y1 = rng.integers(0, 40, n)
    boxes = np.stack(
        [x1, y1, x1 + rng.integers(4, 30, n), y1 + rng.integers(4, 20, n)], 1
    ).astype(np.float32)
    classIds = rng.integers(0, len(colors), n)
    scores = rng.uniform(0.2, 0.99, n).astype(np.float32)

    old = _oldBgrRoundTrip(frame, boxes, classIds, scores)
    new = draw_detections(frame, boxes, scores, classIds, palette=colors_rgb)
    np.testing.assert_array_equal(old, new)

    # disableAnnotations branch (draw_masks + draw_box) must match too.
    oldBgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    oldBgr = draw_masks(oldBgr, boxes, classIds, mask_alpha=0.3)
    for classId, box in zip(classIds, boxes, strict=False):
        draw_box(oldBgr, box, colors[classId])
    oldDisabled = cv2.cvtColor(oldBgr, cv2.COLOR_BGR2RGB)

    newDisabled = frame.copy()
    newDisabled = draw_masks(
        newDisabled, boxes, classIds, mask_alpha=0.3, palette=colors_rgb
    )
    for classId, box in zip(classIds, boxes, strict=False):
        draw_box(newDisabled, box, colors_rgb[classId])
    np.testing.assert_array_equal(oldDisabled, newDisabled)


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


def testWriteRgbFrameKeepsBchwFloatForFfmpegWriter():
    writer = _Writer()
    rgb = np.array([[[255, 127, 0]]], dtype=np.uint8)

    _writeRgbFrame(writer, rgb, writeHwcUint8=False)

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
