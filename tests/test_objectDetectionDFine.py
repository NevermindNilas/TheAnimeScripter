"""Unit tests for the D-FINE DETR decode (src/objectDetection/dfine.py).

These exercise the pure decode math (sigmoid -> top-k -> cxcywh->xyxy ->
de-normalize) without needing the ONNX weights or a GPU. The dfine module pulls
COCO names/draw helpers from yolov9_mit, which imports cv2 + onnxruntime, so we
importorskip those.
"""

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("onnxruntime")

from src.objectDetection.dfine import decodeDFine, splitDFineOutputs  # noqa: E402


def test_split_outputs_is_order_independent():
    logits = np.zeros((1, 300, 80), np.float32)
    boxes = np.zeros((1, 300, 4), np.float32)

    lg, bx = splitDFineOutputs([logits, boxes])
    assert lg.shape[-1] == 80 and bx.shape[-1] == 4

    # reversed export order must still resolve correctly (keyed off last dim == 4)
    lg, bx = splitDFineOutputs([boxes, logits])
    assert lg.shape[-1] == 80 and bx.shape[-1] == 4


def test_split_outputs_rejects_ambiguous():
    bad = np.zeros((1, 300, 4), np.float32)
    with pytest.raises(ValueError):
        splitDFineOutputs([bad, bad])  # two box-shaped tensors, no logits


def test_decode_box_math_and_class_pick():
    queries, classes = 5, 80
    logits = np.full((1, queries, classes), -10.0, np.float32)  # ~0 probability
    logits[0, 2, 7] = 10.0  # query 2, class 7 strongly positive

    boxes = np.zeros((1, queries, 4), np.float32)
    boxes[0, 2] = [0.5, 0.5, 0.2, 0.4]  # cx, cy, w, h normalized

    classIds, boxesXyxy, conf = decodeDFine(
        logits, boxes, confThreshold=0.25, inputWidth=640, inputHeight=640,
        numTopQueries=queries * classes,
    )

    assert classIds.tolist() == [7]
    assert conf[0] > 0.99
    # cx 0.5*640=320, w 0.2*640=128 -> x1=256, x2=384
    # cy 0.5*640=320, h 0.4*640=256 -> y1=192, y2=448
    np.testing.assert_allclose(boxesXyxy[0], [256.0, 192.0, 384.0, 448.0], atol=1e-3)


def test_decode_threshold_filters_everything():
    logits = np.full((1, 4, 80), -10.0, np.float32)  # all near-zero score
    boxes = np.zeros((1, 4, 4), np.float32)

    classIds, boxesXyxy, conf = decodeDFine(
        logits, boxes, confThreshold=0.25, inputWidth=640, inputHeight=640
    )

    assert classIds.shape == (0,)
    assert boxesXyxy.shape == (0, 4)
    assert conf.shape == (0,)


def test_decode_accepts_unbatched_inputs():
    logits = np.full((6, 80), -10.0, np.float32)
    logits[1, 3] = 8.0
    boxes = np.zeros((6, 4), np.float32)
    boxes[1] = [0.25, 0.25, 0.1, 0.1]

    classIds, boxesXyxy, conf = decodeDFine(
        logits, boxes, confThreshold=0.25, inputWidth=640, inputHeight=480,
        numTopQueries=6 * 80,
    )
    assert classIds.tolist() == [3]
    # de-normalized against differing W/H: x uses 640, y uses 480
    np.testing.assert_allclose(
        boxesXyxy[0], [0.2 * 640, 0.2 * 480, 0.3 * 640, 0.3 * 480], atol=1e-3
    )


def test_decode_clips_class_ids_into_palette_range():
    # a logits tensor wider than the color palette must not index out of range
    queries, classes = 3, 200
    logits = np.full((1, queries, classes), -10.0, np.float32)
    logits[0, 0, 199] = 9.0
    boxes = np.zeros((1, queries, 4), np.float32)
    boxes[0, 0] = [0.5, 0.5, 0.1, 0.1]

    classIds, _, _ = decodeDFine(
        logits, boxes, confThreshold=0.25, inputWidth=640, inputHeight=640,
        numTopQueries=queries * classes,
    )
    from src.objectDetection.dfine import colors

    assert classIds[0] <= len(colors) - 1
