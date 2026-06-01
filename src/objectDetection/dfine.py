"""
D-FINE object detection decode helpers.

D-FINE (https://github.com/Peterande/D-FINE, Apache-2.0 code + weights) is a
real-time DETR-family detector. TAS consumes the RAW exported head — a single
image input and TWO raw outputs (no baked-in post-processor, no
``orig_target_sizes`` second input, which would break TAS's single-input ORT
feed dict and TRT binding loop):

    pred_logits [1, num_queries, num_classes]  raw per-class logits (focal/sigmoid)
    pred_boxes  [1, num_queries, 4]            cxcywh, normalized to [0, 1] of input

The detector is NMS-free (top-k query selection happens in-graph), so decoding is
just: sigmoid -> top-k over flattened (query, class) -> cxcywh->xyxy -> de-normalize
to input-pixel space. The reference D-FINE/RT-DETR ``RTDETRPostProcessor`` focal
path is reproduced here (flatten then top-k, label = idx % num_classes), so box
parity with the official post-processor holds.

COCO-80 class names / colors / drawing helpers are shared verbatim with
yolov9_mit so annotations look identical to the incumbent detector.
"""

import numpy as np

from .yolov9_mit import (  # noqa: F401  (re-exported for the backend classes)
    class_names,
    colors,
    draw_detections,
    draw_masks,
    draw_box,
)

# D-FINE decoder query count (num_top_queries); fixed by the exported graph.
NUM_TOP_QUERIES = 300


def splitDFineOutputs(outputs):
    """Identify (logits, boxes) from the model's two raw outputs.

    Export order is not guaranteed, so we key off shape: the boxes tensor is the
    one whose last dimension is 4; the other is the class logits.
    """
    arrays = [np.asarray(o) for o in outputs]
    boxesArrays = [a for a in arrays if a.shape[-1] == 4]
    logitsArrays = [a for a in arrays if a.shape[-1] != 4]
    # RAW head contract is exactly two outputs; fail fast rather than silently
    # mis-route if a future export adds an auxiliary tensor.
    if len(arrays) != 2 or len(boxesArrays) != 1 or len(logitsArrays) != 1:
        raise ValueError(
            "D-FINE RAW head must emit exactly two outputs (one boxes tensor with "
            "last dim 4, one logits tensor); got shapes "
            f"{[a.shape for a in arrays]}"
        )
    return logitsArrays[0], boxesArrays[0]


def _sigmoid(x):
    """Numerically-stable sigmoid (no exp-overflow warnings on large-magnitude
    negative focal logits); identical values to 1/(1+exp(-x))."""
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def _emptyDetections():
    return (
        np.array([], dtype=np.int64),
        np.empty((0, 4), dtype=np.float32),
        np.array([], dtype=np.float32),
    )


def decodeDFine(
    logits,
    boxes,
    confThreshold,
    inputWidth,
    inputHeight,
    numTopQueries=NUM_TOP_QUERIES,
):
    """Decode raw D-FINE outputs into ``(classIds, boxesXyxy, confidences)``.

    ``boxesXyxy`` are returned in MODEL-INPUT pixel space (scaled to
    ``inputWidth`` x ``inputHeight``) so the caller's existing ``rescaleBoxes()``
    maps them to original-image pixels unchanged.
    """
    logits = np.asarray(logits, dtype=np.float32)
    boxes = np.asarray(boxes, dtype=np.float32)

    if logits.ndim == 3:
        logits = logits[0]
    if boxes.ndim == 3:
        boxes = boxes[0]

    numClasses = logits.shape[-1]

    # Focal/sigmoid scoring: every (query, class) pair is an independent candidate.
    scores = _sigmoid(logits)  # [Q, C]
    flat = scores.reshape(-1)  # [Q * C]

    k = int(min(numTopQueries, flat.shape[0]))
    if k <= 0:
        return _emptyDetections()

    # argpartition leaves the top-k unordered; sort descending so detections (and
    # therefore draw order, highest-confidence on top) are deterministic.
    topIdx = np.argpartition(flat, -k)[-k:]
    topIdx = topIdx[np.argsort(-flat[topIdx])]
    topScores = flat[topIdx]

    classIds = (topIdx % numClasses).astype(np.int64)
    queryIdx = topIdx // numClasses

    keep = topScores > confThreshold
    if not np.any(keep):
        return _emptyDetections()

    topScores = topScores[keep]
    classIds = classIds[keep]
    selBoxes = boxes[queryIdx[keep]]  # [N, 4] cxcywh normalized

    cx = selBoxes[:, 0]
    cy = selBoxes[:, 1]
    w = selBoxes[:, 2]
    h = selBoxes[:, 3]

    x1 = (cx - w * 0.5) * inputWidth
    y1 = (cy - h * 0.5) * inputHeight
    x2 = (cx + w * 0.5) * inputWidth
    y2 = (cy + h * 0.5) * inputHeight
    boxesXyxy = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

    classIds = np.clip(classIds, 0, len(colors) - 1)
    return classIds, boxesXyxy, topScores.astype(np.float32)
