"""Roto documents: tracked mattes plus hand-drawn shapes plus edge refinement.

``--segment_mask`` takes either a folder of per-frame mattes or a ``.json`` roto
document a front end writes::

    {
      "version": 1,
      "mattes": "<folder of 000123.png mattes, or null>",
      "width": 1280, "height": 720,          # the frame size shape points are in
      "shapes": [ ... ],                     # applied in order, see below
      "refine": {"feather": 0, "contrast": 0, "shift": 0, "chatter": 0,
                 "specks": 0, "holes": 0, "decontaminate": false}
    }

A shape adds to or cuts from the matte on a span of source frames::

    {"kind": "polygon" | "stroke", "mode": "add" | "subtract",
     "start": 10, "end": 40,
     "keys": [{"frame": 10, "points": [[x, y, inX, inY, outX, outY], ...]}, ...],
     "radius": 12,      # stroke only: half its width
     "feather": 0}      # polygon only: edge softness in pixels

A polygon's points are Bezier vertices with absolute handle positions, keyed on
frames and interpolated linearly in between (every key has the same number of
points). A stroke's keys hold ``[x, y]`` points, painted with round ends; it has
one key unless it was tracked through the picture.

The same functions build the front end's preview and the render, so what the
preview shows is what the render writes.
"""

import itertools
import json
import logging
import os
from collections import OrderedDict

import cv2
import numpy as np

from src.masking import listMaskFrames, readMaskImage

_BEZIER_STEPS = 16
_SUBPIXEL_SHIFT = 3
_MAX_CHATTER = 5
_MAX_SHAPES = 2000
_CACHE_FRAMES = 2 * _MAX_CHATTER + 3
_INTEGER_FULL_SCALE = {"uint8": 255.0, "uint16": 65535.0}

REFINE_DEFAULTS = {
    "feather": 0.0,
    "contrast": 0.0,
    "shift": 0,
    "chatter": 0,
    "specks": 0,
    "holes": 0,
    "decontaminate": False,
}


class RotoDocError(ValueError):
    pass


def _sessionCuts(mattes) -> list:
    """The scene cuts a tracked folder's session.json records, or none."""
    if not mattes:
        return []
    try:
        with open(os.path.join(mattes, "session.json"), encoding="utf-8") as handle:
            cuts = json.load(handle).get("cuts") or []
    except OSError, ValueError, AttributeError:
        return []
    return [c for c in cuts if isinstance(c, int) and c >= 0]


def matteAlpha(image: np.ndarray) -> np.ndarray:
    """A matte image as float32 coverage in 0..1, white being the object.

    Grey is read as is, colour by its luma, and an alpha channel multiplies in,
    so a matte drawn on a transparent background works too.
    """
    scale = _INTEGER_FULL_SCALE.get(image.dtype.name)
    alpha = None
    if image.ndim == 3 and image.shape[2] == 4:
        alpha = image[:, :, 3]
        image = image[:, :, :3]
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coverage = image.astype(np.float32)
    if scale is not None:
        coverage /= scale
    if alpha is not None:
        coverage *= alpha.astype(np.float32) / (scale or 1.0)
    return np.clip(np.nan_to_num(coverage, nan=0.0), 0.0, 1.0)


def _number(value, default, low, high, integer=False):
    try:
        number = float(value)
    except TypeError, ValueError:
        return default
    if not np.isfinite(number):
        return default
    number = min(high, max(low, number))
    return int(round(number)) if integer else number


def sanitizeRefine(raw) -> dict:
    raw = raw if isinstance(raw, dict) else {}
    return {
        "feather": _number(raw.get("feather"), 0.0, 0.0, 50.0),
        "contrast": _number(raw.get("contrast"), 0.0, 0.0, 100.0),
        "shift": _number(raw.get("shift"), 0, -50, 50, integer=True),
        "chatter": _number(raw.get("chatter"), 0, 0, _MAX_CHATTER, integer=True),
        "specks": _number(raw.get("specks"), 0, 0, 1_000_000, integer=True),
        "holes": _number(raw.get("holes"), 0, 0, 1_000_000, integer=True),
        "decontaminate": raw.get("decontaminate") is True,
    }


def _points(raw, arity: int) -> list | None:
    if not isinstance(raw, list) or not raw:
        return None
    points = []
    for point in raw:
        if not isinstance(point, (list, tuple)) or len(point) not in (2, 6):
            return None
        try:
            values = [float(v) for v in point]
        except TypeError, ValueError:
            return None
        if not all(np.isfinite(values)):
            return None
        if len(values) != arity:
            return None
        points.append(values)
    return points


def sanitizeShape(raw) -> dict | None:
    """A shape from a document, or None when it cannot be drawn."""
    if not isinstance(raw, dict):
        return None
    kind = raw.get("kind")
    mode = raw.get("mode", "add")
    if kind not in ("polygon", "stroke") or mode not in ("add", "subtract"):
        return None
    try:
        start, end = int(raw["start"]), int(raw["end"])
    except KeyError, TypeError, ValueError:
        return None
    if start < 0 or end < start:
        return None
    keys = []
    for key in raw.get("keys") or []:
        if not isinstance(key, dict):
            return None
        try:
            frame = int(key["frame"])
        except KeyError, TypeError, ValueError:
            return None
        points = _points(key.get("points"), 6 if kind == "polygon" else 2)
        if points is None:
            return None
        keys.append({"frame": frame, "points": points})
    if not keys:
        return None
    keys.sort(key=lambda k: k["frame"])
    # Keys interpolate point by point, so every key has the shape's point count. A
    # tracked stroke has a key per frame.
    if any(len(k["points"]) != len(keys[0]["points"]) for k in keys):
        return None
    if kind == "polygon" and len(keys[0]["points"]) < 3:
        return None
    shape = {"kind": kind, "mode": mode, "start": start, "end": end, "keys": keys}
    if kind == "stroke":
        shape["radius"] = _number(raw.get("radius"), 8.0, 0.5, 2000.0)
    else:
        shape["feather"] = _number(raw.get("feather"), 0.0, 0.0, 100.0)
    return shape


def loadRotoDoc(path: str) -> dict:
    """What ``--segment_mask`` points at, as a document. Raises RotoDocError."""
    if os.path.isdir(path):
        return {
            "mattes": path,
            "width": 0,
            "height": 0,
            "shapes": [],
            "refine": sanitizeRefine({}),
        }
    try:
        with open(path, "rb") as handle:
            raw = json.loads(handle.read())
    except (OSError, ValueError) as error:
        raise RotoDocError(f"--segment_mask document cannot be read: {error}") from None
    return parseRotoDoc(raw, os.path.dirname(os.path.abspath(path)))


def parseRotoDoc(raw, baseDir: str = "") -> dict:
    if not isinstance(raw, dict) or raw.get("version") != 1:
        raise RotoDocError("--segment_mask document is not a version 1 roto document.")
    mattes = raw.get("mattes")
    if mattes is not None:
        if not isinstance(mattes, str) or not mattes:
            raise RotoDocError("--segment_mask document names no matte folder.")
        if baseDir and not os.path.isabs(mattes):
            mattes = os.path.join(baseDir, mattes)
    shapesRaw = raw.get("shapes") or []
    if not isinstance(shapesRaw, list) or len(shapesRaw) > _MAX_SHAPES:
        raise RotoDocError("--segment_mask document has an invalid shape list.")
    shapes = []
    for entry in shapesRaw:
        shape = sanitizeShape(entry)
        if shape is None:
            raise RotoDocError(
                "--segment_mask document has a shape that cannot be drawn."
            )
        shapes.append(shape)
    width = _number(raw.get("width"), 0, 0, 1 << 16, integer=True)
    height = _number(raw.get("height"), 0, 0, 1 << 16, integer=True)
    if shapes and (width <= 0 or height <= 0):
        raise RotoDocError("--segment_mask document has shapes but no frame size.")
    return {
        "mattes": mattes,
        "width": width,
        "height": height,
        "shapes": shapes,
        "refine": sanitizeRefine(raw.get("refine")),
    }


def _bezier(p0, p1, p2, p3, steps=_BEZIER_STEPS):
    t = np.linspace(0.0, 1.0, steps, endpoint=False)[:, None]
    u = 1.0 - t
    return (
        u**3 * np.asarray(p0)
        + 3 * u**2 * t * np.asarray(p1)
        + 3 * u * t**2 * np.asarray(p2)
        + t**3 * np.asarray(p3)
    )


def flattenPolygon(points: list) -> np.ndarray:
    """A closed Bezier outline as polygon vertices. Straight where the handles sit on
    their vertex, so a plain polygon stays exactly its corners."""
    out = []
    count = len(points)
    for i in range(count):
        x, y, _, _, ox, oy = points[i]
        nx, ny, ix, iy, _, _ = points[(i + 1) % count]
        if (ox, oy) == (x, y) and (ix, iy) == (nx, ny):
            out.append([[x, y]])
        else:
            out.append(_bezier((x, y), (ox, oy), (ix, iy), (nx, ny)))
    return np.concatenate(out, axis=0).astype(np.float64)


def shapePointsAt(shape: dict, frame: int) -> list | None:
    """The shape's points on ``frame``, interpolated between its keys, or None when the
    frame is outside its span."""
    if frame < shape["start"] or frame > shape["end"]:
        return None
    keys = shape["keys"]
    if frame <= keys[0]["frame"]:
        return keys[0]["points"]
    if frame >= keys[-1]["frame"]:
        return keys[-1]["points"]
    for before, after in itertools.pairwise(keys):
        if before["frame"] <= frame <= after["frame"]:
            span = after["frame"] - before["frame"]
            t = 0.0 if span == 0 else (frame - before["frame"]) / span
            a = np.asarray(before["points"], dtype=np.float64)
            b = np.asarray(after["points"], dtype=np.float64)
            return (a + (b - a) * t).tolist()
    return keys[-1]["points"]


def rasterizeShape(
    shape: dict, frame: int, height: int, width: int, scaleX=1.0, scaleY=1.0
):
    """Coverage of one shape on ``frame`` as float32 0..1, or None outside its span."""
    points = shapePointsAt(shape, frame)
    if points is None:
        return None
    canvas = np.zeros((height, width), dtype=np.uint8)
    factor = 1 << _SUBPIXEL_SHIFT
    scale = np.array([scaleX, scaleY])
    # Far past the frame changes nothing drawn on it, and the fixed-point coordinates
    # cv2 takes overflow int32 a long way out.
    reach = 4.0 * max(height, width)
    limit = np.array([[-reach, -reach], [width + reach, height + reach]])
    if shape["kind"] == "polygon":
        outline = np.clip(flattenPolygon(points) * scale, limit[0], limit[1])
        fixed = np.rint(outline * factor).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(canvas, [fixed], 255, cv2.LINE_AA, _SUBPIXEL_SHIFT)
        coverage = canvas.astype(np.float32) / 255.0
        feather = shape.get("feather", 0.0) * (scaleX + scaleY) / 2
        if feather > 0:
            coverage = cv2.GaussianBlur(coverage, (0, 0), feather / 2)
        return coverage
    radius = min(shape["radius"] * (scaleX + scaleY) / 2, reach)
    path = np.clip(np.asarray(points, dtype=np.float64) * scale, limit[0], limit[1])
    fixed = np.rint(path * factor).astype(np.int32)
    thickness = max(1, int(round(radius * 2)))
    if len(fixed) == 1:
        cv2.circle(
            canvas,
            tuple(fixed[0]),
            int(round(radius * factor)),
            255,
            -1,
            cv2.LINE_AA,
            _SUBPIXEL_SHIFT,
        )
    else:
        cv2.polylines(
            canvas,
            [fixed.reshape(-1, 1, 2)],
            False,
            255,
            thickness,
            cv2.LINE_AA,
            _SUBPIXEL_SHIFT,
        )
        for point in (fixed[0], fixed[-1]):
            cv2.circle(
                canvas,
                tuple(point),
                int(round(radius * factor)),
                255,
                -1,
                cv2.LINE_AA,
                _SUBPIXEL_SHIFT,
            )
    return canvas.astype(np.float32) / 255.0


def applyShapes(
    alpha: np.ndarray, shapes: list, frame: int, docSize=(0, 0)
) -> np.ndarray:
    """``alpha`` with every shape that covers ``frame`` added or cut, in order."""
    height, width = alpha.shape
    docW, docH = docSize
    scaleX = width / docW if docW else 1.0
    scaleY = height / docH if docH else 1.0
    out = alpha
    for shape in shapes:
        coverage = rasterizeShape(shape, frame, height, width, scaleX, scaleY)
        if coverage is None:
            continue
        if out is alpha:
            out = alpha.copy()
        if shape["mode"] == "add":
            np.maximum(out, coverage, out=out)
        else:
            out *= 1.0 - coverage
    return out


def _ellipse(radius: int) -> np.ndarray:
    size = 2 * radius + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _removeSmall(binary: np.ndarray, minArea: int, borderKeeps: bool) -> np.ndarray:
    """Components of ``binary`` smaller than ``minArea``, as a mask. With ``borderKeeps``
    a component touching the frame edge is never small (a hole open to the edge is
    background, not a hole)."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8
    )
    small = np.zeros(count, dtype=bool)
    height, width = binary.shape
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area >= minArea:
            continue
        if borderKeeps and (x == 0 or y == 0 or x + w >= width or y + h >= height):
            continue
        small[label] = True
    return small[labels]


def cleanUp(alpha: np.ndarray, specks: int, holes: int) -> np.ndarray:
    if specks <= 0 and holes <= 0:
        return alpha
    out = alpha.copy()
    if specks > 0:
        out[_removeSmall(out >= 0.5, specks, borderKeeps=False)] = 0.0
    if holes > 0:
        out[_removeSmall(out < 0.5, holes, borderKeeps=True)] = 1.0
    return out


def refineAlpha(
    alpha: np.ndarray, refine: dict, neighbours: list | None = None
) -> np.ndarray:
    """Refine Edge: chatter, then shift edge, feather and contrast. Specks and holes are
    cleaned before this, per frame, so the neighbours are compared clean."""
    out = alpha
    chatter = refine.get("chatter", 0)
    if chatter and neighbours:
        stack = np.stack([out, *neighbours])
        binary = out >= 0.5
        edge = (
            cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_GRADIENT, _ellipse(1))
            > 0
        )
        band = cv2.dilate(edge.astype(np.uint8), _ellipse(max(1, 2 * chatter))) > 0
        smoothed = stack.mean(axis=0)
        out = np.where(band, smoothed, out).astype(np.float32)
    shift = refine.get("shift", 0)
    if shift > 0:
        out = cv2.dilate(out, _ellipse(shift))
    elif shift < 0:
        out = cv2.erode(out, _ellipse(-shift))
    feather = refine.get("feather", 0.0)
    if feather > 0:
        out = cv2.GaussianBlur(out, (0, 0), feather / 2)
    # After Feather, as in After Effects: it hardens a feathered edge back toward its
    # middle, so the two together give a smooth outline with a firm edge.
    contrast = refine.get("contrast", 0.0)
    if contrast > 0:
        gain = 1.0 + contrast / 10.0
        out = np.clip((out - 0.5) * gain + 0.5, 0.0, 1.0)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def decontaminate(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Pull partly transparent edge pixels toward the colour of the solid object next to
    them, so the old background does not fringe the cutout. ``rgb`` is float HxWx3."""
    solid = (alpha > 0.98).astype(np.float32)
    edge = (alpha > 0.0) & (alpha < 0.98)
    if not edge.any() or not solid.any():
        return rgb
    sigma = max(2.0, min(rgb.shape[:2]) / 200)
    weight = cv2.GaussianBlur(solid, (0, 0), sigma)
    colour = cv2.GaussianBlur(rgb * solid[..., None], (0, 0), sigma)
    usable = edge & (weight > 1e-3)
    out = rgb.copy()
    estimate = colour[usable] / weight[usable][:, None]
    a = alpha[usable][:, None]
    out[usable] = estimate * (1.0 - a) + rgb[usable] * a
    return out


class RotoRenderer:
    """The final alpha of any frame of a roto document, with a small cache, for the
    render (in order) and for previews (anywhere)."""

    def __init__(self, doc: dict, frameOffset: int = 0):
        self.doc = doc
        self.frameOffset = frameOffset
        mattes = doc.get("mattes")
        self.frameFiles = (
            listMaskFrames(mattes) if mattes and os.path.isdir(mattes) else {}
        )
        # Scene cuts the track found (session.json): reduce chatter stays inside a shot.
        self.cuts = sorted(doc.get("cuts") or _sessionCuts(mattes))
        self.refine = doc.get("refine") or sanitizeRefine({})
        self._cache: OrderedDict = OrderedDict()
        self.warnedMissing = False

    def baseAlpha(self, index: int, height: int, width: int) -> np.ndarray:
        path = self.frameFiles.get(index)
        image = readMaskImage(path) if path is not None else None
        if image is None:
            if path is not None and not self.warnedMissing:
                self.warnedMissing = True
                logging.warning(
                    f"Matte for source frame {index} cannot be read; it is left empty."
                )
            return np.zeros((height, width), dtype=np.float32)
        alpha = matteAlpha(image)
        if alpha.shape != (height, width):
            alpha = cv2.resize(alpha, (width, height), interpolation=cv2.INTER_LINEAR)
        return alpha

    def cleanAlpha(self, index: int, height: int, width: int, base=None) -> np.ndarray:
        """Mattes plus shapes, specks and holes cleaned: what chatter compares."""
        key = (index, height, width)
        if base is None and key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        alpha = self.baseAlpha(index, height, width) if base is None else base
        alpha = applyShapes(
            alpha,
            self.doc.get("shapes") or [],
            index,
            (self.doc.get("width", 0), self.doc.get("height", 0)),
        )
        alpha = cleanUp(
            alpha, self.refine.get("specks", 0), self.refine.get("holes", 0)
        )
        if base is None:
            self._cache[key] = alpha
            while len(self._cache) > _CACHE_FRAMES:
                self._cache.popitem(last=False)
        return alpha

    def alpha(self, index: int, height: int, width: int, base=None) -> np.ndarray:
        """Final alpha of source frame ``index``. ``base`` replaces the matte of that frame
        only (a preview of clicks not yet tracked)."""
        clean = self.cleanAlpha(index, height, width, base)
        chatter = self.refine.get("chatter", 0)
        neighbours = None
        if chatter and self.frameFiles:
            first, last = self._shotOf(index)
            neighbours = [
                self.cleanAlpha(other, height, width)
                for other in range(
                    max(first, index - chatter), min(last, index + chatter) + 1
                )
                # A neighbour outside the tracked frames, or across a cut, is not the
                # object's edge moving. Without mattes there is no edge to steady.
                if other != index and other in self.frameFiles
            ]
        return refineAlpha(clean, self.refine, neighbours)

    def _shotOf(self, index: int) -> tuple[int, int]:
        """The first and last frame of the shot ``index`` is in (cuts start a shot)."""
        first, last = 0, 1 << 30
        for cut in self.cuts:
            if cut <= index:
                first = cut
            elif cut - 1 < last:
                last = cut - 1
                break
        return first, last

    def framesCovered(self, start: int, end: int | None) -> int:
        """How many source frames in ``[start, end)`` have a matte or a shape."""
        stop = end if end is not None else max([start, *self.frameFiles.keys()]) + 1
        if stop <= start:
            return 0
        covered = {f for f in self.frameFiles if start <= f < stop}
        for shape in self.doc.get("shapes") or []:
            covered.update(
                range(max(start, shape["start"]), min(stop, shape["end"] + 1))
            )
        return len(covered)

    def alphaForPosition(self, position: int, height: int, width: int) -> np.ndarray:
        return self.alpha(position + self.frameOffset, height, width)
