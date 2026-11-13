import logging
import os
import re
from collections import OrderedDict
from typing import Self

import cv2
import numpy as np
import torch

_FRAME_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
_FRAME_NAME = re.compile(r"\d+")
# A gap reads two frames and a motion blur window three, at up to two sizes when
# the chain has more than one masked stage.
_SEQUENCE_CACHE_FRAMES = 8
# What each integer image depth counts up to. A float image is already 0..1.
_INTEGER_FULL_SCALE = {"uint8": 255.0, "uint16": 65535.0}


def readMaskImage(path: str):
    """``cv2.imread(path, IMREAD_UNCHANGED)`` that also opens non-ASCII Windows paths.

    ``cv2.imread`` cannot, and a mask folder lives under the user's home directory.
    Returns None for anything that is not a readable image, including a depth the
    mask cannot use (signed or 32-bit integer). A 64-bit float comes back as 32-bit,
    which is all ``cv2.cvtColor`` takes.
    """
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    except OSError, cv2.error:
        return None
    if image is None:
        return None
    if image.dtype == np.float64:
        image = image.astype(np.float32)
    if image.dtype.name not in ("uint8", "uint16", "float32"):
        return None
    return image


def listMaskFrames(directory: str) -> dict[int, str]:
    """The per-frame masks in ``directory``: source frame index -> file path.

    A file counts when its stem is all digits (``000123.png`` is source frame
    123) and its suffix is an image format ``cv2`` reads. The index is the frame's
    position in the untrimmed input, so a mask set keeps its meaning under any
    ``--inpoint``.
    """
    frames: dict[int, str] = {}
    with os.scandir(directory) as entries:
        for entry in entries:
            stem, suffix = os.path.splitext(entry.name)
            if (
                suffix.lower() in _FRAME_IMAGE_SUFFIXES
                and _FRAME_NAME.fullmatch(stem)
                and entry.is_file()
            ):
                frames[int(stem)] = entry.path
    return frames


class ProtectionMask:
    """A user-painted protection mask, shared by motion blur and interpolation.

    ``path`` is one image, applied to every frame, or a directory holding one
    image per source frame (see :func:`listMaskFrames`), which is what a tracked
    object produces. A frame with no image in the directory has no protection.

    Intended input: a transparent PNG where the user paints protected regions
    with opaque dark pixels on a transparent background. Everything else
    (transparent or bright) is processed normally.

    Derivation of per-pixel protection:
        RGBA: protection = alpha * (1 - luma)
        RGB : protection = 1 - luma          (assume fully opaque)
        Gray: protection = 1 - value         (dark = protected)

    8-bit and 16-bit images are scaled by their full range. A float image is taken
    as already 0..1: a value outside that is clamped to it (HDR white is white), and
    NaN reads as 0, so a NaN alpha is transparent and a NaN luma is black.

    The cached tensor is the *process weight*, 1 - protection, so the blend is:
        out = processed * weight + pristine * (1 - weight)

    Weight tensors are built lazily on first use and cached per
    (height, width, device, dtype). The interpolation path needs this because a
    frame reaching the mask may be pre- or post-upscale depending on where
    ``--upscale`` and ``--interpolate`` sit in the command line, and
    DirectML/NCNN backends hand back CPU tensors while CUDA backends keep them
    on the GPU.
    """

    __slots__ = (
        "path",
        "frameOffset",
        "_frames",
        "_cache",
        "_frameCache",
        "_blendKey",
        "_blendWeight",
        "_logged",
        "_warnedMissing",
    )

    def __init__(self, path: str | None, frameOffset: int = 0):
        self.path = path or ""
        self.frameOffset = max(0, int(frameOffset))
        self._frames = listMaskFrames(self.path) if os.path.isdir(self.path) else None
        self._cache: dict[tuple[int, int, str, torch.dtype], torch.Tensor] = {}
        self._frameCache: OrderedDict = OrderedDict()
        self._blendKey = None
        self._blendWeight = None
        self._logged = False
        self._warnedMissing = False

    @classmethod
    def forRun(cls, path: str | None, fps, inpoint, outpoint) -> Self:
        """A mask for a run, aligned to the frames ``--inpoint`` skips."""
        offset = 0
        if path and os.path.isdir(path):
            from src.io.getVideoMetadata import trimFrameRange

            offset = trimFrameRange(fps, inpoint, outpoint)[0]
        return cls(path, frameOffset=offset)

    @property
    def enabled(self) -> bool:
        return bool(self.path)

    @property
    def isSequence(self) -> bool:
        return self._frames is not None

    @staticmethod
    def _read(path: str):
        if not os.path.isfile(path):
            logging.warning(f"Mask not found: {path}. Ignoring.")
            return None

        maskImg = readMaskImage(path)
        if maskImg is None:
            logging.warning(f"Failed to read mask: {path}. Ignoring.")
        return maskImg

    def _build(
        self, path: str, height: int, width: int, device, dtype
    ) -> torch.Tensor | None:
        maskImg = self._read(path)
        if maskImg is None:
            return None

        if maskImg.shape[:2] != (height, width):
            logging.info(
                f"Resizing mask from {maskImg.shape[1]}x{maskImg.shape[0]} "
                f"to {width}x{height}"
            )
            maskImg = cv2.resize(
                maskImg, (width, height), interpolation=cv2.INTER_LINEAR
            )

        norm = _INTEGER_FULL_SCALE.get(maskImg.dtype.name)

        def normalized(channel):
            # Work in fp32, then cast: a 16-bit value above 65504 is inf in
            # fp16, so casting first made every opaque pixel of a 16-bit mask
            # fully protected (and transparent white NaN) on half frames. An
            # 8-bit value is exact in fp16 and the quotient rounds the same.
            tensor = torch.from_numpy(channel).to(device=device, dtype=torch.float32)
            if norm is None:
                # Per channel, so an out-of-range alpha cannot scale a good luma.
                tensor = torch.nan_to_num(
                    tensor, nan=0.0, posinf=1.0, neginf=0.0
                ).clamp_(0.0, 1.0)
            else:
                tensor.div_(norm)
            return tensor.to(dtype)

        if maskImg.ndim == 3 and maskImg.shape[2] == 4:
            bgr = maskImg[:, :, :3]
            alpha = maskImg[:, :, 3]
            luma = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            lumaT = normalized(luma)
            alphaT = normalized(alpha)
            protection = alphaT * (1.0 - lumaT)
            channels = "RGBA"
        elif maskImg.ndim == 3:
            luma = cv2.cvtColor(maskImg, cv2.COLOR_BGR2GRAY)
            lumaT = normalized(luma)
            protection = 1.0 - lumaT
            channels = "RGB (no alpha, assuming opaque)"
        else:
            gray = normalized(maskImg)
            protection = 1.0 - gray
            channels = "grayscale"

        protection = protection.clamp_(0.0, 1.0)
        weightTensor = (1.0 - protection).view(1, 1, height, width)

        if not self._logged:
            logging.info(f"Loaded mask: {self.path} ({channels})")
            self._logged = True
        return weightTensor

    def _frameWeight(self, position: int, height, width, device, dtype):
        """The weight of one decoded frame (0-based), or None if it has no mask."""
        index = position + self.frameOffset
        path = self._frames.get(index)
        if path is None:
            if not self._warnedMissing:
                self._warnedMissing = True
                logging.warning(
                    f"Mask folder has no image for source frame {index}; frames "
                    f"without one are not protected: {self.path}"
                )
            return None

        key = (index, height, width, str(device), dtype)
        cache = self._frameCache
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        weight = self._build(path, height, width, device, dtype)
        cache[key] = weight
        if len(cache) > _SEQUENCE_CACHE_FRAMES:
            cache.popitem(last=False)
        return weight

    def _sequenceWeight(self, positions, height, width, device, dtype):
        """Process weight protecting everywhere the object is at any of ``positions``.

        A generated frame sits between its source frames, so a moving object has
        to be protected at both ends of the gap: masked at one end only, the
        morphed copy of it at the other end shows through as a ghost.
        """
        key = (positions, height, width, str(device), dtype)
        if key == self._blendKey:
            return self._blendWeight

        weights = [
            weight
            for position in dict.fromkeys(positions)
            if (weight := self._frameWeight(position, height, width, device, dtype))
            is not None
        ]
        combined = None
        for weight in weights:
            combined = weight if combined is None else torch.minimum(combined, weight)
        self._blendKey = key
        self._blendWeight = combined
        return combined

    def weightFor(
        self, frame: torch.Tensor, positions: tuple[int, ...] | None = None
    ) -> torch.Tensor | None:
        """Return the [1, 1, H, W] process-weight tensor matching ``frame``.

        ``positions`` are the 0-based decoded frames the generated frame depends
        on; a single-image mask ignores them.
        """
        if not self.path:
            return None

        height, width = frame.shape[-2], frame.shape[-1]
        if self._frames is not None:
            if not positions:
                return None
            return self._sequenceWeight(
                positions, height, width, frame.device, frame.dtype
            )

        key = (height, width, str(frame.device), frame.dtype)
        if key not in self._cache:
            self._cache[key] = self._build(
                self.path, height, width, frame.device, frame.dtype
            )
        return self._cache[key]

    def apply(
        self,
        processed: torch.Tensor,
        pristine: torch.Tensor,
        positions: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        """Blend ``processed`` toward ``pristine`` inside protected regions.

        Returns ``processed`` unchanged when the mask is disabled or unreadable.
        Out-of-place on purpose: ``processed`` may be a CUDA-graph output buffer
        or a decoded frame still owned by another stage.

        ``lerp`` is one fused kernel rather than the three of
        ``processed * w + pristine * (1 - w)`` (2.5x faster at 1080p fp16) and is
        still exact at both boundaries: w=1 returns ``processed`` bit-for-bit and
        w=0 returns ``pristine`` bit-for-bit, so unmasked pixels are untouched.
        """
        weight = self.weightFor(processed, positions)
        if weight is None:
            return processed

        if pristine.device != processed.device or pristine.dtype != processed.dtype:
            pristine = pristine.to(device=processed.device, dtype=processed.dtype)

        return torch.lerp(pristine, processed, weight)


class MaskedSink:
    """Sink proxy that masks every interpolated frame an interpolator emits.

    The interpolation drivers push their intermediate frames straight into a
    sink (``interpQueue`` or ``writeBuffer``) rather than returning them, so the
    mask is applied here instead of inside each backend. Bound per source frame:
    ``pristine`` is the segment's anchor frame, whose protected pixels replace
    the morphed ones.
    """

    __slots__ = ("mask", "sink", "pristine", "positions")

    def __init__(self, mask: ProtectionMask):
        self.mask = mask
        self.sink = None
        self.pristine = None
        self.positions = None

    def bind(
        self,
        sink,
        pristine: torch.Tensor | None,
        positions: tuple[int, ...] | None = None,
    ) -> Self:
        self.sink = sink
        self.pristine = pristine
        self.positions = positions
        return self

    def put(self, frame: torch.Tensor) -> None:
        if self.pristine is None:
            self.sink.put(frame)
            return

        blended = self.mask.apply(frame, self.pristine, self.positions)

        # Everything entering a write buffer must already be materialized: the
        # writer thread copies it to pinned memory on its own private stream
        # (ffmpegSettings.NeluxWriteBuffer.CudaStream)
        # and never waits on ours, so it can read the tensor before these blend
        # kernels have run. The interpolators satisfy that contract by syncing
        # normStream before they hand a frame over; MotionBlurPipeline does the
        # same around its blend. Without this, output is non-deterministic
        # run-to-run (reproduced on rife4.6 with a fully transparent mask, whose
        # blend is a mathematical no-op).
        if blended.is_cuda:
            torch.cuda.current_stream().synchronize()

        self.sink.put(blended)
