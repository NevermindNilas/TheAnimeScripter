import logging
import os

import cv2
import torch


class ProtectionMask:
    """A user-painted protection mask, shared by motion blur and interpolation.

    Intended input: a transparent PNG where the user paints protected regions
    with opaque dark pixels on a transparent background. Everything else
    (transparent or bright) is processed normally.

    Derivation of per-pixel protection:
        RGBA: protection = alpha * (1 - luma)
        RGB : protection = 1 - luma          (assume fully opaque)
        Gray: protection = 1 - value         (dark = protected)

    The cached tensor is the *process weight*, 1 - protection, so the blend is:
        out = processed * weight + pristine * (1 - weight)

    Weight tensors are built lazily on first use and cached per
    (height, width, device, dtype). The interpolation path needs this because a
    frame reaching the mask may be pre- or post-upscale depending on
    ``--interpolate_first``, and DirectML/NCNN backends hand back CPU tensors
    while CUDA backends keep them on the GPU.
    """

    __slots__ = ("path", "_cache", "_logged")

    def __init__(self, path: str | None):
        self.path = path or ""
        self._cache: dict[tuple[int, int, str, torch.dtype], torch.Tensor] = {}
        self._logged = False

    @property
    def enabled(self) -> bool:
        return bool(self.path)

    def _read(self):
        if not os.path.isfile(self.path):
            logging.warning(f"Mask not found: {self.path}. Ignoring.")
            return None

        maskImg = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
        if maskImg is None:
            logging.warning(f"Failed to read mask: {self.path}. Ignoring.")
        return maskImg

    def _build(self, height: int, width: int, device, dtype) -> torch.Tensor | None:
        maskImg = self._read()
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

        norm = 65535.0 if maskImg.dtype == "uint16" else 255.0

        if maskImg.ndim == 3 and maskImg.shape[2] == 4:
            bgr = maskImg[:, :, :3]
            alpha = maskImg[:, :, 3]
            luma = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            lumaT = torch.from_numpy(luma).to(device=device, dtype=dtype).div_(norm)
            alphaT = torch.from_numpy(alpha).to(device=device, dtype=dtype).div_(norm)
            protection = alphaT * (1.0 - lumaT)
            channels = "RGBA"
        elif maskImg.ndim == 3:
            luma = cv2.cvtColor(maskImg, cv2.COLOR_BGR2GRAY)
            lumaT = torch.from_numpy(luma).to(device=device, dtype=dtype).div_(norm)
            protection = 1.0 - lumaT
            channels = "RGB (no alpha, assuming opaque)"
        else:
            gray = torch.from_numpy(maskImg).to(device=device, dtype=dtype).div_(norm)
            protection = 1.0 - gray
            channels = "grayscale"

        protection = protection.clamp_(0.0, 1.0)
        weightTensor = (1.0 - protection).view(1, 1, height, width)

        if not self._logged:
            logging.info(f"Loaded mask: {self.path} ({channels})")
            self._logged = True
        return weightTensor

    def weightFor(self, frame: torch.Tensor) -> torch.Tensor | None:
        """Return the [1, 1, H, W] process-weight tensor matching ``frame``."""
        if not self.path:
            return None

        height, width = frame.shape[-2], frame.shape[-1]
        key = (height, width, str(frame.device), frame.dtype)
        if key not in self._cache:
            self._cache[key] = self._build(height, width, frame.device, frame.dtype)
        return self._cache[key]

    def apply(self, processed: torch.Tensor, pristine: torch.Tensor) -> torch.Tensor:
        """Blend ``processed`` toward ``pristine`` inside protected regions.

        Returns ``processed`` unchanged when the mask is disabled or unreadable.
        Out-of-place on purpose: ``processed`` may be a CUDA-graph output buffer
        or a decoded frame still owned by another stage.

        ``lerp`` is one fused kernel rather than the three of
        ``processed * w + pristine * (1 - w)`` (2.5x faster at 1080p fp16) and is
        still exact at both boundaries: w=1 returns ``processed`` bit-for-bit and
        w=0 returns ``pristine`` bit-for-bit, so unmasked pixels are untouched.
        """
        weight = self.weightFor(processed)
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

    __slots__ = ("mask", "sink", "pristine")

    def __init__(self, mask: ProtectionMask):
        self.mask = mask
        self.sink = None
        self.pristine = None

    def bind(self, sink, pristine: torch.Tensor | None) -> MaskedSink:
        self.sink = sink
        self.pristine = pristine
        return self

    def put(self, frame: torch.Tensor) -> None:
        if self.pristine is None:
            self.sink.put(frame)
            return

        blended = self.mask.apply(frame, self.pristine)

        # Everything entering a write buffer must already be materialized: the
        # writer thread copies it to pinned memory on its own private stream
        # (ffmpegSettings.WriteBuffer.transferStream / NeluxWriteBuffer.CudaStream)
        # and never waits on ours, so it can read the tensor before these blend
        # kernels have run. The interpolators satisfy that contract by syncing
        # normStream before they hand a frame over; MotionBlurPipeline does the
        # same around its blend. Without this, output is non-deterministic
        # run-to-run (reproduced on rife4.6 with a fully transparent mask, whose
        # blend is a mathematical no-op).
        if blended.is_cuda:
            torch.cuda.current_stream().synchronize()

        self.sink.put(blended)
