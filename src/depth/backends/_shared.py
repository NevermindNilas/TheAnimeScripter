import math
import os

os.environ.setdefault("DA3_LOG_LEVEL", "ERROR")

import logging

import torch

from src.infra.cudaAllocator import expandableSegments  # noqa: F401  (re-export)
from src.infra.isCudaInit import CudaChecker

checker = CudaChecker()


class DepthRunOutcome:
    """Makes a depth run report its own outcome honestly.

    `--depth` goes through `_selectProcessingMethod`, not `start()`, so
    `main.py`'s `_notifyAdobe` never runs for it: the After Effects panel sat
    on the last progress string forever, and `main.py:didFail()` judged the run
    purely by the size of a possibly half-written file. Every `except` in these
    backends also logged and continued, so a failed inference batch left the
    progress bar to reach 100%% on a truncated output. `--segment`,
    `--obj_detect` and `--stabilize` already carry this contract; this is the
    same one, shared because eleven depth classes need it.
    """

    processingError: Exception | None = None

    def recordFailure(self, error: Exception) -> None:
        """Remember the first failure; later ones are usually its fallout."""
        if self.processingError is None:
            self.processingError = error

    def guardedProcess(self, processFn=None) -> None:
        """Run the frame loop so a raise cannot hang the process.

        The reader blocks on `put()` into a full decode queue and the writer
        spins until it sees its `None` sentinel, so a `process()` that escapes
        before its own `close()` leaves `ThreadPoolExecutor.__exit__` joining
        forever -- with the original exception buried in a discarded future.
        Same contract `main.py:process` already has in its `finally`.
        """
        from src.io.ffmpegSettings import closeWriterAndDrainReader
        from src.io.runOutcome import truncatedDecodeError

        try:
            (processFn or self.process)()
        except Exception as e:
            self.recordFailure(e)
            logging.exception(f"Depth frame loop failed, {e}")
        finally:
            # The decoder signals a mid-stream death only by putting its
            # end-of-stream sentinel, which reads like a clean EOF, so without
            # this a truncated decode wrote a short depth map and exited 0.
            decodeError = truncatedDecodeError(
                self.readBuffer, self.totalFrames, self.writeBuffer
            )
            if decodeError is not None:
                self.recordFailure(decodeError)
            closeWriterAndDrainReader(self.writeBuffer, self.readBuffer)

    def reportOutcome(self) -> None:
        """Settle the run's outcome and tell the AE panel. Safe to call more
        than once; every backend calls it after its threads have finished."""
        from src.constants import ADOBE
        from src.io.runOutcome import recordWriterFailure

        recordWriterFailure(self)
        if not ADOBE:
            return
        from src.server.aeComms import reportTerminalStatus

        reportTerminalStatus(self.processingError, self.output, self.benchmark)


class _Finished:
    """A result whose producing GPU work is queued up to ``event``.

    Stands where a ``Future`` would in ``OrderedFrameSink``: ``result()``
    blocks the host only until that event, so the stream keeps running the work
    queued after it (the next forward) while the frame is handed on.
    """

    def __init__(self, value, event=None, check=None):
        self._value = value
        self._event = event
        self._check = check

    def result(self):
        if self._event is not None:
            self._event.synchronize()
        if self._check is not None:
            self._check()
        return self._value


class _Failed:
    def __init__(self, error: Exception):
        self._error = error

    def result(self):
        raise self._error


def _readyEvent(value):
    """An event marking the end of the work queued for ``value``'s device, or None.

    Only CUDA (and ROCm's HIP, which torch reports as "cuda") queues work; a CPU
    tensor is finished when it exists.
    """
    device = getattr(value, "device", None)
    if device is None or device.type != "cuda":
        return None
    event = torch.cuda.Event()
    event.record(torch.cuda.current_stream(device))
    return event


class OrderedFrameSink:
    """Depth frames written in decode order, a few frames behind the GPU.

    The depth post-process (1/depth, a two-percentile stretch, the chunk
    alignment of the streaming methods) runs on the model's device, queued
    behind the forward that produced it. Writing a frame the moment it is
    queued would have to wait for the GPU and leave it idle until the next
    forward is issued, so each frame is held here, with an event marking the end
    of its work, until ``maxPending`` newer ones arrive: by then the next
    forward is queued and the wait overlaps it. The writer copies on its own
    stream and never waits on ours, so a frame is only written once its event
    has passed.

    Every result, and every failure, is delivered on the caller's thread in
    submission order. With ``recordFailure`` a failure is recorded and the run
    goes on; with None it is raised from the call that delivers it, and nothing
    queued after it is written.
    """

    def __init__(self, write, recordFailure, progress=None, maxPending=3):
        from collections import deque

        self._write = write
        self._recordFailure = recordFailure
        self._progress = progress
        self._maxPending = max(1, maxPending)
        self._pending = deque()
        self._raised = False

    def submit(self, values, check=None) -> None:
        """Queue frames whose work is already issued on the current stream.

        ``check`` runs once the first of them is ready, before it is written; a
        raise there stands for a failure of that frame.
        """
        values = list(values)
        if not values:
            return
        event = _readyEvent(values[0])
        for index, value in enumerate(values):
            self._push(_Finished(value, event, check if index == 0 else None), 1)

    def fail(self, error: Exception, frames: int) -> None:
        """Queue a failure that stands for ``frames`` decoded frames."""
        self._push(_Failed(error), frames)

    def _push(self, pending, frames: int) -> None:
        self._pending.append((pending, frames))
        while len(self._pending) > self._maxPending:
            self._emit(*self._pending.popleft())

    def _emit(self, pending, frames: int) -> None:
        try:
            result = pending.result()
            if result is not None:
                self._write(result)
        except Exception as e:
            if self._recordFailure is None:
                self._raised = True
                raise
            self._recordFailure(e)
            logging.exception(f"Something went wrong while processing the frame, {e}")
        if self._progress is not None:
            self._progress(frames)

    def close(self) -> None:
        """Write everything still pending.

        Frames queued before a failure the caller raised were already written
        by the serial loop, so they are written here too; after a failure this
        sink raised, nothing more is."""
        try:
            while self._pending and not self._raised:
                self._emit(*self._pending.popleft())
        finally:
            self._pending.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


MEANTENSOR = (
    torch.tensor([0.485, 0.456, 0.406]).contiguous().view(3, 1, 1).to(checker.device)
)
STDTENSOR = (
    torch.tensor([0.229, 0.224, 0.225]).contiguous().view(3, 1, 1).to(checker.device)
)
MEANTENSOR_HALF = MEANTENSOR.half() if checker.cudaAvailable else MEANTENSOR
STDTENSOR_HALF = STDTENSOR.half() if checker.cudaAvailable else STDTENSOR


def calculateAspectRatio(width, height, depthQuality="high", isV3=False):
    if isV3:
        if depthQuality == "high":
            return ((max(width, height) + 13) // 14) * 14
        if depthQuality == "medium":
            return 700
        return 518

    if depthQuality == "high":
        # Whilst this doesn't necessarily allign with the model, it produces
        # sharper results at the cost of performance and some accuracy loss.
        newHeight = ((height + 13) // 14) * 14
        newWidth = ((width + 13) // 14) * 14
    else:
        # I'd suggest 700px as a good middle ground for resizing
        size = 700 if depthQuality == "medium" else 518
        newHeight = size
        newWidth = size

    logging.info(f"Depth Padding: {newWidth}x{newHeight}")
    return newHeight, newWidth


def aspectKeepingInputSize(width, height, depthQuality="low"):
    """Model input as (height, width) for the torch ``small_v2`` backends.

    Upstream Depth Anything V2's ``lower_bound`` resize: the shortest side
    reaches the quality's size, the other side follows the aspect ratio, and
    both land on a multiple of 14. ``high`` keeps the source-size grid. The ONNX
    backends stay on calculateAspectRatio's 518x518: their graphs are fixed to it.
    """
    if depthQuality == "high":
        return calculateAspectRatio(width, height, depthQuality)

    target = 700 if depthQuality == "medium" else 518
    scale = target / min(width, height)

    def snap(side):
        scaled = side * scale / 14
        snapped = round(scaled) * 14
        return snapped if snapped >= target else math.ceil(scaled) * 14

    newHeight, newWidth = snap(height), snap(width)
    logging.info(f"Depth Padding: {newWidth}x{newHeight}")
    return newHeight, newWidth


def videoDepthInputSize(width, height, depthQuality="low"):
    """Shortest-side target for VideoDepthAnything's ``input_size``.

    VDA keeps the aspect ratio and resizes so the shortest side reaches this
    (``lower_bound``, multiple of 14), so it is not a longest-side or square
    size like the image models get. ``low`` is the 518 the model was trained
    at, ``medium`` the 700 every other method uses, ``high`` the source's own
    shortest side. VDA still shrinks it for sources wider than 16:9.
    """
    if depthQuality == "high":
        size = ((min(width, height) + 13) // 14) * 14
    else:
        size = 700 if depthQuality == "medium" else 518
    logging.info(f"Video depth input size (shortest side): {size}")
    return size


# (height, width) of the two resolutions Limbo is hosted at -- the ones the
# model was trained at, and what --depth_quality low runs at on every backend.
LIMBO_SHAPES = ((280, 504), (378, 504))


def limboResolution(width, height, depthQuality="low"):
    """Limbo's input size as (height, width), matching calculateAspectRatio.

    ``low`` picks the hosted export closest to the source aspect ratio.
    Compared in log space, so a 16:10 or 1.85:1 source lands on the widescreen
    export rather than on 4:3 by a rounding accident, and anything squarer than
    ~1.55:1 (portrait included, since neither export is taller than it is wide)
    lands on 504x378.

    ``medium``/``high`` size it like the ``*_v3`` methods: source aspect, the
    longest side at 700 or at the source's own, in multiples of 14. The ONNX
    backends export a graph at that size themselves (limboOnnxPath).
    """
    if depthQuality in ("medium", "high"):
        longest = calculateAspectRatio(width, height, depthQuality, isV3=True)
        scale = longest / max(width, height)
        shape = (
            max(14, (max(1, round(height * scale)) // 14) * 14),
            max(14, (max(1, round(width * scale)) // 14) * 14),
        )
    else:
        aspect = max(width, 1) / max(height, 1)
        shape = min(
            LIMBO_SHAPES,
            key=lambda hw: abs(math.log(aspect) - math.log(hw[1] / hw[0])),
        )
    logging.info(f"Limbo input resolution: {shape[1]}x{shape[0]}")
    return shape


def limboOnnxPath(depthMethod, height, width, half, backend):
    """The Limbo ONNX for one input size, as a local path.

    The two LIMBO_SHAPES are hosted; any other size is exported once from the
    checkpoint into the same cache ``video_limbo*-tensorrt`` uses. ``backend``
    is the registry suffix, "tensorrt" or "directml" (the OpenVINO path).
    """
    from src.model.download import resolveWeightPath
    from src.model.registry import modelsMap

    base = depthMethod.removeprefix("video_").split("-")[0]
    if (height, width) in LIMBO_SHAPES:
        # The two exports differ only in resolution, so the shape is what
        # picks the weight file.
        aspect = "" if (height, width) == (280, 504) else "_43"
        registryModel = f"{base}{aspect}-{backend}"
        return resolveWeightPath(
            registryModel.replace(f"-{backend}", "-onnx"),
            modelsMap(model=registryModel, modelType="onnx", half=half),
            downloadModel=registryModel,
            half=half,
            modelType="onnx",
        )

    from pathlib import Path

    from src.depth.streaming_export import exportStreamingDepth

    checkpoint = resolveWeightPath(
        base, modelsMap(base, modelType="pth"), modelType="pth", half=half
    )
    return exportStreamingDepth(
        checkpoint, Path(checkpoint).parent / "streaming", height, width, half
    )


# Elements the stretch works on at once: its elementwise temporaries run to
# ~8x the maps' own float32 size, so a 16-map block of 720p ``*_v3`` maps is
# stretched a couple of maps at a time rather than holding ~0.5 GB at once.
_STRETCH_ELEMENTS = 1 << 21


def _mapPercentiles(sample, qs):
    """``qs`` quantiles of each [H, W] map of ``sample``, NaN skipped.

    One tensor per quantile, shaped ``[..., 1, 1]`` to broadcast against the
    maps. Linear interpolation over the ``(n - 1) * q`` rank, as
    ``np.percentile`` does. A quantile vector sorts each map once for both.

    One map per call: a batched ``dim=-1`` call runs a segmented sort that is
    slower on the GPU (4.0 vs 2.9 ms for 16 Limbo-sized maps on an RTX 3090)
    and needs ~20x the memory, for bit-identical results.
    """
    lead = sample.shape[:-2]
    rows = sample.reshape(-1, sample.shape[-2] * sample.shape[-1])
    q = rows.new_tensor(qs)
    bounds = torch.stack([torch.nanquantile(row, q) for row in rows], dim=1)
    return [bound.reshape(*lead, 1, 1) for bound in bounds]


def _stretchMaps(depth, minValid, widenFlat, depthFloor):
    depth = torch.nan_to_num(depth.float(), nan=0.0, posinf=0.0, neginf=0.0)
    valid = depth > 0

    safe = depth if depthFloor is None else depth.clamp_min(depthFloor)
    disparity = torch.where(valid, 1.0 / safe, torch.zeros_like(depth))

    sample = torch.where(valid, disparity, torch.full_like(disparity, float("nan")))
    low, high = _mapPercentiles(sample, [0.02, 0.98])
    del sample
    if widenFlat:
        flat = low == high
        low = torch.where(flat, low - 1e-6, low)
        high = torch.where(flat, high + 1e-6, high)
        gray = (disparity - low) / (high - low)
    else:
        gray = (disparity - low) / (high - low).clamp_min(1e-6)
    # An all-invalid map makes both quantiles NaN; write black rather than
    # letting a NaN reach the writer's quantization.
    gray = torch.nan_to_num(gray, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
    if minValid:
        enough = valid.flatten(-2).sum(-1) > minValid
        gray = torch.where(enough[..., None, None], gray, torch.zeros_like(gray))
    return gray


def disparityStretch(depth, *, minValid=0, widenFlat=False, depthFloor=1e-6):
    """Positive depth -> [0, 1] disparity, each [H, W] map stretched on its own.

    1/depth on the valid (finite, > 0) pixels, then the 2nd and 98th
    percentiles of those map to 0 and 1; invalid pixels are 0. Runs on the
    tensor's own device and never reads a value back to the host: the
    percentiles come from ``nanquantile`` over a NaN-masked copy rather than a
    boolean gather, which would need the valid count on the host every frame.
    A batch of maps is stretched ``_STRETCH_ELEMENTS`` at a time.

    The knobs keep each caller's established semantics:

    - ``minValid``: a map with this many valid pixels or fewer is all 0 (the
      ``*_v3`` methods use 10; Limbo none, an all-invalid map is 0 either way).
    - ``widenFlat``: a map whose bounds coincide is stretched over
      ``[low - 1e-6, high + 1e-6]`` (the ``*_v3`` image methods) rather than
      dividing by a range floored at 1e-6 (Limbo and the streaming methods).
    - ``depthFloor``: Limbo clamps depth to it before the reciprocal; None
      takes 1/depth as is.
    """
    perMap = depth.shape[-2] * depth.shape[-1]
    if depth.ndim < 3 or depth.numel() <= max(perMap, _STRETCH_ELEMENTS):
        return _stretchMaps(depth, minValid, widenFlat, depthFloor)
    maps = depth.reshape(-1, *depth.shape[-2:])
    step = max(1, _STRETCH_ELEMENTS // perMap)
    gray = torch.empty(maps.shape, dtype=torch.float32, device=depth.device)
    for i in range(0, len(maps), step):
        gray[i : i + step] = _stretchMaps(
            maps[i : i + step], minValid, widenFlat, depthFloor
        )
    return gray.reshape(depth.shape)


def limboDisparity(depth):
    """Limbo's positive depth -> a [0, 1] disparity map, as a torch tensor.

    Limbo is a Depth Anything 3 finetune, so it emits positive depth (further =
    larger) exactly like every ``*_v3`` method and gets the same 1/depth +
    2/98-percentile stretch those use, on the frame's own device.
    """
    return disparityStretch(depth)


def toDevice(frame, device):
    """One decoded frame (numpy or a CPU tensor) on ``device``, without blocking.

    A pageable host-to-device copy blocks the host until it lands, which on a
    stream the GPU is busy with means until everything queued before it has
    run. Staged through pinned memory, the copy is only queued: it runs in
    stream order (a few ms per 32-frame chunk, against a forward of ~100 ms)
    and the caching host allocator keeps the staging buffer until it has. The
    staging copy is numpy's, on this thread: ``Tensor.pin_memory`` copies on
    the OpenMP pool, which then spins a dozen cores.
    """
    tensor = frame if isinstance(frame, torch.Tensor) else torch.from_numpy(frame)
    device = torch.device(device)
    if device.type != "cuda" or tensor.device.type != "cpu":
        return tensor.to(device)
    staged = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True)
    staged.numpy()[...] = tensor.numpy()
    return staged.to(device, non_blocking=True)


def stackFrames(frames, device):
    """Decoded [H, W, 3] frames -> one [N, H, W, 3] tensor on ``device``."""
    return torch.stack([toDevice(frame, device) for frame in frames])
