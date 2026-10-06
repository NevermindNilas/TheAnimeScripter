import math
import os

os.environ.setdefault("DA3_LOG_LEVEL", "ERROR")

import logging

import torch

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
        """Tell the AE panel how the run ended. Safe to call more than once."""
        from src.constants import ADOBE

        if not ADOBE:
            return
        from src.server.aeComms import reportTerminalStatus

        reportTerminalStatus(self.processingError, self.output, self.benchmark)


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

    base = "limbo_v2" if depthMethod.startswith("limbo_v2") else "limbo"
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


def limboDisparity(depth):
    """Limbo's positive depth -> a [0, 1] disparity map, as a torch tensor.

    Limbo is a Depth Anything 3 finetune, so it emits positive depth (further =
    larger) exactly like every ``*_v3`` method and gets the same 1/depth +
    2/98-percentile stretch those use. Written against torch rather than the
    numpy the v3 CUDA/MPS drivers use so the CUDA and TensorRT backends can run
    it on the frame's own device without a round trip.

    `nanquantile` over a NaN-masked copy rather than boolean indexing keeps the
    whole thing free of device syncs: a masked gather would have to read the
    valid-pixel count back to the host every frame.
    """
    depth = torch.nan_to_num(depth.float(), nan=0.0, posinf=0.0, neginf=0.0)
    valid = depth > 0

    disparity = torch.where(valid, 1.0 / depth.clamp_min(1e-6), torch.zeros_like(depth))

    sample = torch.where(valid, disparity, torch.full_like(disparity, float("nan")))
    # A quantile vector sorts the map once for both bounds.
    low, high = torch.nanquantile(sample.flatten(), sample.new_tensor([0.02, 0.98]))
    gray = (disparity - low) / (high - low).clamp_min(1e-6)
    # An all-invalid frame makes both quantiles NaN; write black rather than
    # letting a NaN reach the writer's quantization.
    return torch.nan_to_num(gray, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
