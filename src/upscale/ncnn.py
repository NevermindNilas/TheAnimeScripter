import math
import os

import numpy as np
import torch

from src.infra.vulkanDevice import CPU_DEVICE, bestVulkanDevice
from src.model.download import resolveWeightPath
from src.model.registry import modelsMap, ncnnUpscaleRefusal

# GPU memory the NCNN upscalers use per input pixel: measured peak on span and
# shufflecugan, 2.3 KiB at 720p and 2.5 KiB at 360p.
_BYTES_PER_PIXEL = 2600
# Leaves room for an NCNN interpolate stage and whatever else holds the GPU.
_BUDGET_SHARE = 0.7
_MIN_TILE = 64
# shufflecugan's UNet returns garbage unless both sides divide by this.
_MODULUS = 4
# Context each tile sees past its own edges (input pixels), cropped from the output.
_TILE_PADDING = 48


def tileSizeFor(width: int, height: int, budgetMegabytes: int) -> int:
    """The tile side that fits ``budgetMegabytes``, or 0 for a whole-frame run."""
    usable = budgetMegabytes * 1024 * 1024 * _BUDGET_SHARE
    if width * height * _BYTES_PER_PIXEL <= usable:
        return 0
    side = int(math.sqrt(usable / _BYTES_PER_PIXEL)) - 2 * _TILE_PADDING
    return max(_MIN_TILE, side) & ~(_MODULUS - 1)


class UniversalNCNN:
    def __init__(self, upscaleMethod, upscaleFactor):
        self.upscaleMethod = upscaleMethod
        self.upscaleFactor = upscaleFactor

        # Backstop: the CLI (validator._finalizeChain) refuses this first.
        problem = ncnnUpscaleRefusal(upscaleMethod)
        if problem is not None:
            raise ImportError(problem)

        import ncnn  # ty: ignore[unresolved-import]

        self.ncnn = ncnn

        self.filename = modelsMap(
            self.upscaleMethod,
            modelType="ncnn",
        )

        if self.filename.endswith("-ncnn.zip"):
            self.filename = self.filename[:-9]
        elif self.filename.endswith("-ncnn"):
            self.filename = self.filename[:-5]

        modelPath = resolveWeightPath(
            self.upscaleMethod,
            self.filename,
            modelType="ncnn",
        )

        if modelPath.endswith("-ncnn.zip"):
            modelPath = modelPath[:-9]
        elif modelPath.endswith("-ncnn"):
            modelPath = modelPath[:-5]

        lastSlash = os.path.basename(modelPath)
        modelStem = os.path.join(modelPath, lastSlash)

        self.device = bestVulkanDevice()
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = self.device != CPU_DEVICE
        # Shader FP16 math turns shufflecugan's output into NaN (seen on Intel Mesa).
        self.net.opt.use_fp16_arithmetic = False
        if self.device != CPU_DEVICE:
            self.net.set_vulkan_device(self.device)

        for path, load in (
            (f"{modelStem}.param", self.net.load_param),
            (f"{modelStem}.bin", self.net.load_model),
        ):
            if load(path) != 0:
                raise FileNotFoundError(
                    f"{upscaleMethod}: could not load {path}. Delete that folder "
                    "so TAS downloads the model again."
                )

        self.tileSize = None
        self.outputChecked = False

    def _budgetTileSize(self, width, height):
        if self.device == CPU_DEVICE:
            return 0
        budget = self.ncnn.get_gpu_device(self.device).get_heap_budget()
        return tileSizeFor(width, height, budget)

    def _infer(self, chw):
        extractor = self.net.create_extractor()
        extractor.input("data", self.ncnn.Mat(chw).clone())
        status, output = extractor.extract("output")
        if status != 0:
            raise RuntimeError(
                f"{self.upscaleMethod}: NCNN failed with status {status} on a "
                f"{chw.shape[2]}x{chw.shape[1]} input. The GPU may be out of memory."
            )
        return np.array(output)

    def _inferTiled(self, chw, tile):
        scale = self.upscaleFactor
        _, height, width = chw.shape
        stitched = np.empty((3, height * scale, width * scale), dtype=np.float32)
        for top in range(0, height, tile):
            for left in range(0, width, tile):
                bottom, right = min(top + tile, height), min(left + tile, width)
                paddedTop = max(top - _TILE_PADDING, 0)
                paddedLeft = max(left - _TILE_PADDING, 0)
                paddedBottom = min(bottom + _TILE_PADDING, height)
                paddedRight = min(right + _TILE_PADDING, width)
                # A copy: ncnn.Mat reads the buffer as contiguous, a slice is not.
                output = self._infer(
                    chw[:, paddedTop:paddedBottom, paddedLeft:paddedRight].copy()
                )
                y, x = (top - paddedTop) * scale, (left - paddedLeft) * scale
                stitched[
                    :, top * scale : bottom * scale, left * scale : right * scale
                ] = output[
                    :, y : y + (bottom - top) * scale, x : x + (right - left) * scale
                ]
        return stitched

    def __call__(self, frame, nextFrame: None) -> torch.tensor:
        dtype = frame.dtype
        chw = frame.squeeze(0).float().contiguous().cpu().numpy()
        _, height, width = chw.shape

        padded = np.pad(
            chw,
            ((0, 0), (0, -height % _MODULUS), (0, -width % _MODULUS)),
            mode="edge",
        )

        if self.tileSize is None:
            self.tileSize = self._budgetTileSize(padded.shape[2], padded.shape[1])

        if self.tileSize:
            output = self._inferTiled(padded, self.tileSize)
        else:
            output = self._infer(padded)

        # NaN would be encoded as a black video on a run that reports success.
        if not self.outputChecked:
            if not np.isfinite(output).all():
                raise RuntimeError(
                    f"{self.upscaleMethod}: this GPU's Vulkan driver produced invalid "
                    "output. Update the driver or use another backend."
                )
            self.outputChecked = True

        scale = self.upscaleFactor
        return (
            torch.from_numpy(output[:, : height * scale, : width * scale])
            .clamp(0, 1)
            .to(dtype)
            .unsqueeze(0)
        )
