import logging
import platform

import torch

from src.constants import ADOBE
from src.infra.isCudaInit import CudaChecker
from src.interpolate._timesteps import interpolateTimestep

if ADOBE:
    from src.server.aeComms import progressState


def _isSupportedGpu(capability, system):
    """VFX SDK 1.3 ships Video Frame Generation for Ada and Blackwell, plus
    Hopper on Linux. Anything else fails NvVFX_Load with an opaque
    "unknown error (code -1999)", so check up front instead."""
    if capability == (8, 9) or capability[0] >= 10:
        return True
    return capability == (9, 0) and system == "Linux"


class MaxineInterpolate:
    """
    NVIDIA Maxine Video Frame Generation (nvidia-vfx >= 0.2.0.0).

    Quality mode is parsed from `interpolateMethod`: `maxine-low`,
    `maxine-medium`, `maxine-high`.

    API hard limits (VFX SDK 1.3):
      - Inputs are two (3, H, W) float32 CUDA frames; output is the same shape
      - Input dimensions are fixed at load()
      - Ada / Blackwell GPUs (Hopper on Linux) only
      - On a pair it detects as a shot change, the SDK skips interpolation and
        returns the current frame -- the same hold --scenechange produces.

    Every slot goes through the SDK's explicit-timestep mode: it covers the
    gap planner's fractional and --smooth_dedup timesteps as well as the
    integer ladder, which multiplier mode caps at 8x.
    """

    _VALID_MODES = {"LOW", "MEDIUM", "HIGH"}

    def __init__(
        self,
        interpolateMethod: str = "maxine-medium",
        width: int = 1920,
        height: int = 1080,
    ):
        self.interpolateMethod = interpolateMethod
        self.width = width
        self.height = height
        self.modeName = self._parseMode(interpolateMethod)

        checker = CudaChecker()
        if not checker.cudaAvailable or checker.rocmAvailable:
            raise RuntimeError(
                "NVIDIA Maxine frame generation requires an NVIDIA CUDA GPU."
            )
        self.device = checker.device

        self.handleModel()

    @classmethod
    def _parseMode(cls, method: str) -> str:
        name = method.lower().replace("maxine", "", 1).lstrip("-").upper()
        if name not in cls._VALID_MODES:
            raise ValueError(
                f"Unknown Maxine frame generation mode '{name}' in "
                f"interpolateMethod '{method}'. Valid: {sorted(cls._VALID_MODES)}"
            )
        return name

    def handleModel(self):
        deviceIdx = self.device.index if self.device.index is not None else 0

        capability = tuple(torch.cuda.get_device_capability(deviceIdx))
        if not _isSupportedGpu(capability, platform.system()):
            raise RuntimeError(
                "NVIDIA Maxine frame generation requires an Ada or Blackwell GPU "
                "(RTX 40/50 series; Hopper on Linux); "
                f"{torch.cuda.get_device_name(deviceIdx)} (compute "
                f"{capability[0]}.{capability[1]}) is not supported. "
                "Use a RIFE method instead."
            )

        if ADOBE:
            progressState.update(
                {"status": f"Loading NVIDIA frame generation ({self.modeName})..."}
            )

        try:
            from nvvfx import VideoFrameGeneration
        except ImportError as e:
            raise RuntimeError(
                "NVIDIA Maxine frame generation needs nvidia-vfx 0.2.0.0 or newer."
            ) from e

        self.model = VideoFrameGeneration(
            mode=VideoFrameGeneration.Mode[self.modeName], device=deviceIdx
        )
        self.model.input_width = self.width
        self.model.input_height = self.height

        try:
            self.model.load()
        except Exception as e:
            logging.error(f"NVIDIA frame generation load failed: {e}")
            raise

        self.I0 = torch.zeros(
            (3, self.height, self.width), device=self.device, dtype=torch.float32
        )
        self.I1 = torch.zeros_like(self.I0)

        self.firstRun = True

    def _stage(self, target, frame):
        expected = (1, 3, self.height, self.width)
        if tuple(frame.shape) != expected:
            raise ValueError(
                f"Maxine frame generation was built for {expected}, "
                f"expected that shape but got {tuple(frame.shape)}."
            )
        target.copy_(frame[0])

    @torch.inference_mode()
    def cacheFrameReset(self, frame):
        # No feature cache to re-seed: the SDK takes both endpoints every call.
        self._stage(self.I0, frame)
        self.firstRun = False

    @torch.inference_mode()
    def __call__(self, frame, interpQueue, framesToInsert: int = 1, timesteps=None):
        if self.firstRun:
            self._stage(self.I0, frame)
            self.firstRun = False
            return

        self._stage(self.I1, frame)

        stream = torch.cuda.current_stream(self.device)
        for i in range(framesToInsert):
            result = self.model.run_at_timestep(
                self.I0,
                self.I1,
                interpolateTimestep(i, framesToInsert, timesteps),
                stream_ptr=stream.cuda_stream,
            )
            # The capsule aliases the SDK's single output buffer, which the next
            # run overwrites, so it has to be copied out before the next slot.
            output = torch.from_dlpack(result.image).unsqueeze(0).clone()
            # The writer copies on its own stream and never waits on this one.
            stream.synchronize()
            interpQueue.put(output)

        self.I0, self.I1 = self.I1, self.I0
        # The staging copy above may still be reading `frame`; finish it before
        # the caller recycles that storage on another stream.
        stream.synchronize()
