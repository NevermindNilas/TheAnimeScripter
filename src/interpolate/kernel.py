"""The `-kernel` interpolate backend: RIFE and DistilDRBA on the tas-kernels package's CUDA kernels.

    driver = RifeKernel(half, width, height, "rife4.25-kernel", interpolateFactor=2)   # RifeCuda's contract
    driver(frame, sink, framesToInsert, timesteps); driver.cacheFrameReset(frame)
    driver = DrbaKernel(half, width, height, "distildrba-kernel", interpolateFactor=2) # DistilDRBACuda's contract
    driver(frame, nextFrame, sink, framesToInsert, timesteps); driver.cacheFrameReset(frame)

A `-kernel` method reads the CUDA method's weights and runs TAS's fast RIFE path (per-pair encode,
per-timestep flow, flow caching) as hand-written kernels; tas_kernels.rife pads, rolls the frame
slots and allocates fresh outputs itself. Warp coordinates, accumulated flow and the final blend are
fp32 in both precisions (TAS's own fp16 paths build an fp16 sampling grid, which flickers on slow
pans), so the output differs from RifeCuda's fp16 output by design and is closer to its fp32 output.

What the drivers rely on (tas_kernels.rife):
  * reset() / pair() ENQUEUE kernels on the current stream that read the caller's frame: the driver's
    stream waits for the caller's first, and is synchronized before the call returns (every path
    below does), so the frame is no longer read afterwards.
  * flows(ts) returns new [1, 3, H, W] tensors (each bit-identical to flow(t)), the pair's steps
    running concurrently on streams joined back into the current one; the sink may keep them.
  * rife4.26-ultra shares rife4.26-heavy's weights: the variant name is always passed.
  * RIFE frames above 1920x1080 (both dimensions) run at scale 0.5 in fp32, TAS's rule (_uhdScale);
    DistilDRBA above 1920x1080 (either dimension) at scale 0.5 with /128 replicate padding (TAS pads
    /64 there, which misaligns v1 and makes lite raise).
  * DrbaModel.pair(frame, next) takes I1 from the previous call's next frame (cached image and
    features) unless fresh=True; DrbaKernel passes fresh=True whenever `frame` is not the tensor it
    was handed as the previous nextFrame.

Settings the kernels do not implement (--ensemble, --dynamic_scale, --static_step) make the factory
build the CUDA method instead: kernelFallbackReason() says why.
"""

import importlib

import torch

from src.constants import ADOBE
from src.infra.cudaStreams import waitForCaller
from src.interpolate._timesteps import interpolateTimestep
from src.model.download import resolveWeightPath
from src.model.registry import modelsMap
from src.upscale.kernel import KERNEL_SUFFIX, readStateDict

if ADOBE:
    from src.server.aeComms import progressState

# At most this many outputs are computed before they go to the sink: flows() runs them concurrently,
# and a long smooth-dedup gap goes out in chunks instead of all at once.
CHUNK = 8
# src/interpolate/rife.py: the only methods --static_step applies to
_STATIC_STEP_METHODS = ("rife4.6", "rife4.15", "rife4.15-lite")


def baseMethod(interpolateMethod: str) -> str:
    """'rife4.25-kernel' -> 'rife4.25' (TAS's tables key on the base name)."""
    return interpolateMethod.removesuffix(KERNEL_SUFFIX)


def kernelFallbackReason(
    interpolateMethod: str,
    ensemble: bool = False,
    dynamicScale: bool = False,
    staticStep: bool = False,
) -> str | None:
    """Why a `-kernel` interpolate method must run TAS's CUDA driver instead (None: the kernels cover it).

    Called with the stage's settings after the validator resolved them.
    """
    base = baseMethod(interpolateMethod)
    if base.startswith("distildrba"):
        return None  # DistilDRBA has none of these options
    if ensemble:
        return "--ensemble (a second, swapped evaluation of every block) is not implemented by the kernels"
    if dynamicScale:
        return "--dynamic_scale (a per-pair pyramid scale) is not implemented by the kernels"
    if staticStep and base in _STATIC_STEP_METHODS:
        return "--static_step (TAS's reference arch) is not implemented by the kernels"
    return None


def _loadStateDict(interpolateMethod: str) -> dict:
    """The CUDA method's weight file, read the way its CUDA class reads it."""
    base = baseMethod(interpolateMethod)
    filename = modelsMap(base)  # rife4.26-ultra -> rife4.26-heavy's file
    if base.startswith("distildrba"):
        path = resolveWeightPath(base, filename)
    else:
        path = resolveWeightPath("rife", filename, downloadModel=base)
    return readStateDict(path, trusted=True)


class _KernelDriver:
    """What RifeKernel and DrbaKernel share: model loading, frame conversion, emitting a gap."""

    interpolateMethod: str
    model: object
    stream: torch.cuda.Stream

    def _status(self):
        if ADOBE:
            progressState.update(
                {"status": f"Loading interpolation model: {self.interpolateMethod}..."}
            )

    def _frame(self, frame):
        """The frame on the model's device and dtype, contiguous NCHW (usually the same tensor)."""
        frame = frame.to(
            device=self.model.device, dtype=self.model.dtype, non_blocking=True
        )
        return frame.contiguous()

    def _emit(self, interpQueue, framesToInsert, timesteps):
        """flows() over the gap's timesteps in chunks of CHUNK; each chunk is finished before its
        frames go to the sink, in ascending time order, outside the private stream context."""
        ts = [
            interpolateTimestep(i, framesToInsert, timesteps)
            for i in range(framesToInsert)
        ]
        for k in range(0, len(ts), CHUNK):
            with torch.cuda.stream(self.stream):
                outputs = self.model.flows(ts[k : k + CHUNK])
            self.stream.synchronize()
            for output in outputs:
                interpQueue.put(output)


class RifeKernel(_KernelDriver):
    """RIFE 4.x / rife_elexor / rife4.26-ultra with RifeCuda's contract. Raises ValueError (a
    checkpoint or size the kernels do not cover) or RuntimeError (no usable GPU / library)."""

    temporalWindow = (0, 0)

    def __init__(self, half, width, height, interpolateMethod, interpolateFactor=2):
        # imported by name: the wheel holds compiled modules only, with no stubs
        RifeModel = importlib.import_module("tas_kernels.rife").RifeModel

        self.interpolateMethod = interpolateMethod
        self.interpolateFactor = interpolateFactor
        self.width, self.height = width, height
        self._status()
        if width > 1920 and height > 1080:  # RifeCuda's _uhdScale: scale 0.5 in fp32
            half = False
        self.half = half
        base = baseMethod(interpolateMethod)
        self.model = RifeModel(
            _loadStateDict(interpolateMethod), height, width, half=half, variant=base
        )
        self.dType = self.model.dtype
        self.stream = torch.cuda.Stream()
        self.firstRun = True
        self.useGraph = (
            False  # RifeCuda's flag for src/motionBlur.py (no -kernel methods there)
        )

    @torch.inference_mode()
    def cacheFrameReset(self, frame):
        """Scene cut: `frame` becomes the anchor; nothing is emitted."""
        waitForCaller(self.stream)
        with torch.cuda.stream(self.stream):
            self.model.reset(self._frame(frame))
        self.stream.synchronize()
        self.firstRun = False

    @torch.inference_mode()
    def __call__(self, frame, interpQueue, framesToInsert: int = 2, timesteps=None):
        waitForCaller(self.stream)
        with torch.cuda.stream(self.stream):
            paired = self.model.pair(
                self._frame(frame)
            )  # False: the first frame only seeds
        if self.firstRun or not paired:
            self.firstRun = False
            self.stream.synchronize()
            return
        if framesToInsert:
            self._emit(interpQueue, framesToInsert, timesteps)
        else:
            self.stream.synchronize()  # the read of `frame` queued by pair() must finish


class DrbaKernel(_KernelDriver):
    """DistilDRBA / DistilDRBA-lite with DistilDRBACuda's contract (frame + next frame)."""

    temporalWindow = (0, 1)

    def __init__(self, half, width, height, interpolateMethod, interpolateFactor=2):
        DrbaModel = importlib.import_module("tas_kernels.rife").DrbaModel

        self.interpolateMethod = interpolateMethod
        self.interpolateFactor = interpolateFactor
        self.width, self.height, self.half = width, height, half
        self._status()
        # earlySteps (a 4th frame slot for t < 0.5) stays on: smooth dedup and fractional factors ask
        # for early timesteps.
        self.model = DrbaModel(
            _loadStateDict(interpolateMethod),
            height,
            width,
            half=half,
            variant=baseMethod(interpolateMethod),
        )
        self.dType = self.model.dtype
        self.stream = torch.cuda.Stream()
        self.firstRun = True
        self._lastNext = (
            None  # the previous call's nextFrame: the model cached it as the next I1
        )

    @torch.inference_mode()
    def cacheFrameReset(self, frame):
        """Scene cut: I0 = Iprev = frame, every cached frame dropped; nothing is emitted."""
        waitForCaller(self.stream)
        with torch.cuda.stream(self.stream):
            self.model.reset(self._frame(frame))
        self.stream.synchronize()
        self.firstRun = False
        self._lastNext = None

    @torch.inference_mode()
    def cacheFrame(self, frame):
        """DistilDRBACuda.cacheFrame: advance to `frame` without output."""
        self(frame, None, None, 0, None)

    @torch.inference_mode()
    def __call__(
        self, frame, nextFrame, interpQueue, framesToInsert: int = 2, timesteps=None
    ):
        if nextFrame is not None and nextFrame.shape[-2:] != frame.shape[-2:]:
            raise RuntimeError(
                f"DistilDRBA got a next frame of {tuple(nextFrame.shape[-2:])} against a current "
                f"frame of {tuple(frame.shape[-2:])}; the caller must supply the neighbour in the "
                "same domain."
            )
        waitForCaller(self.stream)
        with torch.cuda.stream(self.stream):
            x = self._frame(frame)
            if self.firstRun:
                self.model.reset(x)  # TAS: I0 = Iprev = I1, nothing emitted
                paired = False
            else:
                # I2 = nextFrame (None at the end of the stream or before a cut: I2 = I1)
                fresh = frame is not self._lastNext
                paired = self.model.pair(
                    x,
                    None if nextFrame is None else self._frame(nextFrame),
                    fresh=fresh,
                )
        self._lastNext = nextFrame
        if not paired:
            self.firstRun = False
            self.stream.synchronize()
            return
        if framesToInsert:
            # t >= 0.5: (I0, I1, I2) at t; t < 0.5: (Iprev, I0, I1) at 1 + t (TAS's _drbaStep)
            self._emit(interpQueue, framesToInsert, timesteps)
        else:
            self.stream.synchronize()
