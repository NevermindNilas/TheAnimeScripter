import logging

import torch
import torch.nn.functional as F

from src.constants import ADOBE
from src.infra.isCudaInit import CudaChecker
from src.infra.logAndPrint import logAndPrint
from src.interpolate._padding import _padMultiple
from src.interpolate._shared import importRifeArch
from src.interpolate._timesteps import fillTimestepBuffer, interpolateTimestep
from src.model.download import resolveWeightPath
from src.model.registry import modelsMap

if ADOBE:
    from src.server.aeComms import progressState


checker = CudaChecker()

torch.set_float32_matmul_precision("medium")


class RifeCuda:
    def __init__(
        self,
        half,
        width,
        height,
        interpolateMethod,
        ensemble=False,
        interpolateFactor=2,
        dynamicScale=False,
        staticStep=False,
        compileMode: str = "default",
    ):
        """
        Initialize the RIFE model

        Args:
            half (bool): Half resolution
            width (int): Width of the frame
            height (int): Height of the frame
            interpolateMethod (str): Interpolation method
            ensemble (bool, optional): Ensemble. Defaults to False.
            interpolateFactor (int, optional): Interpolation factor. Defaults to 2.
            dynamicScale (bool, optional): Use Dynamic scale. Defaults to False.
            staticStep (bool, optional): Use static timestep. Defaults to False.
        """
        self.half = half
        self.scale = 1.0
        self.width = width
        self.height = height
        self.interpolateMethod = interpolateMethod
        self.ensemble = ensemble
        self.interpolateFactor = interpolateFactor
        self.dynamicScale = dynamicScale
        self.staticStep = staticStep
        self.compileMode = compileMode

        if self.width > 1920 and self.height > 1080:
            self.scale = 0.5
            if self.half:
                logAndPrint(
                    "UHD and fp16 are not compatible with RIFE, defaulting to fp32",
                    "yellow",
                )
                self.half = False

        self.handleModel()

    def handleModel(self):
        """
        Load the desired model
        """
        if ADOBE:
            progressState.update(
                {
                    "status": f"Loading RIFE interpolation model: {self.interpolateMethod}..."
                }
            )

        self.filename = modelsMap(self.interpolateMethod)
        modelPath = resolveWeightPath(
            "rife", self.filename, downloadModel=self.interpolateMethod
        )

        self.dType = torch.float16 if self.half else torch.float32

        IFNet = importRifeArch(self.interpolateMethod, "v1", half=self.half)
        if self.interpolateMethod in ["rife_elexor"] and self.staticStep:
            self.staticStep = False
            logAndPrint(
                "Static step is not supported for rife_elexor, automatically disabling it",
                "yellow",
            )
        if (
            self.interpolateMethod not in ["rife4.6", "rife4.15", "rife4.15-lite"]
            and self.staticStep
        ):
            self.staticStep = False
            logAndPrint(
                "Static step is not supported for this interpolation model yet, automatically disabling it",
                "yellow",
            )
        if self.interpolateMethod in ["rife_elexor"]:
            self.model = IFNet(
                self.scale,
                self.ensemble,
                self.dType,
                checker.device,
                self.width,
                self.height,
                self.interpolateFactor,
            )
        else:
            if self.interpolateMethod in ["rife4.6", "rife4.15", "rife4.15-lite"]:
                self.model = IFNet(
                    self.ensemble,
                    self.dynamicScale,
                    self.scale,
                    self.interpolateFactor,
                    self.staticStep,
                )
            else:
                self.model = IFNet(
                    self.ensemble,
                    self.dynamicScale,
                    self.scale,
                    self.interpolateFactor,
                )

        stateDict = torch.load(modelPath, map_location="cpu")
        self.model.load_state_dict(stateDict)
        del stateDict

        if hasattr(self.model, "repackWeights"):
            self.model.repackWeights()

        if checker.cudaAvailable and self.half:
            self.model = self.model.half()
        else:
            self.half = False
            self.model = self.model.float()

        self.model = self.model.eval()
        if checker.cudaAvailable:
            self.model = self.model.cuda()
            torch.cuda.empty_cache()
        self.model = self.model.to(memory_format=torch.channels_last)

        if self.compileMode != "default":
            try:
                if self.compileMode == "max":
                    self.model.compile(mode="max-autotune-no-cudagraphs")
                elif self.compileMode == "max-graphs":
                    self.model.compile(
                        mode="max-autotune-no-cudagraphs", fullgraph=True
                    )
            except Exception as e:
                logging.error(
                    f"Error compiling model {self.interpolateMethod} with mode {self.compileMode}: {e}"
                )
                logAndPrint(
                    f"Error compiling model {self.interpolateMethod} with mode {self.compileMode}: {e}",
                    "red",
                )

            self.compileMode = "default"

        mul = _padMultiple(self.interpolateMethod, self.scale, self.dynamicScale)
        ph = ((self.height - 1) // mul + 1) * mul
        pw = ((self.width - 1) // mul + 1) * mul
        self.padding = (0, pw - self.width, 0, ph - self.height)

        self.I0 = torch.zeros(
            1,
            3,
            self.height + self.padding[3],
            self.width + self.padding[1],
            dtype=self.dType,
            device=checker.device,
        ).to(memory_format=torch.channels_last)

        self.I1 = torch.zeros(
            1,
            3,
            self.height + self.padding[3],
            self.width + self.padding[1],
            dtype=self.dType,
            device=checker.device,
        ).to(memory_format=torch.channels_last)

        self.firstRun = True
        self.stream = torch.cuda.Stream()
        self.normStream = torch.cuda.Stream()

        self._timestep_buffer = torch.zeros(
            (1, 1, self.height + self.padding[3], self.width + self.padding[1]),
            dtype=self.dType,
            device=checker.device,
        )
        self._cachedTimestepValue = None

        self._setupCudaGraph()

    @torch.inference_mode()
    def _setupCudaGraph(self):
        """
        Capture the per-frame model forward into a CUDA graph and replay it in
        the "infer" path, removing eager per-kernel launch overhead.

        Only the forward is captured; it is replayed on ``normStream`` in the
        exact spot the eager call ran, so every existing stream synchronize /
        cross-op race guard (decode-buffer / upscale interaction) is preserved
        unchanged. I0/I1/_timestep_buffer are fixed buffers already, so replay
        reads their current contents.

        Disabled when:
          - not CUDA,
          - ``staticStep`` (different forward signature/return), or
          - ``dynamicScale`` (scale_list recomputed per frame from data -> a
            single captured graph would bake in one scale).
        Any arch whose forward is not safely capturable is caught by the
        self-check below (graph replay must match an eager forward) and falls
        back to the eager path.
        """
        self.cudaGraph = None
        self._graphOut = None
        self.useGraph = (
            checker.cudaAvailable and not self.staticStep and not self.dynamicScale
        )
        if not self.useGraph:
            return

        try:
            warmStream = torch.cuda.Stream()
            warmStream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warmStream):
                for _ in range(3):
                    self.model(self.I0, self.I1, self._timestep_buffer)
            torch.cuda.current_stream().wait_stream(warmStream)
            torch.cuda.synchronize()

            self.cudaGraph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.cudaGraph, stream=self.normStream):
                self._graphOut = self.model(self.I0, self.I1, self._timestep_buffer)
            self.normStream.synchronize()

            # Self-check: replay must match a fresh eager forward on the same
            # inputs, else disable the graph (protects arches where capture
            # silently misbehaves).
            self._timestep_buffer.fill_(0.5)
            self.I1.copy_(self.I0)
            eagerRef = self.model(self.I0, self.I1, self._timestep_buffer).clone()
            self.cudaGraph.replay()
            self.normStream.synchronize()
            if not torch.allclose(eagerRef, self._graphOut, rtol=1e-3, atol=1e-3):
                raise RuntimeError("graph replay output != eager forward")
            self._timestep_buffer.zero_()
            self.I1.zero_()
        except Exception as e:
            logging.error(
                f"RifeCuda CUDA-graph capture disabled for "
                f"{self.interpolateMethod}: {e}"
            )
            self.cudaGraph = None
            self._graphOut = None
            self.useGraph = False

    @torch.inference_mode()
    def cacheFrameReset(self, frame):
        """
        Scene-cut reset: make ``frame`` the sole anchor (I0) and re-seed the
        encoder feature ``f0 = encode(frame)``, so the next call interpolates
        ``frame`` <-> next-frame with no bleed from the previous scene. Leaves
        ``firstRun`` False (we already have an anchor) — the next ``__call__``
        takes the normal I1/interpolate path.

        GRAPH-SAFE: the forward is captured in a CUDA graph that reads ``I0`` and
        ``self.model.f0`` at FIXED addresses. Both are updated IN-PLACE via
        ``copy_`` — never reassigned. The arch's own ``cacheReset`` does
        ``self.f0 = self.encode(...)`` (a reassignment), which would leave graph
        replay reading the stale captured tensor; that is why we do not call it
        here when the graph is active.
        """
        with torch.cuda.stream(self.normStream):
            padded = self.padFrame(
                frame.to(device=checker.device, dtype=self.dType, non_blocking=True)
            )
            self.I0.copy_(padded, non_blocking=True)
            if getattr(self.model, "encode", None) is not None:
                if getattr(self.model, "f0", None) is not None:
                    # In-place re-seed of the persistent (graph-captured) f0.
                    self.model.f0.copy_(
                        self.model.encode(padded[:, :3]), non_blocking=True
                    )
                else:
                    # No forward has run yet (eager path, graph disabled): the
                    # next forward lazily encodes img0=I0, so leaving f0 None is
                    # correct.
                    self.model.f0 = None
            # Re-arm the arch's ensemble/factor counter to its initial state.
            if hasattr(self.model, "counter"):
                self.model.counter = 1
        self.normStream.synchronize()

    @torch.inference_mode()
    def processFrame(self, frame, toNorm):
        match toNorm:
            case "I0":
                with torch.cuda.stream(self.normStream):
                    frame = frame.to(
                        device=checker.device,
                        dtype=self.dType,
                        non_blocking=True,
                    )
                    frame = self.padFrame(frame)
                    self.I0.copy_(frame, non_blocking=True)
                # Sync only when data must be ready for inference
                self.normStream.synchronize()

            case "I1":
                with torch.cuda.stream(self.normStream):
                    frame = frame.to(
                        device=checker.device,
                        dtype=self.dType,
                        non_blocking=True,
                    )
                    frame = self.padFrame(frame)
                    self.I1.copy_(frame, non_blocking=True)
                # Sync only when data must be ready for inference
                self.normStream.synchronize()

            case "cache":
                with torch.cuda.stream(self.normStream):
                    self.I0.copy_(self.I1, non_blocking=True)
                    self.model.cache()
                self.normStream.synchronize()

            case "infer":
                with torch.cuda.stream(self.normStream):
                    if self.useGraph:
                        # `frame` here is self._timestep_buffer (filled in-place
                        # by __call__); the graph reads it + I0/I1 at their fixed
                        # addresses. Replay on normStream == same stream/ordering
                        # as the eager forward it replaces.
                        self.cudaGraph.replay()
                        output = self._graphOut[
                            :, :, : self.height, : self.width
                        ].clone()
                    elif self.staticStep:
                        output = self.model(self.I0, self.I1, frame).clone()
                    else:
                        output = self.model(self.I0, self.I1, frame)[
                            :, :, : self.height, : self.width
                        ].clone()
                self.normStream.synchronize()
                return output

            case "model":
                with torch.cuda.stream(self.normStream):
                    self.model.cacheReset(frame)
                self.normStream.synchronize()

    @torch.inference_mode()
    def padFrame(self, frame):
        return (
            F.pad(frame, [0, self.padding[1], 0, self.padding[3]])
            if self.padding != (0, 0, 0, 0)
            else frame
        )

    @torch.inference_mode()
    def __call__(self, frame, interpQueue, framesToInsert: int = 2, timesteps=None):
        if self.firstRun:
            self.processFrame(frame, "I0")
            self.firstRun = False
            return
        self.processFrame(frame, "I1")

        for i in range(framesToInsert):
            t = interpolateTimestep(i, framesToInsert, timesteps)
            # The common 2x path uses the same 0.5 timestep every frame. Avoid
            # refilling the full HxW tensor unless the requested timestep changes.
            self._cachedTimestepValue = fillTimestepBuffer(
                self._timestep_buffer, self._cachedTimestepValue, t
            )
            output = self.processFrame(self._timestep_buffer, "infer")
            interpQueue.put(output)

        self.processFrame(None, "cache")


class RifeMPS:
    """
    Apple Silicon (MPS) RIFE interpolator. Mirrors RifeCuda but drops
    torch.cuda.Stream — MPS has no stream equivalent. Shares .pth weights
    with the CUDA path: the "-mps" suffix on interpolateMethod is stripped
    before resolving model filenames and importing the arch.
    """

    def __init__(
        self,
        half,
        width,
        height,
        interpolateMethod,
        ensemble=False,
        interpolateFactor=2,
        dynamicScale=False,
        staticStep=False,
        compileMode: str = "default",
    ):
        self.half = half
        self.scale = 1.0
        self.width = width
        self.height = height
        self.interpolateMethod = interpolateMethod
        self.baseMethod = interpolateMethod.replace("-mps", "")
        self.ensemble = ensemble
        self.interpolateFactor = interpolateFactor
        self.dynamicScale = dynamicScale
        self.staticStep = staticStep
        self.compileMode = compileMode
        self.device = torch.device("mps")

        if self.width > 1920 and self.height > 1080:
            self.scale = 0.5
            if self.half:
                logAndPrint(
                    "UHD and fp16 are not compatible with RIFE, defaulting to fp32",
                    "yellow",
                )
                self.half = False

        self.handleModel()

    def handleModel(self):
        if ADOBE:
            progressState.update(
                {"status": f"Loading MPS RIFE model: {self.interpolateMethod}..."}
            )

        self.filename = modelsMap(self.baseMethod)
        modelPath = resolveWeightPath(
            "rife", self.filename, downloadModel=self.baseMethod
        )

        self.dType = torch.float16 if self.half else torch.float32

        IFNet = importRifeArch(self.baseMethod, "v1")

        if self.baseMethod in ["rife_elexor"] and self.staticStep:
            self.staticStep = False
            logAndPrint(
                "Static step is not supported for rife_elexor, automatically disabling it",
                "yellow",
            )
        if (
            self.baseMethod not in ["rife4.6", "rife4.15", "rife4.15-lite"]
            and self.staticStep
        ):
            self.staticStep = False
            logAndPrint(
                "Static step is not supported for this interpolation model yet, automatically disabling it",
                "yellow",
            )

        if self.baseMethod in ["rife_elexor"]:
            self.model = IFNet(
                self.scale,
                self.ensemble,
                self.dType,
                self.device,
                self.width,
                self.height,
                self.interpolateFactor,
            )
        else:
            if self.baseMethod in ["rife4.6", "rife4.15", "rife4.15-lite"]:
                self.model = IFNet(
                    self.ensemble,
                    self.dynamicScale,
                    self.scale,
                    self.interpolateFactor,
                    self.staticStep,
                )
            else:
                self.model = IFNet(
                    self.ensemble,
                    self.dynamicScale,
                    self.scale,
                    self.interpolateFactor,
                )

        stateDict = torch.load(modelPath, map_location="cpu")
        self.model.load_state_dict(stateDict)
        del stateDict

        if self.half:
            self.model = self.model.half()
        else:
            self.model = self.model.float()

        self.model = self.model.eval()
        self.model = self.model.to(self.device)
        self.model = self.model.to(memory_format=torch.channels_last)

        if self.compileMode != "default":
            try:
                if self.compileMode == "max":
                    self.model.compile(mode="max-autotune-no-cudagraphs")
                elif self.compileMode == "max-graphs":
                    self.model.compile(
                        mode="max-autotune-no-cudagraphs", fullgraph=True
                    )
            except Exception as e:
                logging.error(
                    f"Error compiling MPS RIFE model {self.interpolateMethod} "
                    f"with mode {self.compileMode}: {e}"
                )
                logAndPrint(
                    f"Error compiling MPS RIFE model {self.interpolateMethod} "
                    f"with mode {self.compileMode}: {e}",
                    "red",
                )
            self.compileMode = "default"

        mul = _padMultiple(self.baseMethod, self.scale, self.dynamicScale)
        ph = ((self.height - 1) // mul + 1) * mul
        pw = ((self.width - 1) // mul + 1) * mul
        self.padding = (0, pw - self.width, 0, ph - self.height)

        self.I0 = torch.zeros(
            1,
            3,
            self.height + self.padding[3],
            self.width + self.padding[1],
            dtype=self.dType,
            device=self.device,
        ).to(memory_format=torch.channels_last)

        self.I1 = torch.zeros(
            1,
            3,
            self.height + self.padding[3],
            self.width + self.padding[1],
            dtype=self.dType,
            device=self.device,
        ).to(memory_format=torch.channels_last)

        self.firstRun = True

        self._timestep_buffer = torch.zeros(
            (1, 1, self.height + self.padding[3], self.width + self.padding[1]),
            dtype=self.dType,
            device=self.device,
        )
        self._cachedTimestepValue = None

    @torch.inference_mode()
    def cacheFrameReset(self, frame):
        # Scene-cut reset: anchor I0 = frame and re-seed the encoder feature
        # f0 = encode(frame). MPS runs eager (no CUDA graph), so the arch's
        # cacheReset reassigning f0 is fine. firstRun stays False.
        self.processFrame(frame, "I0")
        self.processFrame(self.I0, "model")

    @torch.inference_mode()
    def processFrame(self, frame, toNorm):
        match toNorm:
            case "I0":
                frame = frame.to(device=self.device, dtype=self.dType)
                frame = self.padFrame(frame)
                self.I0.copy_(frame)

            case "I1":
                frame = frame.to(device=self.device, dtype=self.dType)
                frame = self.padFrame(frame)
                self.I1.copy_(frame)

            case "cache":
                self.I0.copy_(self.I1)
                self.model.cache()

            case "infer":
                if self.staticStep:
                    output = self.model(self.I0, self.I1, frame).clone()
                else:
                    output = self.model(self.I0, self.I1, frame)[
                        :, :, : self.height, : self.width
                    ].clone()
                # `output.cpu()` will wait for the pending MPS work to finish.
                return output.contiguous().cpu()

            case "model":
                self.model.cacheReset(frame)

    @torch.inference_mode()
    def padFrame(self, frame):
        return (
            F.pad(frame, [0, self.padding[1], 0, self.padding[3]])
            if self.padding != (0, 0, 0, 0)
            else frame
        )

    @torch.inference_mode()
    def __call__(self, frame, interpQueue, framesToInsert: int = 2, timesteps=None):
        if self.firstRun:
            self.processFrame(frame, "I0")
            self.firstRun = False
            return
        self.processFrame(frame, "I1")

        for i in range(framesToInsert):
            t = interpolateTimestep(i, framesToInsert, timesteps)
            self._cachedTimestepValue = fillTimestepBuffer(
                self._timestep_buffer, self._cachedTimestepValue, t
            )
            output = self.processFrame(self._timestep_buffer, "infer")
            interpQueue.put(output)

        self.processFrame(None, "cache")
