import contextlib
import logging
import os

import torch

from src.constants import ADOBE
from src.infra.cudaStreams import waitForCaller
from src.infra.isCudaInit import CudaChecker
from src.infra.logAndPrint import logAndPrint
from src.model.download import resolveWeightPath
from src.model.modelOptimizer import ModelOptimizer
from src.model.registry import modelsMap

if ADOBE:
    from src.server.aeComms import progressState

checker = CudaChecker()

torch.set_float32_matmul_precision("medium")


def _loadUpscaleArch(method, upscaleFactor, modelPath, customModel):
    """Build ``method``'s arch from ``modelPath`` and return it in eval mode."""
    from src.spandrelCompat import ImageModelDescriptor, loadSpandrelModel

    if method == "saryn":
        from src.extraArches.RTMoSR import RTMoSR

        model = RTMoSR()
        stateDict = torch.load(modelPath, map_location="cpu")
        model.load_state_dict(stateDict)
        del stateDict
    elif method == "figsr":
        from src.extraArches.figsr import FIGSR

        stateDict = torch.load(modelPath, map_location="cpu", weights_only=False)
        model = FIGSR(scale=upscaleFactor, dim=32)
        model.load_state_dict(stateDict, strict=False)
        del stateDict
    elif method == "smosr":
        # SMoSR is a registered spandrel arch; spandrel reads the
        # .safetensors, auto-detects dim/depth/rep, and reparameterizes
        # (fuses) the rep branches when .eval() runs below.
        model = loadSpandrelModel(modelPath)
        if not isinstance(model, ImageModelDescriptor):
            raise TypeError(
                f"SMoSR model {modelPath} did not resolve to an image model descriptor"
            )
        model = model.model
    elif method == "gauss":
        from safetensors.torch import load_file

        from src.extraArches.DIS import DIS

        model = DIS(scale=2, num_features=32, num_blocks=12)
        stateDict = load_file(modelPath)
        model.load_state_dict(stateDict)
        del stateDict
    else:
        if customModel:
            model = loadSpandrelModel(modelPath)
            if not isinstance(model, ImageModelDescriptor):
                raise TypeError(
                    f"Custom upscale model {modelPath} did not resolve to an image model descriptor"
                )
            from src.upscale._shared import customScaleRefusal

            # A 1x model passes, as it always has: its frame is resized to the
            # stage's size downstream (the writer's fallback when it is last).
            problem = (
                None
                if model.scale == 1
                else customScaleRefusal(modelPath, model.scale, upscaleFactor)
            )
            if problem is not None:
                raise ValueError(problem)
        else:
            model = torch.load(modelPath, map_location="cpu", weights_only=False)
            if isinstance(model, dict):
                model = loadSpandrelModel(modelPath, model)

        try:
            # SPANDREL HAXX
            model = model.model
        except Exception:
            pass

    return model.eval()


def _torchProbe(model, device, dtype, channelsLast):
    """``smallestValidMultiple``'s runOK: one tiny forward of ``model``."""

    def runOK(h, w):
        try:
            probe = torch.zeros((1, 3, h, w), device=device, dtype=dtype)
            if channelsLast:
                probe = probe.contiguous().to(memory_format=torch.channels_last)
            # no_grad, not inference_mode: archs that lazily reparameterize on
            # first forward (SPAN's eval_conv) would otherwise cache an
            # inference-mode tensor that the grad-tracked warmup cannot reuse.
            with torch.no_grad():
                model(probe)
            return True
        except Exception:
            return False

    return runOK


class UniversalPytorch:
    def __init__(
        self,
        upscaleMethod: str = "shufflecugan",
        upscaleFactor: int = 2,
        half: bool = False,
        width: int = 1920,
        height: int = 1080,
        customModel: str = None,
        compileMode: str = "default",
    ):
        """
        Initialize the upscaler with the desired model

        Args:
            upscaleMethod (str): The method to use for upscaling
            upscaleFactor (int): The factor to upscale by
            half (bool): Whether to use half precision
            width (int): The width of the input frame
            height (int): The height of the input frame
            customModel (str): The path to a custom model file
            compileMode: (str): The compile mode to use for the model
        """
        self.upscaleMethod = upscaleMethod
        self.upscaleFactor = upscaleFactor
        self.half = half
        self.width = width
        self.height = height
        self.customModel = customModel
        self.compileMode: str = compileMode

        self.handleModel()

    def handleModel(self):
        """
        Load the desired model
        """
        if ADOBE:
            progressState.update(
                {"status": f"Loading upscale model: {self.upscaleMethod}..."}
            )

        from src.spandrelCompat import UnsupportedDtypeError

        if not self.customModel:
            self.filename = modelsMap(
                self.upscaleMethod, self.upscaleFactor, modelType="pth"
            )
            modelPath = resolveWeightPath(
                self.upscaleMethod,
                self.filename,
                upscaleFactor=self.upscaleFactor,
            )
        else:
            if os.path.isfile(self.customModel):
                modelPath = self.customModel
            else:
                raise FileNotFoundError(
                    f"Custom model file {self.customModel} not found"
                )

        self.model = _loadUpscaleArch(
            self.upscaleMethod, self.upscaleFactor, modelPath, self.customModel
        )

        if self.half and checker.cudaAvailable:
            try:
                self.model = self.model.half()
            except UnsupportedDtypeError as e:
                logging.error(f"Model does not support half precision: {e}")
                self.model = self.model.float()
                self.half = False
            except Exception as e:
                logging.error(f"Error converting model to half precision: {e}")
                self.model = self.model.float()
                self.half = False

        if checker.cudaAvailable:
            self.model = self.model.cuda()
            torch.cuda.empty_cache()

        self.model = ModelOptimizer(
            self.model,
            torch.float16 if self.half else torch.float32,
            memoryFormat=torch.channels_last,
        ).optimizeModel()

        # Resolve the arch's required input multiple BEFORE compiling: for
        # unknown archs the fallback probe runs 1-3 tiny forwards, and doing
        # that after model.compile() would trigger a needless (and at
        # max-autotune, very slow) compilation per probe shape.
        # RealCUGAN-family archs (adore, fallin_*, shufflecugan) need both
        # input dims divisible by 4; feeding an odd dim raises a skip-Add size
        # mismatch, so the input is reflect-padded to the multiple and the
        # surplus cropped off the output. Fully-convolutional archs resolve to
        # 1, so padding stays zero and the output is bit-identical.
        from src.upscale._shared import calculatePadding, resolveRequiredMultiple

        self.requiredMultiple = resolveRequiredMultiple(
            self.upscaleMethod,
            self.customModel,
            lambda: _torchProbe(
                self.model,
                checker.device,
                torch.float16 if self.half else torch.float32,
                channelsLast=True,
            ),
        )

        self.padding = calculatePadding(self.width, self.height, self.requiredMultiple)
        self.paddedWidth = self.width + self.padding[1]
        self.paddedHeight = self.height + self.padding[3]

        if self.compileMode != "default":
            try:
                if self.compileMode == "max":
                    self.model.compile(mode="max-autotune")
                elif self.compileMode == "max-graphs":
                    self.model.compile(
                        mode="max-autotune-no-cudagraphs", fullgraph=True
                    )
            except Exception as e:
                logging.error(
                    f"Error compiling model {self.upscaleMethod} with mode {self.compileMode}: {e}"
                )
                logAndPrint(
                    f"Error compiling model {self.upscaleMethod} with mode {self.compileMode}: {e}",
                    "red",
                )

            self.compileMode = "default"

        self.dummyInput = (
            torch.zeros(
                (1, 3, self.paddedHeight, self.paddedWidth),
                device=checker.device,
                dtype=torch.float16 if self.half else torch.float32,
            )
            .contiguous()
            .to(memory_format=torch.channels_last)
        )

        self.dummyOutput = (
            torch.zeros(
                (
                    1,
                    3,
                    self.paddedHeight * self.upscaleFactor,
                    self.paddedWidth * self.upscaleFactor,
                ),
                device=checker.device,
                dtype=torch.float16 if self.half else torch.float32,
            )
            .contiguous()
            .to(memory_format=torch.channels_last)
        )

        self.stream = torch.cuda.Stream()

        waitForCaller(self.stream)
        # no_grad, not inference_mode (see _torchProbe): with grad
        # on, every warmup forward recorded an autograd graph holding all of
        # its activations (span at 714p fp16: 4.6 GiB peak instead of 1.7).
        # TorchScript models (rtmosr) keep the caller's grad mode, as before:
        # the profiling executor specializes on grad mode, and a no_grad
        # warmup lets the fuser pick different kernels, changing the output.
        scripted = any(
            isinstance(m, torch.jit.ScriptModule) for m in self.model.modules()
        )
        gradMode = contextlib.nullcontext() if scripted else torch.no_grad()
        with torch.cuda.stream(self.stream), gradMode:
            # 3 warmup iters before CUDA-graph capture: cudnn.benchmark is False
            # (no algo autotune), so 3 is sufficient to JIT cudnn kernels +
            # allocate workspace before capture (PyTorch-documented minimum).
            for _ in range(3):
                self.model(self.dummyInput)
                self.stream.synchronize()

        self.normStream = torch.cuda.Stream()
        self.outputStream = torch.cuda.Stream()

        # NOTE: the unconditional `self.compileMode = "default"` reset above
        # means a successful --compile still lands here with "default", so the
        # outer CUDA graph is always captured (for mode "max" that stacks
        # inductor cudagraphs inside the outer graph -- cf. misc.py which uses
        # max-autotune-no-cudagraphs to avoid exactly this). Resetting only on
        # compile failure would flip __call__ onto the compiled-eager branch;
        # that ownership change needs a warm-GPU before/after FPS/VRAM/parity
        # comparison first, so it is deliberately not done here.
        if self.compileMode == "default":
            self.cudaGraph = torch.cuda.CUDAGraph()
            self.initTorchCudaGraph()

    @torch.inference_mode()
    def initTorchCudaGraph(self):
        with torch.cuda.graph(self.cudaGraph, stream=self.stream):
            self.dummyOutput = self.model(self.dummyInput)
        self.stream.synchronize()

    @torch.inference_mode()
    def processFrame(self, frame):
        waitForCaller(self.normStream)
        with torch.cuda.stream(self.normStream):
            if self.padding[1] or self.padding[3]:
                frame = torch.nn.functional.pad(frame, self.padding, mode="reflect")
            self.dummyInput.copy_(
                frame.to(dtype=self.dummyInput.dtype),
                non_blocking=True,
            )
        self.normStream.synchronize()

    @torch.inference_mode()
    def __call__(self, frame: torch.tensor, nextFrame: None) -> torch.tensor:
        self.processFrame(frame)
        waitForCaller(self.stream)
        with torch.cuda.stream(self.stream):
            if self.compileMode == "default":
                self.cudaGraph.replay()
            else:
                self.dummyOutput.copy_(
                    self.model(self.dummyInput),
                    non_blocking=True,
                )
            output = self.dummyOutput.clone()
            # Crop before the sync: the writer copies on its own stream and
            # never waits on ours, so a crop queued after it could be read
            # before it ran.
            if self.padding[1] or self.padding[3]:
                output = output[
                    :,
                    :,
                    : self.height * self.upscaleFactor,
                    : self.width * self.upscaleFactor,
                ].contiguous()
        self.stream.synchronize()

        return output


class UniversalPytorchMPS:
    """
    Apple Silicon (MPS) PyTorch upscaler. Mirrors UniversalPytorch but without
    CUDA streams / CUDA graphs, since MPS does not support them. Shares .pth
    weights with the CUDA path — the "-mps" suffix on upscaleMethod is stripped
    before resolving model filenames.
    """

    def __init__(
        self,
        upscaleMethod: str = "shufflecugan-mps",
        upscaleFactor: int = 2,
        half: bool = False,
        width: int = 1920,
        height: int = 1080,
        customModel: str = None,
        compileMode: str = "default",
    ):
        self.upscaleMethod = upscaleMethod
        self.baseMethod = upscaleMethod.replace("-mps", "")
        self.upscaleFactor = upscaleFactor
        self.half = half
        self.width = width
        self.height = height
        self.customModel = customModel
        self.compileMode = compileMode
        self.device = torch.device("mps")
        self.dtype = torch.float16 if self.half else torch.float32

        self.handleModel()

    def handleModel(self):
        if ADOBE:
            progressState.update(
                {"status": f"Loading MPS upscale model: {self.upscaleMethod}..."}
            )

        from src.spandrelCompat import UnsupportedDtypeError

        if not self.customModel:
            self.filename = modelsMap(
                self.baseMethod, self.upscaleFactor, modelType="pth"
            )
            modelPath = resolveWeightPath(
                self.baseMethod,
                self.filename,
                upscaleFactor=self.upscaleFactor,
            )
        else:
            if not os.path.isfile(self.customModel):
                raise FileNotFoundError(
                    f"Custom model file {self.customModel} not found"
                )
            modelPath = self.customModel

        self.model = _loadUpscaleArch(
            self.baseMethod, self.upscaleFactor, modelPath, self.customModel
        )

        if self.half:
            try:
                self.model = self.model.half()
            except UnsupportedDtypeError as e:
                logging.error(f"Model does not support half precision on MPS: {e}")
                self.model = self.model.float()
                self.half = False
                self.dtype = torch.float32
            except Exception as e:
                logging.error(f"Error converting model to half precision on MPS: {e}")
                self.model = self.model.float()
                self.half = False
                self.dtype = torch.float32

        self.model = self.model.to(self.device)

        if self.compileMode != "default":
            # torch.compile on MPS is unstable as of torch 2.11 — warn and skip.
            logAndPrint(
                f"compileMode '{self.compileMode}' ignored on MPS backend (unsupported).",
                "yellow",
            )
            self.compileMode = "default"

        # RealCUGAN-family archs (fallin_*, shufflecugan, aniscale2) need both
        # input dims divisible by 4; detect the requirement and reflect-pad to it,
        # cropping the surplus off the output. Fully-conv archs report 1 (no pad).
        from src.upscale._shared import calculatePadding, resolveRequiredMultiple

        self.requiredMultiple = resolveRequiredMultiple(
            self.baseMethod,
            self.customModel,
            lambda: _torchProbe(
                self.model, self.device, self.dtype, channelsLast=False
            ),
        )

        self.padding = calculatePadding(self.width, self.height, self.requiredMultiple)
        self.paddedWidth = self.width + self.padding[1]
        self.paddedHeight = self.height + self.padding[3]

        # Warm-up to force MPS kernel compilation before the first real frame.
        with torch.inference_mode():
            dummy = torch.zeros(
                (1, 3, self.paddedHeight, self.paddedWidth),
                device=self.device,
                dtype=self.dtype,
            )
            for _ in range(2):
                _ = self.model(dummy)
            torch.mps.synchronize()

    @torch.inference_mode()
    def __call__(self, frame: torch.Tensor, nextFrame=None) -> torch.Tensor:
        if frame.device != self.device or frame.dtype != self.dtype:
            frame = frame.to(device=self.device, dtype=self.dtype)
        if self.padding[1] or self.padding[3]:
            frame = torch.nn.functional.pad(frame, self.padding, mode="reflect")
        output = self.model(frame)
        torch.mps.synchronize()
        if self.padding[1] or self.padding[3]:
            output = output[
                :,
                :,
                : self.height * self.upscaleFactor,
                : self.width * self.upscaleFactor,
            ]
        return output.cpu()


class UniversalPytorchROCm:
    """
    AMD ROCm (HIP) PyTorch upscaler. Eager, no CUDA graphs or custom streams:
    HIP graphs are less mature than CUDA graphs across the RDNA/CDNA range,
    so this mirrors UniversalPytorchMPS's eager path but stays on the
    torch.cuda (HIP) device and returns GPU tensors like the CUDA path.

    Shares .pth weights with the CUDA path — the "-rocm" suffix is stripped
    before resolving model filenames.
    """

    def __init__(
        self,
        upscaleMethod: str = "shufflecugan-rocm",
        upscaleFactor: int = 2,
        half: bool = False,
        width: int = 1920,
        height: int = 1080,
        customModel: str = None,
        compileMode: str = "default",
    ):
        self.upscaleMethod = upscaleMethod
        self.baseMethod = upscaleMethod.replace("-rocm", "")
        self.upscaleFactor = upscaleFactor
        self.half = half
        self.width = width
        self.height = height
        self.customModel = customModel
        self.compileMode = compileMode
        self.device = checker.device
        self.dtype = torch.float16 if self.half else torch.float32

        self.handleModel()

    def handleModel(self):
        if ADOBE:
            progressState.update(
                {"status": f"Loading ROCm upscale model: {self.upscaleMethod}..."}
            )

        from src.spandrelCompat import UnsupportedDtypeError

        if not self.customModel:
            self.filename = modelsMap(
                self.baseMethod, self.upscaleFactor, modelType="pth"
            )
            modelPath = resolveWeightPath(
                self.baseMethod,
                self.filename,
                upscaleFactor=self.upscaleFactor,
            )
        else:
            if not os.path.isfile(self.customModel):
                raise FileNotFoundError(
                    f"Custom model file {self.customModel} not found"
                )
            modelPath = self.customModel

        self.model = _loadUpscaleArch(
            self.baseMethod, self.upscaleFactor, modelPath, self.customModel
        )

        if self.half:
            try:
                self.model = self.model.half()
            except UnsupportedDtypeError as e:
                logging.error(f"Model does not support half precision on ROCm: {e}")
                self.model = self.model.float()
                self.half = False
                self.dtype = torch.float32
            except Exception as e:
                logging.error(f"Error converting model to half precision on ROCm: {e}")
                self.model = self.model.float()
                self.half = False
                self.dtype = torch.float32

        self.model = self.model.to(self.device)

        self.model = ModelOptimizer(
            self.model,
            torch.float16 if self.half else torch.float32,
            memoryFormat=torch.channels_last,
        ).optimizeModel()

        if self.compileMode != "default":
            logAndPrint(
                f"compileMode '{self.compileMode}' ignored on ROCm backend (unsupported).",
                "yellow",
            )
            self.compileMode = "default"

        from src.upscale._shared import calculatePadding, resolveRequiredMultiple

        self.requiredMultiple = resolveRequiredMultiple(
            self.baseMethod,
            self.customModel,
            lambda: _torchProbe(self.model, self.device, self.dtype, channelsLast=True),
        )

        self.padding = calculatePadding(self.width, self.height, self.requiredMultiple)

        with torch.inference_mode():
            dummy = (
                torch.zeros(
                    (
                        1,
                        3,
                        self.height + self.padding[3],
                        self.width + self.padding[1],
                    ),
                    device=self.device,
                    dtype=self.dtype,
                )
                .contiguous()
                .to(memory_format=torch.channels_last)
            )
            for _ in range(2):
                _ = self.model(dummy)
            if self.device.type == "cuda":
                torch.cuda.synchronize()

    @torch.inference_mode()
    def __call__(self, frame: torch.Tensor, nextFrame=None) -> torch.Tensor:
        if frame.device != self.device or frame.dtype != self.dtype:
            frame = frame.to(device=self.device, dtype=self.dtype)
            frame = frame.to(memory_format=torch.channels_last)
        if self.padding[1] or self.padding[3]:
            frame = torch.nn.functional.pad(frame, self.padding, mode="reflect")
        output = self.model(frame)
        if self.padding[1] or self.padding[3]:
            output = output[
                :,
                :,
                : self.height * self.upscaleFactor,
                : self.width * self.upscaleFactor,
            ].contiguous()
        # Sync after the crop: the writer copies on its own stream and never
        # waits on ours.
        if self.device.type == "cuda":
            torch.cuda.current_stream().synchronize()
        return output
