"""The `-kernel` backend: models run on the tas-kernels package's hand-written CUDA kernels.

tas-kernels (PyPI, compiled wheels for Windows and Linux) covers four architectures:
SRVGG Compact (24 or 64 features), SPAN, Real-CUGAN 2x / ShuffleCUGAN and RealPLKSR.
A `-kernel` method reads the same `.pth`/`.safetensors` weights as its CUDA method and
hands the state dict to the matching tas_kernels class. It needs an NVIDIA RTX 30
series GPU or newer; `kernelUnavailableReason()` says why it cannot run, and the
factories then build the CUDA method instead.

`loadKernelModel` is shared with `src/unifiedRestore.py:UnifiedRestoreKernel`.
"""

import importlib
import logging
import os

import torch

from src.constants import ADOBE
from src.infra.cudaStreams import waitForCaller
from src.model.download import resolveWeightPath
from src.model.registry import modelsMap

if ADOBE:
    from src.server.aeComms import progressState

KERNEL_SUFFIX = "-kernel"

# Keys a checkpoint may nest its state dict under.
_STATE_DICT_KEYS = ("params_ema", "params", "state_dict", "model")


def kernelUnavailableReason() -> str | None:
    """Why the `-kernel` backend cannot run on this machine, or None when it can."""
    try:
        # Imported by name: the wheel holds compiled modules only, with no stubs.
        tas_kernels = importlib.import_module("tas_kernels")
    except ImportError as error:
        return (
            f"the tas-kernels package is not installed ({error}); it comes with the "
            "windows-cuda and linux-cuda dependency profiles"
        )
    return tas_kernels.unsupportedReason()


def readStateDict(modelPath: str, trusted: bool) -> dict:
    """The state dict stored at ``modelPath``.

    ``trusted`` is for TAS's own weights, some of which are pickled modules; a
    custom model is only ever read as tensors.
    """
    if modelPath.lower().endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(modelPath)

    try:
        loaded = torch.load(modelPath, map_location="cpu", weights_only=True)
    except Exception as error:
        if not trusted:
            raise ValueError(
                f"The -kernel backend cannot read {os.path.basename(modelPath)} as "
                "plain weights (a pickled or TorchScript model is not accepted for "
                "a custom file); use the CUDA method for it."
            ) from error
        loaded = torch.load(modelPath, map_location="cpu", weights_only=False)
    if not isinstance(loaded, dict):
        return loaded.state_dict()
    for key in _STATE_DICT_KEYS:
        if isinstance(loaded.get(key), dict):
            return loaded[key]
    return loaded


def requiredMultiple(stateDict: dict) -> int:
    """The multiple both input dimensions must have for ``stateDict``'s architecture.

    Only the Real-CUGAN family has one (ShuffleCUGAN 4, CUGAN 2); the other
    kernel architectures take any size.
    """
    cugan = importlib.import_module("tas_kernels.cugan")
    try:
        return 4 if cugan.archOf(stateDict) == "ShuffleCUGAN" else 2
    except Exception:
        return 1


def loadKernelModel(stateDict: dict, height: int, width: int, half: bool):
    """The tas_kernels model for ``stateDict`` at ``height`` x ``width``.

    Each class refuses a state dict of another architecture with ValueError, so
    they are tried in turn. Raises ValueError when none takes it.
    """
    tas_kernels = importlib.import_module("tas_kernels")
    refusals = []
    for className in ("CuganModel", "SpanModel", "PlksrModel", "CompactModel"):
        try:
            return getattr(tas_kernels, className)(stateDict, height, width, half=half)
        except ValueError as error:
            refusals.append(str(error))
    # "not a <arch> state dict" is every class's answer to a foreign file; anything
    # else is the matching class saying which variant it has no kernels for.
    specific = [text for text in refusals if not text.startswith("not a ")]
    detail = (
        "; ".join(specific)
        if specific
        else ("it is not a Compact, SPAN, Real-CUGAN or RealPLKSR model")
    )
    raise ValueError(f"The -kernel backend cannot run this model: {detail}.")


class UniversalKernel:
    def __init__(
        self,
        upscaleMethod: str = "shufflecugan-kernel",
        upscaleFactor: int = 2,
        half: bool = False,
        width: int = 1920,
        height: int = 1080,
        customModel: str | None = None,
    ):
        """
        Upscale with the tas-kernels CUDA kernels.

        Args:
            upscaleMethod (str): The `-kernel` method to run
            upscaleFactor (int): The factor the stage upscales by
            half (bool): fp16 frames when true, fp32 otherwise
            width (int): The width of the input frame
            height (int): The height of the input frame
            customModel (str): The path to a custom model file
        """
        self.upscaleMethod = upscaleMethod
        self.upscaleFactor = upscaleFactor
        self.half = half
        self.width = width
        self.height = height
        self.customModel = customModel

        self.handleModel()

    def handleModel(self):
        if ADOBE:
            progressState.update(
                {"status": f"Loading upscale model: {self.upscaleMethod}..."}
            )

        baseMethod = self.upscaleMethod.removesuffix(KERNEL_SUFFIX)
        if self.customModel:
            if not os.path.isfile(self.customModel):
                raise FileNotFoundError(
                    f"Custom model file {self.customModel} not found"
                )
            modelPath = self.customModel
        else:
            self.filename = modelsMap(baseMethod, self.upscaleFactor, modelType="pth")
            modelPath = resolveWeightPath(
                baseMethod, self.filename, upscaleFactor=self.upscaleFactor
            )

        stateDict = readStateDict(modelPath, trusted=not self.customModel)

        from src.upscale._shared import calculatePadding, customScaleRefusal

        # The Real-CUGAN family needs both dimensions divisible by 4 (or 2): the
        # frame is reflect-padded to it and the surplus cropped off the output,
        # as UniversalPytorch does.
        self.requiredMultiple = requiredMultiple(stateDict)
        self.padding = calculatePadding(self.width, self.height, self.requiredMultiple)
        self.paddedWidth = self.width + self.padding[1]
        self.paddedHeight = self.height + self.padding[3]

        self.model = loadKernelModel(
            stateDict, self.paddedHeight, self.paddedWidth, self.half
        )
        del stateDict

        self.dummyInput = torch.zeros(
            (1, 3, self.paddedHeight, self.paddedWidth),
            device="cuda",
            dtype=torch.float16 if self.half else torch.float32,
        )

        self.stream = torch.cuda.Stream()
        waitForCaller(self.stream)
        with torch.cuda.stream(self.stream):
            warmup = self.model(self.dummyInput)
        self.stream.synchronize()

        modelScale = warmup.shape[-1] // self.paddedWidth
        if self.customModel:
            # A 1x model passes, as on the PyTorch backend: its frame is resized
            # to the stage's size downstream.
            problem = (
                None
                if modelScale == 1
                else customScaleRefusal(modelPath, modelScale, self.upscaleFactor)
            )
            if problem is not None:
                raise ValueError(problem)
        self.modelScale = modelScale
        logging.info(
            f"{self.upscaleMethod}: {type(self.model).__name__} at "
            f"{self.paddedWidth}x{self.paddedHeight}, {modelScale}x"
        )

    @torch.inference_mode()
    def __call__(self, frame: torch.Tensor, nextFrame=None) -> torch.Tensor:
        waitForCaller(self.stream)
        with torch.cuda.stream(self.stream):
            if self.padding[1] or self.padding[3]:
                frame = torch.nn.functional.pad(frame, self.padding, mode="reflect")
            self.dummyInput.copy_(frame, non_blocking=True)
            # The model returns its own output buffer, rewritten by the next call.
            output = self.model(self.dummyInput).clone()
            # Crop before the sync: the writer never waits on this stream.
            if self.padding[1] or self.padding[3]:
                output = output[
                    :,
                    :,
                    : self.height * self.modelScale,
                    : self.width * self.modelScale,
                ].contiguous()
        self.stream.synchronize()

        return output
