import logging
import os

import torch

from src.constants import ADOBE
from src.infra.isCudaInit import CudaChecker
from src.infra.logAndPrint import logAndPrint
from src.model.download import resolveWeightPath
from src.model.modelOptimizer import ModelOptimizer
from src.model.registry import modelsMap

if ADOBE:
    from src.server.aeComms import progressState

checker = CudaChecker()

torch.set_float32_matmul_precision("medium")


def calculatePadding(width, height, multiple=4):
    padW = (multiple - (width % multiple)) % multiple
    padH = (multiple - (height % multiple)) % multiple
    return (0, padW, 0, padH)


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

        from src.spandrelCompat import (
            ImageModelDescriptor,
            ModelLoader,
            UnsupportedDtypeError,
        )

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

        if self.upscaleMethod == "saryn":
            from src.extraArches.RTMoSR import RTMoSR

            self.model = RTMoSR()
            stateDict = torch.load(modelPath, map_location="cpu")
            self.model.load_state_dict(stateDict)
            del stateDict
        elif self.upscaleMethod == "figsr":
            from src.extraArches.figsr import FIGSR

            stateDict = torch.load(modelPath, map_location="cpu", weights_only=False)
            self.model = FIGSR(scale=self.upscaleFactor, dim=32)

            self.model.load_state_dict(stateDict, strict=False)
            del stateDict
        elif self.upscaleMethod == "smosr":
            # SMoSR is a registered spandrel arch; ModelLoader reads the
            # .safetensors, auto-detects dim/depth/rep, and reparameterizes
            # (fuses) the rep branches when .eval() runs below.
            self.model = ModelLoader().load_from_file(modelPath)
            if not isinstance(self.model, ImageModelDescriptor):
                raise TypeError(
                    f"SMoSR model {modelPath} did not resolve to an image model descriptor"
                )
            self.model = self.model.model
        elif self.upscaleMethod == "gauss":
            from safetensors.torch import load_file

            from src.extraArches.DIS import DIS

            self.model = DIS(scale=2, num_features=32, num_blocks=12)
            stateDict = load_file(modelPath)
            self.model.load_state_dict(stateDict)
            del stateDict
        else:
            if self.customModel:
                self.model = ModelLoader().load_from_file(modelPath)
                if not isinstance(self.model, ImageModelDescriptor):
                    raise TypeError(
                        f"Custom upscale model {modelPath} did not resolve to an image model descriptor"
                    )
            else:
                self.model = torch.load(
                    modelPath, map_location="cpu", weights_only=False
                )

                if isinstance(self.model, dict):
                    self.model = ModelLoader().load_from_state_dict(self.model)

            if self.customModel:
                assert isinstance(self.model, ImageModelDescriptor)

            try:
                # SPANDREL HAXX
                self.model = self.model.model
            except Exception:
                pass

        self.model = self.model.eval()

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

        # RealCUGAN-family archs (fallin_*, shufflecugan, aniscale2) need both
        # input dims divisible by 4; feeding an odd dim raises a skip-Add size
        # mismatch. Detect the arch's requirement once and reflect-pad the input
        # to it, cropping the surplus off the output. Fully-convolutional archs
        # report 1 here, so padding stays zero and the output is bit-identical.
        self.requiredMultiple = self._detectRequiredMultiple()
        self.padding = calculatePadding(self.width, self.height, self.requiredMultiple)
        self.paddedWidth = self.width + self.padding[1]
        self.paddedHeight = self.height + self.padding[3]

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

        with torch.cuda.stream(self.stream):
            # 3 warmup iters before CUDA-graph capture: cudnn.benchmark is False
            # (no algo autotune), so 3 is sufficient to JIT cudnn kernels +
            # allocate workspace before capture (PyTorch-documented minimum).
            for _ in range(3):
                self.model(self.dummyInput)
                self.stream.synchronize()

        self.normStream = torch.cuda.Stream()
        self.outputStream = torch.cuda.Stream()

        if not self.compileMode != "default":
            self.cudaGraph = torch.cuda.CUDAGraph()
            self.initTorchCudaGraph()

    def _detectRequiredMultiple(self) -> int:
        """Probe the model on tiny inputs to learn its required spatial multiple."""
        from src.upscale._shared import smallestValidMultiple

        dtype = torch.float16 if self.half else torch.float32

        def runOK(h, w):
            try:
                probe = (
                    torch.zeros((1, 3, h, w), device=checker.device, dtype=dtype)
                    .contiguous()
                    .to(memory_format=torch.channels_last)
                )
                # no_grad, not inference_mode: archs that lazily reparameterize on
                # first forward (SPAN's eval_conv) would otherwise cache an
                # inference-mode tensor that the grad-tracked warmup cannot reuse.
                with torch.no_grad():
                    self.model(probe)
                return True
            except Exception:
                return False

        return smallestValidMultiple(runOK)

    @torch.inference_mode()
    def initTorchCudaGraph(self):
        with torch.cuda.graph(self.cudaGraph, stream=self.stream):
            self.dummyOutput = self.model(self.dummyInput)
        self.stream.synchronize()

    @torch.inference_mode()
    def processFrame(self, frame):
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
        with torch.cuda.stream(self.stream):
            if self.compileMode == "default":
                self.cudaGraph.replay()
            else:
                self.dummyOutput.copy_(
                    self.model(self.dummyInput),
                    non_blocking=True,
                )
            output = self.dummyOutput.clone()
        self.stream.synchronize()

        if self.padding[1] or self.padding[3]:
            output = output[
                :,
                :,
                : self.height * self.upscaleFactor,
                : self.width * self.upscaleFactor,
            ].contiguous()

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

        from src.spandrelCompat import (
            ImageModelDescriptor,
            ModelLoader,
            UnsupportedDtypeError,
        )

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

        if self.baseMethod == "saryn":
            from src.extraArches.RTMoSR import RTMoSR

            self.model = RTMoSR()
            stateDict = torch.load(modelPath, map_location="cpu")
            self.model.load_state_dict(stateDict)
            del stateDict
        elif self.baseMethod == "figsr":
            from src.extraArches.figsr import FIGSR

            stateDict = torch.load(modelPath, map_location="cpu", weights_only=False)
            self.model = FIGSR(scale=self.upscaleFactor, dim=32)
            self.model.load_state_dict(stateDict, strict=False)
            del stateDict
        elif self.baseMethod == "smosr":
            self.model = ModelLoader().load_from_file(modelPath)
            if not isinstance(self.model, ImageModelDescriptor):
                raise TypeError(
                    f"SMoSR model {modelPath} did not resolve to an image model descriptor"
                )
            self.model = self.model.model
        elif self.baseMethod == "gauss":
            from safetensors.torch import load_file

            from src.extraArches.DIS import DIS

            self.model = DIS(scale=2, num_features=32, num_blocks=12)
            stateDict = load_file(modelPath)
            self.model.load_state_dict(stateDict)
            del stateDict
        else:
            if self.customModel:
                self.model = ModelLoader().load_from_file(modelPath)
                if not isinstance(self.model, ImageModelDescriptor):
                    raise TypeError(
                        f"Custom upscale model {modelPath} did not resolve to an image model descriptor"
                    )
            else:
                self.model = torch.load(
                    modelPath, map_location="cpu", weights_only=False
                )
                if isinstance(self.model, dict):
                    self.model = ModelLoader().load_from_state_dict(self.model)

            try:
                # SPANDREL HAXX
                self.model = self.model.model
            except Exception:
                pass

        self.model = self.model.eval()

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
        self.requiredMultiple = self._detectRequiredMultiple()
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

    def _detectRequiredMultiple(self) -> int:
        """Probe the model on tiny inputs to learn its required spatial multiple."""
        from src.upscale._shared import smallestValidMultiple

        def runOK(h, w):
            try:
                probe = torch.zeros((1, 3, h, w), device=self.device, dtype=self.dtype)
                # no_grad, not inference_mode: archs that lazily reparameterize on
                # first forward (SPAN's eval_conv) would otherwise cache an
                # inference-mode tensor that the grad-tracked warmup cannot reuse.
                with torch.no_grad():
                    self.model(probe)
                return True
            except Exception:
                return False

        return smallestValidMultiple(runOK)

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
