import logging

import torch

from src.constants import ADOBE
from src.utils.downloadModels import (
    modelsMap,
    resolveWeightPath,
)
from src.utils.isCudaInit import CudaChecker
from src.utils.logAndPrint import logAndPrint

if ADOBE:
    from src.utils.aeComms import progressState

checker = CudaChecker()

torch.set_float32_matmul_precision("medium")


def calculatePadding(width, height, multiple=4):
    padW = (multiple - (width % multiple)) % multiple
    padH = (multiple - (height % multiple)) % multiple
    return (0, padW, 0, padH)


class NvidiaVSR:
    """
    NVIDIA Maxine Video Super Resolution (RTX VSR).

    Quality preset is parsed from `upscaleMethod` suffix, e.g.
    `maxine-ultra` -> ULTRA. Default `maxine` -> HIGH.
    Supported: BICUBIC, LOW, MEDIUM, HIGH, ULTRA, HIGHBITRATE_{LOW..ULTRA}.

    API hard limits (Maxine 1.2.0):
      - Input dtype must be float32 (no fp16/bf16)
      - Input shape must be (3, H, W) (no batch, no channels_last)
      - Scale factor must be 1..4

    """

    _VALID_QUALITIES = {
        "BICUBIC",
        "LOW",
        "MEDIUM",
        "HIGH",
        "ULTRA",
        "HIGHBITRATE_LOW",
        "HIGHBITRATE_MEDIUM",
        "HIGHBITRATE_HIGH",
        "HIGHBITRATE_ULTRA",
    }

    def __init__(
        self,
        upscaleMethod: str = "maxine-high",
        upscaleFactor: int = 2,
        half: bool = False,
        width: int = 1920,
        height: int = 1080,
        customModel: str = None,
        compileMode: str = "default",
    ):
        self.upscaleMethod = upscaleMethod
        self.upscaleFactor = upscaleFactor
        self.half = half
        self.width = width
        self.height = height
        self.customModel = customModel
        self.compileMode = compileMode

        if not checker.cudaAvailable:
            raise RuntimeError("NvidiaVSR requires a CUDA-capable NVIDIA GPU")

        if self.half:
            logAndPrint(
                "NVIDIA VSR API requires float32 input; forcing half=False.",
                "yellow",
            )
            self.half = False

        if self.customModel:
            logAndPrint(
                "NVIDIA VSR does not support custom models (bundled in wheel); "
                "ignoring customModel.",
                "yellow",
            )
            self.customModel = None

        if self.compileMode != "default":
            logAndPrint(
                f"compileMode '{self.compileMode}' ignored on NVIDIA VSR backend.",
                "yellow",
            )
            self.compileMode = "default"

        self.qualityName = self._parseQuality(self.upscaleMethod)

        self.handleModel()

    @classmethod
    def _parseQuality(cls, method: str) -> str:
        suffix = method.lower().replace("maxine", "", 1).lstrip("-")
        name = suffix.upper() if suffix else "HIGH"
        if name not in cls._VALID_QUALITIES:
            raise ValueError(
                f"Unknown Maxine VSR quality '{name}' in upscaleMethod "
                f"'{method}'. Valid: {sorted(cls._VALID_QUALITIES)}"
            )
        return name

    def handleModel(self):
        if ADOBE:
            progressState.update(
                {"status": f"Loading NVIDIA VSR ({self.qualityName})..."}
            )

        import nvvfx
        from nvvfx import VideoSuperRes

        self.nvvfx = nvvfx

        quality = VideoSuperRes.QualityLevel[self.qualityName]
        deviceIdx = checker.device.index if checker.device.index is not None else 0

        self.model = VideoSuperRes(quality=quality, device=deviceIdx)
        self.model.output_width = self.width * self.upscaleFactor
        self.model.output_height = self.height * self.upscaleFactor

        try:
            self.model.load()
        except Exception as e:
            logging.error(f"NVIDIA VSR load failed: {e}")
            raise

        self.dummyInput = torch.zeros(
            (3, self.height, self.width),
            device=checker.device,
            dtype=torch.float32,
        ).contiguous()

        self.dummyOutput = torch.zeros(
            (
                1,
                3,
                self.height * self.upscaleFactor,
                self.width * self.upscaleFactor,
            ),
            device=checker.device,
            dtype=torch.float32,
        ).contiguous()

        for _ in range(5):
            _out = self.model.run(self.dummyInput, stream_ptr=0)
            _ = torch.from_dlpack(_out.image).clone()
        self.dummyInput.uniform_(0.0, 1.0)
        for _ in range(5):
            _out = self.model.run(self.dummyInput, stream_ptr=0)
            _ = torch.from_dlpack(_out.image).clone()
        self.dummyInput.zero_()
        torch.cuda.synchronize()

    @torch.inference_mode()
    def __call__(
        self, frame: torch.Tensor, nextFrame: torch.Tensor = None
    ) -> torch.Tensor:
        src = frame.squeeze(
            0
        )  # We are always using 4 dim throughout the process, but Maxine API expects 3 dim (C,H,W), so remove batch dim here.
        self.dummyInput.copy_(src.contiguous(), non_blocking=False)
        outCapsule = self.model.run(self.dummyInput, stream_ptr=0)
        upscaled = torch.from_dlpack(outCapsule.image)  # (3, H', W')
        self.dummyOutput[0].copy_(upscaled, non_blocking=False)
        output = self.dummyOutput.clone()
        torch.cuda.synchronize()
        return output


class AnimeSR:
    def __init__(
        self,
        upscaleFactor: int = 2,
        half: bool = False,
        width: int = 1920,
        height: int = 1080,
        compileMode: str = "default",
    ):
        self.upscaleFactor = upscaleFactor
        self.half = half
        self.width = width
        self.height = height
        self.compileMode: str = compileMode

        self.handleModel()

    def handleModel(self):
        """
        Load the desired model
        """
        self.filename = modelsMap("animesr", self.upscaleFactor, modelType="pth")
        modelPath = resolveWeightPath(
            "animesr",
            self.filename,
            upscaleFactor=self.upscaleFactor,
        )

        from src.extraArches.AnimeSR import MSRSWVSR

        self.model = MSRSWVSR(num_feat=64, num_block=[5, 3, 2], netscale=4)

        stateDict = torch.load(modelPath, map_location="cpu")
        self.model.load_state_dict(stateDict)
        del stateDict

        self.model = self.model.eval()

        if self.half and checker.cudaAvailable:
            try:
                self.model = self.model.half()
            except Exception as e:
                logging.error(f"Error converting model to half precision: {e}")
                self.model = self.model.float()
                self.half = False

        if checker.cudaAvailable:
            self.model = self.model.cuda()
            torch.cuda.empty_cache()

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
                    f"Error compiling model {self.upscaleMethod} with mode {self.compileMode}: {e}"
                )
                logAndPrint(
                    f"Error compiling model {self.upscaleMethod} with mode {self.compileMode}: {e}",
                    "red",
                )

            self.compileMode = "default"

        # padding related logic
        ph = (4 - self.height % 4) % 4
        pw = (4 - self.width % 4) % 4
        self.padding = (0, pw, 0, ph)

        # The arch requires 3 inputs, so we create dummy inputs for the other two
        self.prevFrame = torch.zeros(
            (
                1,
                3,
                self.padding[3] + self.height + self.padding[2],
                self.padding[1] + self.width + self.padding[0],
            ),
            device=checker.device,
            dtype=torch.float16 if self.half else torch.float32,
        )
        self.nextFrame = torch.zeros(
            (
                1,
                3,
                self.padding[3] + self.height + self.padding[2],
                self.padding[1] + self.width + self.padding[0],
            ),
            device=checker.device,
            dtype=torch.float16 if self.half else torch.float32,
        )

        self.dummyOutput = self.prevFrame.new_zeros(
            1, 3, self.height * 4, self.width * 4
        )

        # The model has some caching functionality that requires a state
        self.state = self.prevFrame.new_zeros(1, 64, self.height, self.width)

        self.stream = torch.cuda.Stream()
        self.normStream = torch.cuda.Stream()
        self.outputStream = torch.cuda.Stream()

        self.firstRun = True

    def padFrame(self, frame: torch.tensor) -> torch.tensor:
        return torch.nn.functional.pad(frame, self.padding, mode="reflect")

    @torch.inference_mode()
    def __call__(self, frame: torch.tensor, nextFrame: torch.tensor) -> torch.tensor:
        if self.firstRun:
            with torch.cuda.stream(self.normStream):
                self.prevFrame.copy_(
                    frame.to(dtype=frame.dtype),
                    non_blocking=False,
                )
                if nextFrame is None:
                    self.nextFrame.copy_(
                        frame.to(dtype=frame.dtype),
                        non_blocking=False,
                    )
                else:
                    self.nextFrame.copy_(
                        nextFrame.to(dtype=frame.dtype),
                        non_blocking=False,
                    )
            self.normStream.synchronize()

            self.firstRun = False
        else:
            with torch.cuda.stream(self.normStream):
                if nextFrame is None:
                    self.nextFrame.copy_(
                        frame.to(dtype=frame.dtype),
                        non_blocking=False,
                    )
                else:
                    self.nextFrame.copy_(
                        nextFrame.to(dtype=frame.dtype),
                        non_blocking=False,
                    )
            self.normStream.synchronize()

        # preparing that mofo
        with torch.cuda.stream(self.normStream):
            frame = self.padFrame(frame)
        self.normStream.synchronize()

        with torch.cuda.stream(self.outputStream):
            self.dummyOutput, state = self.model(
                self.prevFrame,
                frame,
                self.nextFrame,
                self.dummyOutput,
                self.state,
            )

            self.state = state
        self.outputStream.synchronize()

        with torch.cuda.stream(self.normStream):
            self.prevFrame.copy_(frame, non_blocking=False)
        self.normStream.synchronize()

        # resize the output to self.height*2 and self.width * 2
        with torch.cuda.stream(self.outputStream):
            output = torch.nn.functional.interpolate(
                self.dummyOutput,
                size=(self.height * 2, self.width * 2),
                mode="bicubic",
                align_corners=False,
            )
        self.outputStream.synchronize()

        return output
