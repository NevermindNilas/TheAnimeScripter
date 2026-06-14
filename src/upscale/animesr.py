import torch
import logging

from src.model.download import resolveWeightPath
from src.model.registry import modelsMap
from src.infra.isCudaInit import CudaChecker
from src.infra.logAndPrint import logAndPrint

checker = CudaChecker()

torch.set_float32_matmul_precision("medium")


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
        ).to(memory_format=torch.channels_last)
        self.nextFrame = torch.zeros(
            (
                1,
                3,
                self.padding[3] + self.height + self.padding[2],
                self.padding[1] + self.width + self.padding[0],
            ),
            device=checker.device,
            dtype=torch.float16 if self.half else torch.float32,
        ).to(memory_format=torch.channels_last)

        self.dummyOutput = self.prevFrame.new_zeros(
            1, 3, self.height * 4, self.width * 4
        ).to(memory_format=torch.channels_last)

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
                    frame.to(dtype=frame.dtype).to(memory_format=torch.channels_last),
                    non_blocking=False,
                )
                if nextFrame is None:
                    self.nextFrame.copy_(
                        frame.to(dtype=frame.dtype).to(
                            memory_format=torch.channels_last
                        ),
                        non_blocking=False,
                    )
                else:
                    self.nextFrame.copy_(
                        nextFrame.to(dtype=frame.dtype).to(
                            memory_format=torch.channels_last
                        ),
                        non_blocking=False,
                    )
            self.normStream.synchronize()

            self.firstRun = False
        else:
            with torch.cuda.stream(self.normStream):
                if nextFrame is None:
                    self.nextFrame.copy_(
                        frame.to(dtype=frame.dtype).to(
                            memory_format=torch.channels_last
                        ),
                        non_blocking=False,
                    )
                else:
                    self.nextFrame.copy_(
                        nextFrame.to(dtype=frame.dtype).to(
                            memory_format=torch.channels_last
                        ),
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
