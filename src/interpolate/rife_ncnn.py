import torch

from src.infra.isCudaInit import CudaChecker
from src.model.download import resolveWeightPath
from src.model.registry import modelsMap

checker = CudaChecker()

torch.set_float32_matmul_precision("medium")


class RifeNCNN:
    def __init__(
        self,
        interpolateMethod,
        ensemble=False,
        width=1920,
        height=1080,
        half=True,
        interpolateFactor=2,
    ):
        self.interpolateMethod = interpolateMethod
        self.height = height
        self.width = width
        self.ensemble = ensemble
        self.half = half
        self.interpolateFactor = interpolateFactor

        UHD = True if width >= 3840 or height >= 2160 else False
        # scale = 2 if UHD else 1
        from rife_ncnn_vulkan_python import wrapped

        self.wrapped = wrapped

        self.filename = modelsMap(
            self.interpolateMethod,
            ensemble=self.ensemble,
            modelType="ncnn",
        )

        if self.filename.endswith("-ncnn.zip"):
            self.filename = self.filename[:-9]
        elif self.filename.endswith("-ncnn"):
            self.filename = self.filename[:-5]

        modelPath = resolveWeightPath(
            self.interpolateMethod,
            self.filename,
            ensemble=self.ensemble,
            modelType="ncnn",
        )

        if modelPath.endswith("-ncnn.zip"):
            modelPath = modelPath[:-9]
        elif modelPath.endswith("-ncnn"):
            modelPath = modelPath[:-5]

        padding = 32  # ADD 64 once 4.25-ncnn is added

        self.Rife = self.wrapped.RifeWrapped(
            0,
            self.ensemble,
            False,
            UHD,
            2,
            False,
            True,
            padding,
        )

        modelDir = self.wrapped.StringType()
        modelDir.wstr = self.wrapped.new_wstr_p()
        self.wrapped.wstr_p_assign(modelDir.wstr, str(modelPath))
        self.Rife.load(modelDir)

        bufSize = width * height * 3
        self.outputBytes = bytearray(bufSize)
        self.output = self.wrapped.Image(self.outputBytes, self.width, self.height, 3)

        self._frameBytes = [bytearray(bufSize), bytearray(bufSize)]
        self._frameBufs = [
            torch.frombuffer(self._frameBytes[0], dtype=torch.uint8).reshape(
                height, width, 3
            ),
            torch.frombuffer(self._frameBytes[1], dtype=torch.uint8).reshape(
                height, width, 3
            ),
        ]
        self._frameImages = [
            self.wrapped.Image(self._frameBytes[0], self.width, self.height, 3),
            self.wrapped.Image(self._frameBytes[1], self.width, self.height, 3),
        ]
        self._frame0Idx = 0
        self._firstFrame = True
        self.shape = (self.height, self.width)

    def _writeFrameInto(self, frame, idx):
        src = (
            frame.mul(255)
            .clamp(0, 255)
            .squeeze(0)
            .permute(1, 2, 0)
            .to(torch.uint8)
            .cpu()
            .contiguous()
        )
        self._frameBufs[idx].copy_(src)

    def cacheFrame(self):
        self._frame0Idx = 1 - self._frame0Idx

    def cacheFrameReset(self, frame):
        # Scene-cut reset: make `frame` the anchor (frame0 slot). Reuse the
        # normal 0..1-float -> HWC-uint8 conversion (_writeFrameInto); NCNN is
        # stateless per pair, so there is no feature cache to clear.
        self._writeFrameInto(frame, self._frame0Idx)
        self._firstFrame = False

    def __call__(self, frame, interpQueue, framesToInsert=1, timesteps=None):
        if self._firstFrame:
            self._writeFrameInto(frame, self._frame0Idx)
            self._firstFrame = False
            return False

        frame1Idx = 1 - self._frame0Idx
        self._writeFrameInto(frame, frame1Idx)

        image0 = self._frameImages[self._frame0Idx]
        image1 = self._frameImages[frame1Idx]

        for i in range(framesToInsert):
            if timesteps is not None and i < len(timesteps):
                t = timesteps[i]
            else:
                t = (i + 1) * 1 / (framesToInsert + 1)

            self.Rife.process(image0, image1, t, self.output)

            output = (
                torch.frombuffer(self.outputBytes, dtype=torch.uint8)
                .reshape(self.height, self.width, 3)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(dtype=torch.float16 if self.half else torch.float32)
                .mul(1 / 255.0)
            )
            interpQueue.put(output)

        self.cacheFrame()
