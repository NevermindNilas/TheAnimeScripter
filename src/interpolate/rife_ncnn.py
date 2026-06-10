import os
import torch
import logging
import torch.nn.functional as F
import math
import numpy as np

from src.model.downloadModels import downloadModels, weightsDir, modelsMap, resolveWeightPath
from src.infra.isCudaInit import CudaChecker
from src.infra.logAndPrint import logAndPrint

from src.constants import ADOBE

if ADOBE:
    from src.server.aeComms import progressState


checker = CudaChecker()

torch.set_float32_matmul_precision("medium")


_RIFE_V1 = {
    "rife":           ("IFNet425",      "IFNet_rife425"),
    "rife4.25":       ("IFNet425",      "IFNet_rife425"),
    "rife4.25-heavy": ("IFNet425Heavy", "IFNet_rife425heavy"),
    "rife4.25-lite":  ("IFNet425Lite",  "IFNet_rife425lite"),
    "rife4.22":       ("IFNet422",      "IFNet_rife422"),
    "rife4.22-lite":  ("IFNet422Lite",  "IFNet_rife422lite"),
    "rife4.21":       ("IFNet421",      "IFNet_rife421"),
    "rife4.20":       ("IFNet420",      "IFNet_rife420"),
    "rife4.18":       ("IFNet418",      "IFNet_rife418"),
    "rife4.17":       ("IFNet417",      "IFNet_rife417"),
    "rife4.15-lite":  ("IFNet415Lite",  "IFNet_rife415lite"),
    "rife4.16-lite":  ("IFNet416Lite",  "IFNet_rife416lite"),
    "rife4.6":        ("IFNet46",       "IFNet_rife46"),
    "rife_elexor":    (None,            "IFNet_elexor_cuda"),
}


def _loadV1(method, half):
    fastName, baseMod = _RIFE_V1[method]
    if half and fastName:
        from src.rifearches import rife_fast
        return getattr(rife_fast, fastName)
    mod = __import__(f"src.rifearches.{baseMod}", fromlist=["IFNet"])
    return mod.IFNet


def importRifeArch(interpolateMethod, version, half=True):
    match version:
        case "v1":
            return _loadV1(interpolateMethod, half)

        case "v3":
            match interpolateMethod:
                case "rife4.25-heavy-tensorrt":
                    from src.rifearches.Rife425_heavy_v3 import IFNet

                    Head = True

                case "rife4.25-lite-tensorrt":
                    from src.rifearches.Rife425_lite_v3 import IFNet

                    Head = True
                case "rife4.25-tensorrt":
                    from src.rifearches.Rife425_v3 import IFNet

                    Head = True
                case "rife4.22-tensorrt":
                    from src.rifearches.Rife422_v3 import IFNet

                    Head = True
                case "rife4.22-lite-tensorrt":
                    from src.rifearches.Rife422_lite_v3 import IFNet

                    Head = True
                case "rife4.21-tensorrt":
                    from src.rifearches.Rife422_v3 import IFNet

                    Head = True
                case "rife4.20-tensorrt":
                    from src.rifearches.Rife420_v3 import IFNet

                    Head = True
                case "rife4.18-tensorrt":
                    from src.rifearches.Rife415_v3 import IFNet

                    Head = True
                case "rife4.17-tensorrt":
                    from src.rifearches.Rife415_v3 import IFNet

                    Head = True
                case "rife4.15-tensorrt":
                    from src.rifearches.Rife415_v3 import IFNet

                    Head = True
                case "rife4.6-tensorrt":
                    from src.rifearches.Rife46_v3 import IFNet

                    Head = False
                case "rife4.6-directml" | "rife4.6-openvino":
                    from src.rifearches.Rife_directml import IFNet_46 as IFNet

                    Head = False
                case "rife4.22-directml" | "rife4.22-openvino":
                    from src.rifearches.Rife_directml import IFNet_422 as IFNet

                    Head = True
                case (
                    "rife4.15-directml"
                    | "rife4.17-directml"
                    | "rife4.18-directml"
                    | "rife4.15-openvino"
                    | "rife4.17-openvino"
                    | "rife4.18-openvino"
                ):
                    from src.rifearches.Rife_directml import IFNet_415 as IFNet

                    Head = True
                case (
                    "rife4.20-directml"
                    | "rife4.21-directml"
                    | "rife4.20-openvino"
                    | "rife4.21-openvino"
                ):
                    from src.rifearches.Rife_directml import IFNet_420 as IFNet

                    Head = True
                case "rife4.22-lite-directml" | "rife4.22-lite-openvino":
                    from src.rifearches.Rife_directml import IFNet_422_lite as IFNet

                    Head = True
                case "rife4.25-directml" | "rife4.25-openvino":
                    from src.rifearches.Rife_directml import IFNet_425 as IFNet

                    Head = True
                case "rife4.25-lite-directml" | "rife4.25-lite-openvino":
                    from src.rifearches.Rife_directml import IFNet_425_lite as IFNet

                    Head = True
                case "rife4.25-heavy-directml" | "rife4.25-heavy-openvino":
                    from src.rifearches.Rife_directml import IFNet_425_heavy as IFNet

                    Head = True
                case "rife_elexor-tensorrt":
                    from src.rifearches.IFNet_elexor_tensorrt import IFNet

                    Head = True
            return IFNet, Head


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
        arr = (
            frame.cpu().numpy().astype("uint8")
            if torch.is_tensor(frame)
            else np.asarray(frame).astype("uint8")
        )
        arr = np.ascontiguousarray(arr).reshape(self.height, self.width, 3)
        self._frameBytes[self._frame0Idx][:] = arr.tobytes()
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


