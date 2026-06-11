import os
import torch
import logging
import torch.nn.functional as F
import math
import numpy as np

from src.model.download import downloadModels
from src.model.registry import weightsDir, modelsMap
from src.model.download import resolveWeightPath
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


