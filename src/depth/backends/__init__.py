from importlib import import_module
from typing import TYPE_CHECKING

__all__ = [
    "DepthCuda",
    "OGDepthV2CUDA",
    "OGDepthV3Cuda",
    "DepthMPS",
    "OGDepthV2MPS",
    "OGDepthV3MPS",
    "DepthTensorRTV2",
    "OGDepthV2TensorRT",
    "DepthDirectMLV2",
    "OGDepthV2DirectML",
    "VideoDepthAnythingCUDA",
    "VideoDepthAnythingTorch",
]

_EXPORTS = {
    "DepthCuda": ".cuda",
    "OGDepthV2CUDA": ".cuda",
    "OGDepthV3Cuda": ".cuda",
    "DepthMPS": ".mps",
    "OGDepthV2MPS": ".mps",
    "OGDepthV3MPS": ".mps",
    "DepthTensorRTV2": ".tensorrt",
    "OGDepthV2TensorRT": ".tensorrt",
    "DepthDirectMLV2": ".directml",
    "OGDepthV2DirectML": ".directml",
    "VideoDepthAnythingCUDA": ".video",
    "VideoDepthAnythingTorch": ".video",
}

if TYPE_CHECKING:
    from .cuda import DepthCuda, OGDepthV2CUDA, OGDepthV3Cuda
    from .directml import DepthDirectMLV2, OGDepthV2DirectML
    from .mps import DepthMPS, OGDepthV2MPS, OGDepthV3MPS
    from .tensorrt import DepthTensorRTV2, OGDepthV2TensorRT
    from .video import VideoDepthAnythingCUDA, VideoDepthAnythingTorch


def __getattr__(name):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
