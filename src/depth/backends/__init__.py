from importlib import import_module
from typing import TYPE_CHECKING

__all__ = [
    "DepthCuda",
    "DepthV3Cuda",
    "DA3StreamingCuda",
    "LimboStreamingCuda",
    "DepthMPS",
    "DepthV3MPS",
    "DepthTensorRTV2",
    "VideoDepthTensorRT",
    "DepthDirectMLV2",
    "VideoDepthAnythingTorch",
]

_EXPORTS = {
    "DepthCuda": ".cuda",
    "DepthV3Cuda": ".cuda",
    "DA3StreamingCuda": ".da3_streaming",
    "LimboStreamingCuda": ".da3_streaming",
    "DepthMPS": ".mps",
    "DepthV3MPS": ".mps",
    "DepthTensorRTV2": ".tensorrt",
    "VideoDepthTensorRT": ".tensorrt",
    "DepthDirectMLV2": ".directml",
    "VideoDepthAnythingTorch": ".video",
}

if TYPE_CHECKING:
    from .cuda import DepthCuda, DepthV3Cuda
    from .da3_streaming import DA3StreamingCuda, LimboStreamingCuda
    from .directml import DepthDirectMLV2
    from .mps import DepthMPS, DepthV3MPS
    from .tensorrt import DepthTensorRTV2, VideoDepthTensorRT
    from .video import VideoDepthAnythingTorch


def __getattr__(name):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
