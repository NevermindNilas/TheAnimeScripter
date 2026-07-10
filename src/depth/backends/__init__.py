from .cuda import DepthCuda, OGDepthV2CUDA, OGDepthV3Cuda
from .directml import DepthDirectMLV2, OGDepthV2DirectML
from .mps import DepthMPS, OGDepthV2MPS, OGDepthV3MPS
from .tensorrt import DepthTensorRTV2, OGDepthV2TensorRT
from .video import VideoDepthAnythingCUDA, VideoDepthAnythingTorch

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
