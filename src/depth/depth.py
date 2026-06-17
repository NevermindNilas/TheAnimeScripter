"""
Backward-compatibility shim. Import from src.depth.backends.* directly.

  src.depth.backends.cuda     -- DepthCuda, OGDepthV2CUDA, OGDepthV3Cuda
  src.depth.backends.tensorrt -- DepthTensorRTV2, OGDepthV2TensorRT
  src.depth.backends.directml -- DepthDirectMLV2, OGDepthV2DirectML
  src.depth.backends.video    -- VideoDepthAnythingCUDA, VideoDepthAnythingTorch
"""

from src.depth.backends.cuda import DepthCuda, OGDepthV2CUDA, OGDepthV3Cuda
from src.depth.backends.directml import DepthDirectMLV2, OGDepthV2DirectML
from src.depth.backends.tensorrt import DepthTensorRTV2, OGDepthV2TensorRT
from src.depth.backends.video import VideoDepthAnythingCUDA, VideoDepthAnythingTorch

__all__ = [
    "DepthCuda",
    "OGDepthV2CUDA",
    "OGDepthV3Cuda",
    "DepthTensorRTV2",
    "OGDepthV2TensorRT",
    "DepthDirectMLV2",
    "OGDepthV2DirectML",
    "VideoDepthAnythingCUDA",
    "VideoDepthAnythingTorch",
]
