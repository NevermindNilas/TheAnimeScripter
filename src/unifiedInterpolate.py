"""
Backward-compatibility shim. Import from src.interpolate.* directly.

  src.interpolate.rife          -- RifeCuda, RifeMPS
  src.interpolate.rife_tensorrt -- RifeTensorRT
  src.interpolate.rife_ncnn     -- RifeNCNN
  src.interpolate.rife_directml -- RifeDirectML
  src.interpolate.distildrba    -- DistilDRBACuda, DistilDRBATensorRT
"""

from src.interpolate.rife import RifeCuda, RifeMPS
from src.interpolate.rife_tensorrt import RifeTensorRT
from src.interpolate.rife_ncnn import RifeNCNN
from src.interpolate.rife_directml import RifeDirectML
from src.interpolate.distildrba import DistilDRBACuda, DistilDRBATensorRT

__all__ = [
    "RifeCuda",
    "RifeMPS",
    "RifeTensorRT",
    "RifeNCNN",
    "RifeDirectML",
    "DistilDRBACuda",
    "DistilDRBATensorRT",
]
