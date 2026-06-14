"""
Backward-compatibility shim. Import from src.upscale.* directly.

  src.upscale.pytorch   -- UniversalPytorch, UniversalPytorchMPS
  src.upscale.tensorrt  -- UniversalTensorRT, AnimeSRTensorRT
  src.upscale.directml  -- UniversalDirectML, AnimeSRDirectML
  src.upscale.artcnn    -- _ArtCNNLumaMixin, ArtCNNTensorRT, ArtCNNDirectML
  src.upscale.ncnn      -- UniversalNCNN
  src.upscale.misc      -- NvidiaVSR, AnimeSR
"""

from src.upscale.artcnn import ArtCNNDirectML, ArtCNNTensorRT, _ArtCNNLumaMixin
from src.upscale.directml import AnimeSRDirectML, UniversalDirectML
from src.upscale.misc import AnimeSR, NvidiaVSR
from src.upscale.ncnn import UniversalNCNN
from src.upscale.pytorch import UniversalPytorch, UniversalPytorchMPS
from src.upscale.tensorrt import AnimeSRTensorRT, UniversalTensorRT

__all__ = [
    "UniversalPytorch",
    "UniversalPytorchMPS",
    "UniversalTensorRT",
    "AnimeSRTensorRT",
    "UniversalDirectML",
    "AnimeSRDirectML",
    "_ArtCNNLumaMixin",
    "ArtCNNTensorRT",
    "ArtCNNDirectML",
    "UniversalNCNN",
    "NvidiaVSR",
    "AnimeSR",
]
