"""
Backward-compatibility shim. Import from src.upscale.* directly.

  src.upscale.pytorch   -- UniversalPytorch, UniversalPytorchMPS
  src.upscale.tensorrt  -- UniversalTensorRT, AnimeSRTensorRT
  src.upscale.directml  -- UniversalDirectML, AnimeSRDirectML
  src.upscale.artcnn    -- _ArtCNNLumaMixin, ArtCNNTensorRT, ArtCNNDirectML
  src.upscale.ncnn      -- UniversalNCNN
  src.upscale.misc      -- NvidiaVSR, AnimeSR
"""

from src.upscale.pytorch import UniversalPytorch, UniversalPytorchMPS
from src.upscale.tensorrt import UniversalTensorRT, AnimeSRTensorRT
from src.upscale.directml import UniversalDirectML, AnimeSRDirectML
from src.upscale.artcnn import _ArtCNNLumaMixin, ArtCNNTensorRT, ArtCNNDirectML
from src.upscale.ncnn import UniversalNCNN
from src.upscale.misc import NvidiaVSR, AnimeSR

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
