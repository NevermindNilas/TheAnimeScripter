from .artcnn import ArtCNNDirectML, ArtCNNTensorRT
from .directml import AnimeSRDirectML, UniversalDirectML
from .misc import AnimeSR, NvidiaVSR
from .ncnn import UniversalNCNN
from .pytorch import UniversalPytorch, UniversalPytorchMPS
from .tensorrt import AnimeSRTensorRT, UniversalTensorRT

__all__ = [
    "UniversalPytorch",
    "UniversalPytorchMPS",
    "UniversalTensorRT",
    "AnimeSRTensorRT",
    "UniversalDirectML",
    "AnimeSRDirectML",
    "ArtCNNTensorRT",
    "ArtCNNDirectML",
    "UniversalNCNN",
    "NvidiaVSR",
    "AnimeSR",
]
