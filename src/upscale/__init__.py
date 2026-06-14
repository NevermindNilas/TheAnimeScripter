from .pytorch import UniversalPytorch, UniversalPytorchMPS
from .tensorrt import UniversalTensorRT
from .directml import UniversalDirectML
from .artcnn import _ArtCNNLumaMixin, ArtCNNTensorRT, ArtCNNDirectML
from .ncnn import UniversalNCNN
from .nvidiavsr import NvidiaVSR
from .animesr import AnimeSR, AnimeSRTensorRT, AnimeSRDirectML

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
