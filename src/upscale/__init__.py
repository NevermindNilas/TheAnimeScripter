"""Upscale backends namespace (lazy).

Importing this package must not import torch or probe CUDA: backend modules
are imported function-level lazy by src/factories/upscale.py, and an eager
re-export here would pay a torch import plus a CudaChecker probe per backend
on every import -- and break torch-less loading (bare CI venv) for the pure
integer-math helpers in src/upscale/_shared.py. Attribute access still works
via PEP 562, just resolved on first use.
"""

_LAZY_BACKENDS = {
    "ArtCNNDirectML": ".artcnn",
    "ArtCNNTensorRT": ".artcnn",
    "AnimeSRDirectML": ".directml",
    "UniversalDirectML": ".directml",
    "AnimeSR": ".misc",
    "NvidiaVSR": ".misc",
    "UniversalNCNN": ".ncnn",
    "UniversalPytorch": ".pytorch",
    "UniversalPytorchMPS": ".pytorch",
    "AnimeSRTensorRT": ".tensorrt",
    "UniversalTensorRT": ".tensorrt",
}

__all__ = sorted(_LAZY_BACKENDS)


def __getattr__(name: str):
    if name in _LAZY_BACKENDS:
        import importlib

        module = importlib.import_module(_LAZY_BACKENDS[name], __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
