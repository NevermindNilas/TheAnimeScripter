"""Caching-allocator settings scoped to one run."""

import contextlib
import os

import torch

# Every name torch reads its CUDA allocator config from.
_USER_CONF_VARS = ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF")

# Backends whose model memory is not torch's caching allocator's: TensorRT
# engines, ONNX Runtime, NCNN, and the MPS/ROCm builds.
_OTHER_BACKEND_SUFFIXES = (
    "-tensorrt",
    "-directml",
    "-openvino",
    "-ncnn",
    "-mps",
    "-rocm",
)
# The duplicate detectors that run a PyTorch CUDA model (a CUDA graph).
_TORCH_CUDA_DETECTORS = frozenset({"flownets"})


def _isTorchCudaModel(method: str) -> bool:
    if not method or method.endswith(_OTHER_BACKEND_SUFFIXES):
        return False
    if method.startswith("maxine-"):  # NVIDIA's SDK allocates its own
        return False
    # Plain linethinner runs on the CPU; its -cuda twins are torch.
    return not method.startswith("linethinner-") or method.endswith("-cuda")


def runsTorchCudaModel(chain) -> bool:
    """Whether a stage of ``chain`` (``StageSpec``s) runs a PyTorch CUDA model."""
    for spec in chain:
        settings = spec.settings
        if spec.kind == "restore":
            methods = tuple(settings.get("methods", ()))
        elif spec.kind in ("upscale", "interpolate"):
            methods = (settings.get("method", ""),)
        else:
            methods = ()
        if any(_isTorchCudaModel(method) for method in methods):
            return True
        detectors = (settings.get("method", ""),) if spec.kind == "dedup" else ()
        if spec.smooth:
            detectors = (settings.get("detector", ""),)
        if any(detector in _TORCH_CUDA_DETECTORS for detector in detectors):
            return True
    return False


@contextlib.contextmanager
def expandableSegments(enabled: bool = True):
    """Back new CUDA allocations with expandable segments for one run.

    With fixed segments the caching allocator reserves well past the live peak
    whenever a run allocates many large, differently sized blocks: depth heads
    (large_v3 at 1080p high: 6.2 GB reserved for a 3.7 GB peak, 3.9 GB with
    this), and the CUDA-graph pools a capture fills (shufflecugan at 1080p:
    2.3 GB pool, 1.7 GB with this). Output is unaffected. Restored afterwards so
    nothing outside the run inherits it. CUDA only; ``enabled=False`` makes it
    a no-op.

    Skipped whenever the user set PYTORCH_CUDA_ALLOC_CONF or PYTORCH_ALLOC_CONF
    at all: the setter replaces the whole allocator config, so toggling here
    would silently reset their other keys (max_split_size_mb,
    garbage_collection_threshold, ...) for the run and after it.
    """
    setSettings = getattr(torch._C, "_accelerator_setAllocatorSettings", None)
    if (
        not enabled
        or setSettings is None
        or not torch.cuda.is_available()
        or torch.version.cuda is None
        or any(os.environ.get(name) for name in _USER_CONF_VARS)
    ):
        yield
        return
    setSettings("expandable_segments:True")
    try:
        yield
    finally:
        setSettings("expandable_segments:False")
