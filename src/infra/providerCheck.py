"""
ONNX Runtime execution-provider verification.

onnxruntime will happily construct an InferenceSession with a requested GPU
execution provider and then silently place every node on CPU when that provider
cannot actually serve the model (missing driver, unsupported op, ABI mismatch).
The only user-visible symptom is a ~30x slowdown with nothing on the console,
because the root logger writes to file only.

``get_providers()`` reports the providers the session actually got, so we compare
that against what was requested and emit a visible warning when the GPU provider
is absent. Logging/observability only -- this never raises and never changes
inference behaviour, device placement, or dtype.
"""

import sys

from src.infra.logAndPrint import logWarning

_GPU_ALTERNATIVES = {
    "linux": (
        "a -ncnn method (Vulkan, any GPU) or, on AMD, a -rocm method with the "
        "linux-rocm profile"
    ),
    "darwin": "a -mps or -ncnn method",
}


def warnCpuFallback(windowsMessage: str) -> None:
    """Warn that an ONNX stage has no GPU provider and is running on the CPU.

    ``windowsMessage`` is what each backend has always printed. DirectML only
    exists on Windows, and the onnxruntime wheels TAS installs elsewhere have
    no OpenVINO provider, so drivers and DirectX advice would be wrong there:
    other platforms are pointed at the backends that do reach their GPU.
    """
    if sys.platform == "win32":
        logWarning(windowsMessage)
        return
    alternatives = _GPU_ALTERNATIVES.get(sys.platform, "a -ncnn method")
    logWarning(
        "DirectML and OpenVINO are not available in this onnxruntime build, so "
        "this stage is running on the CPU and will be much slower. For GPU "
        f"acceleration use {alternatives}, where one exists (see --list_methods)."
    )


def warnIfProviderMissing(session, requested: str, context: str) -> bool:
    """Warn (visibly) if `requested` provider is not active on `session`.

    Returns True when the provider is present, False when it is missing and a
    warning was emitted.
    """
    try:
        active = session.get_providers()
    except Exception:
        # Never let an observability check break real work.
        return True

    if requested in active:
        return True

    logWarning(
        f"{context}: requested {requested} but ONNX Runtime fell back to {active}. "
        f"Expect significantly worse performance."
    )
    return False
