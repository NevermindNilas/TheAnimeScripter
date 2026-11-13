"""Create the CUDA primary context while torch is still importing.

A CUDA run pays for the driver's init and the device's primary context the first
time torch touches the GPU, ~150 ms on an RTX 3090 under Linux, after torch's own
~550 ms import. Neither needs torch: a daemon thread asks the driver for them
through ctypes (the call releases the GIL) while the main thread imports torch,
and torch's runtime then finds the context already made. It is the same context
torch would create (the device's primary one, on device 0, which every TAS
backend uses), so nothing about the run changes but when it is made.

It starts only for a ``main.py`` launch (``enableCudaContextWarmup``) whose
parsed arguments describe a run that creates that context anyway: an input that
can exist and a processing stage other than a CPU autoclip (every such run opens
CUDA streams in the reader and writer). A CPU autoclip, a run with no input or
a missing file, and a launch with nothing to process never hold a context
(~260 MiB VRAM, ~150 MiB RSS), as before.

Linux only for now: on Windows ``cuInit`` would run beside torch's DLL loading,
which shares the loader lock, and that has not been verified there.

Torch-free and best-effort: no NVIDIA driver, a CPU/ROCm torch build or any
failure simply means torch creates the context itself, as before.
"""

import ctypes
import os
import re
import sys
import threading
from importlib.util import find_spec

_TORCH_CUDA_BUILD = re.compile(r"^cuda\s*(?::[^=]*)?=\s*['\"]", re.MULTILINE)
_TORCH_HIP_BUILD = re.compile(r"^hip\s*(?::[^=]*)?=\s*['\"]", re.MULTILINE)

# Stage switches whose runs build a CUDA reader/writer (BuildBuffer's and
# NeluxWriteBuffer's CUDA streams) when CUDA is available. smooth_dedup implies
# interpolate later in validation.
_PROCESSING_FLAGS = (
    "interpolate",
    "upscale",
    "restore",
    "resize",
    "dedup",
    "smooth_dedup",
    "segment",
    "depth",
    "obj_detect",
    "stabilize",
    "moblur",
)

# Autoclip takes precedence over every other stage and writes no video; only
# these methods run on CUDA.
_CUDA_AUTOCLIP_METHODS = ("maxxvit-tensorrt", "transnetv2")

_enabled = False


def enableCudaContextWarmup() -> None:
    """Allow the warmup in this process (main.py's entry point only)."""
    global _enabled
    _enabled = True


def _driverLibraryName() -> str | None:
    if sys.platform.startswith("linux"):
        return "libcuda.so.1"
    return None  # Windows unverified (see module docstring); macOS has no CUDA


def _torchIsCudaBuild() -> bool:
    """Whether the installed torch is a CUDA build, read without importing it."""
    try:
        spec = find_spec("torch")
        if spec is None or not spec.submodule_search_locations:
            return False
        for folder in spec.submodule_search_locations:
            path = os.path.join(folder, "version.py")
            if os.path.isfile(path):
                with open(path, encoding="utf-8") as f:
                    text = f.read()
                return bool(_TORCH_CUDA_BUILD.search(text)) and not bool(
                    _TORCH_HIP_BUILD.search(text)
                )
    except Exception:
        return False
    return False


def _inputCanExist(inputSpec: str) -> bool:
    """False only for a single local path that does not exist (validation then
    fails before any CUDA work). Lists, sequence patterns and URLs count as
    present."""
    text = inputSpec.strip()
    if not text:
        return False
    if ";" in text or "%" in text or os.path.exists(text):
        return True
    try:
        from src.io.inputOutputHandler import isUrlInput

        return isUrlInput(text)
    except Exception:
        return True


def runWillUseCuda(args) -> bool:
    """Whether these parsed arguments describe a run that creates the CUDA
    context on an NVIDIA machine anyway."""
    inputSpec = getattr(args, "input", None)
    if not inputSpec or not _inputCanExist(str(inputSpec)):
        return False
    if getattr(args, "autoclip", False):
        return getattr(args, "autoclip_method", None) in _CUDA_AUTOCLIP_METHODS
    return any(bool(getattr(args, flag, False)) for flag in _PROCESSING_FLAGS)


def _retainPrimaryContext(libraryName: str) -> None:
    try:
        driver = ctypes.CDLL(libraryName)
        if driver.cuInit(0) != 0:
            return
        device = ctypes.c_int()
        if driver.cuDeviceGet(ctypes.byref(device), 0) != 0:
            return
        context = ctypes.c_void_p()
        # Retained for the life of the process, as torch's runtime keeps it.
        driver.cuDevicePrimaryCtxRetain(ctypes.byref(context), device)
    except Exception:
        pass


def startCudaContextWarmup(args) -> threading.Thread | None:
    """Start making the CUDA primary context in the background, when useful.

    ``args`` is the parsed namespace (JSON merged, chain built). Returns the
    started daemon thread, or None when not enabled, when the run would not
    touch CUDA, or without a CUDA driver or a CUDA torch build.
    """
    try:
        if not _enabled or not runWillUseCuda(args):
            return None
        libraryName = _driverLibraryName()
        if libraryName is None or not _torchIsCudaBuild():
            return None
        thread = threading.Thread(
            target=_retainPrimaryContext,
            args=(libraryName,),
            name="cuda-context-warmup",
            daemon=True,
        )
        thread.start()
        return thread
    except Exception:
        return None
