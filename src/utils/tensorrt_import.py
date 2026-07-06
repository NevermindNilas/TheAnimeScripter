from __future__ import annotations

import importlib
from types import ModuleType


def _load_tensorrt() -> ModuleType:
    tensorrt = importlib.import_module("tensorrt")
    if hasattr(tensorrt, "Logger"):
        return tensorrt

    return importlib.import_module("tensorrt_bindings")


trt = _load_tensorrt()
