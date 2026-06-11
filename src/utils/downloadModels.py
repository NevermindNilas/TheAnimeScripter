"""Backward-compatibility shim. Import from src.model.downloadModels directly."""
import sys as _sys
import src.model.downloadModels as _canonical
_sys.modules[__name__] = _canonical
