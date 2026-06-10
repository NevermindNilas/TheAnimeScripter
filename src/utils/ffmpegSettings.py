"""Backward-compatibility shim. Import from src.io.ffmpegSettings directly."""
import sys as _sys
import src.io.ffmpegSettings as _canonical
_sys.modules[__name__] = _canonical
