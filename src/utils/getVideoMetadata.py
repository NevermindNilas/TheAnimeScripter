"""Backward-compatibility shim. Import from src.io.getVideoMetadata directly."""

import sys as _sys

import src.io.getVideoMetadata as _canonical

_sys.modules[__name__] = _canonical
