"""Backward-compatibility shim. Import from src.io.encodingSettings directly."""

import sys as _sys

import src.io.encodingSettings as _canonical

_sys.modules[__name__] = _canonical
