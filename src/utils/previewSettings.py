"""Backward-compatibility shim. Import from src.server.previewSettings directly."""

import sys as _sys

import src.server.previewSettings as _canonical

_sys.modules[__name__] = _canonical
