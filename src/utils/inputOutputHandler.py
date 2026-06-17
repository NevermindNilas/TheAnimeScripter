"""Backward-compatibility shim. Import from src.io.inputOutputHandler directly."""

import sys as _sys

import src.io.inputOutputHandler as _canonical

_sys.modules[__name__] = _canonical
