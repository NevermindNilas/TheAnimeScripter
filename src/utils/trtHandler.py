"""Backward-compatibility shim. Import from src.model.trtHandler directly."""

import sys as _sys

import src.model.trtHandler as _canonical

_sys.modules[__name__] = _canonical
