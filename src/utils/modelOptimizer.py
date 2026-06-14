"""Backward-compatibility shim. Import from src.model.modelOptimizer directly."""

import sys as _sys

import src.model.modelOptimizer as _canonical

_sys.modules[__name__] = _canonical
