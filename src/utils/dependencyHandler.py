"""Backward-compatibility shim. Import from src.infra.dependencyHandler directly."""

import sys as _sys

import src.infra.dependencyHandler as _canonical

_sys.modules[__name__] = _canonical
