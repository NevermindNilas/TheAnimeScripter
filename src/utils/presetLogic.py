"""Backward-compatibility shim. Import from src.server.presetLogic directly."""

import sys as _sys

import src.server.presetLogic as _canonical

_sys.modules[__name__] = _canonical
