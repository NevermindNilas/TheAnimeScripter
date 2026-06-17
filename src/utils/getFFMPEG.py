"""Backward-compatibility shim. Import from src.infra.getFFMPEG directly."""

import sys as _sys

import src.infra.getFFMPEG as _canonical

_sys.modules[__name__] = _canonical
