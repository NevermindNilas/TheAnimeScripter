"""Backward-compatibility shim. Import from src.server.aeComms directly."""
import sys as _sys
import src.server.aeComms as _canonical
_sys.modules[__name__] = _canonical
