"""Backward-compatibility shim. Import from src.model.modelProtocol directly."""
import sys as _sys
import src.model.modelProtocol as _canonical
_sys.modules[__name__] = _canonical
