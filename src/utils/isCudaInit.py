"""Backward-compatibility shim. Import from src.infra.isCudaInit directly."""
import sys as _sys
import src.infra.isCudaInit as _canonical
_sys.modules[__name__] = _canonical
