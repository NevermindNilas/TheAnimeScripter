"""Backward-compatibility shim. Import from src.infra.logAndPrint directly."""
import sys as _sys
import src.infra.logAndPrint as _canonical
_sys.modules[__name__] = _canonical
