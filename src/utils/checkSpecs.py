"""Backward-compatibility shim. Import from src.infra.checkSpecs directly."""
import sys as _sys
import src.infra.checkSpecs as _canonical
_sys.modules[__name__] = _canonical
