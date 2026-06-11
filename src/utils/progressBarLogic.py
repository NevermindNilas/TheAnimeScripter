"""Backward-compatibility shim. Import from src.infra.progressBarLogic directly."""
import sys as _sys
import src.infra.progressBarLogic as _canonical
_sys.modules[__name__] = _canonical
