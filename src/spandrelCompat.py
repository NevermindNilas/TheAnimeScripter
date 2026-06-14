import sys
from pathlib import Path

_base_dir = Path(__file__).resolve().parent / "spandrel"
_spandrel_lib = _base_dir / "libs" / "spandrel"

_path_str = str(_spandrel_lib)
if _path_str not in sys.path:
    sys.path.insert(0, _path_str)

from spandrel import (  # type: ignore[attr-defined]  # noqa: E402  (sys.path shim above)
    ImageModelDescriptor,
    ModelLoader,
    UnsupportedDtypeError,
)

__all__ = [
    "ImageModelDescriptor",
    "ModelLoader",
    "UnsupportedDtypeError",
]
