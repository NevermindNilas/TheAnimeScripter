from pathlib import Path
import sys

_base_dir = Path(__file__).resolve().parent / "spandrel"
_spandrel_lib = _base_dir / "libs" / "spandrel"
_extra_arches_lib = _base_dir / "libs" / "spandrel_extra_arches"

for _path in (_spandrel_lib, _extra_arches_lib):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from spandrel import (  # type: ignore[attr-defined]
    ImageModelDescriptor,
    ModelLoader,
    UnsupportedDtypeError,
)

try:
    from spandrel_extra_arches import install as _install_extra_arches  # type: ignore[attr-defined]

    _install_extra_arches(ignore_duplicates=True)
except Exception:
    pass

__all__ = [
    "ImageModelDescriptor",
    "ModelLoader",
    "UnsupportedDtypeError",
]
