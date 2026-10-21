import functools
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

import src.constants as cs

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
    "loadSpandrelModel",
]

ARCH_CACHE_NAME = ".spandrelArchCache.json"

# Which architecture claims a file depends on the registry's order and on every
# earlier architecture's detect(), so a record is only trusted while no source
# file of the spandrel package has changed.
_PACKAGE_DIR = _spandrel_lib / "spandrel"


@functools.cache
def _packageFingerprint() -> str | None:
    """Digest of every .py file's path, size and mtime in the spandrel package.

    ~2 ms for its ~190 files, once per process.
    """
    entries = []
    stack = [str(_PACKAGE_DIR)]
    try:
        while stack:
            folder = stack.pop()
            with os.scandir(folder) as it:
                for entry in it:
                    if entry.is_dir(follow_symlinks=False):
                        if entry.name != "__pycache__":
                            stack.append(entry.path)
                    elif entry.name.endswith(".py"):
                        st = entry.stat()
                        rel = os.path.relpath(entry.path, _PACKAGE_DIR)
                        entries.append(f"{rel}|{st.st_size}|{st.st_mtime_ns}")
    except OSError:
        return None
    return hashlib.sha1("\n".join(sorted(entries)).encode()).hexdigest()


def _cachePath() -> str | None:
    root = cs.WHEREAMIRUNFROM
    return os.path.join(root, ARCH_CACHE_NAME) if root else None


def _readCache() -> dict:
    path = _cachePath()
    if not path:
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            cache = json.load(f)
        return cache if isinstance(cache, dict) else {}
    except OSError, ValueError:
        return {}


def _fingerprint(path: str) -> dict | None:
    package = _packageFingerprint()
    try:
        st = os.stat(path)
    except OSError:
        return None
    if package is None:
        return None
    return {"size": st.st_size, "mtime": st.st_mtime_ns, "spandrel": package}


def _cacheKey(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def _recordedArch(path: str) -> str | None:
    fingerprint = _fingerprint(path)
    entry = _readCache().get(_cacheKey(path))
    if fingerprint is None or not isinstance(entry, dict):
        return None
    if {k: entry.get(k) for k in fingerprint} != fingerprint:
        return None
    arch = entry.get("arch")
    return arch if isinstance(arch, str) else None


def _record(path: str, descriptor) -> None:
    cachePath = _cachePath()
    fingerprint = _fingerprint(path)
    archType = type(descriptor.architecture)
    if not cachePath or fingerprint is None:
        return
    if not archType.__module__.startswith("spandrel.architectures."):
        return
    cache = _readCache()
    cache[_cacheKey(path)] = {
        **fingerprint,
        "arch": f"{archType.__module__}:{archType.__qualname__}",
    }
    # Written whole and swapped in: an AE render can launch TAS back to back.
    tmp = f"{cachePath}.{os.getpid()}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cache, f)
        os.replace(tmp, cachePath)
    except OSError as e:
        logging.debug(f"Could not write {cachePath}: {e}")
        try:
            os.remove(tmp)
        except OSError:
            pass


def _oneArchLoader(arch: str):
    """A ModelLoader whose registry holds only ``arch`` ("module:Class").

    MAIN_REGISTRY registers every architecture through
    ArchSupport.from_architecture, so this detects and loads exactly as the
    full registry does once it reaches that architecture.
    """
    from importlib import import_module

    from spandrel import ArchRegistry, ArchSupport

    moduleName, _, className = arch.partition(":")
    registry = ArchRegistry()
    registry.add(
        ArchSupport.from_architecture(getattr(import_module(moduleName), className)())
    )
    return ModelLoader(registry=registry)


def _load(loader, path: str, stateDict):
    if stateDict is not None:
        return loader.load_from_state_dict(stateDict)
    return loader.load_from_file(path)


def loadSpandrelModel(path: str, stateDict=None):
    """Load the model at ``path`` (or ``stateDict``, read from it) through spandrel.

    The first load of a file runs the full registry, which imports all ~47
    architectures (~70 ms), and records which one claimed it. Later loads of
    the same, unchanged file import only that architecture. A record that no
    longer matches falls back to the full registry, so it can cost time but
    never pick a different model.
    """
    from spandrel import UnsupportedModelError

    arch = _recordedArch(path)
    if arch is not None:
        try:
            return _load(_oneArchLoader(arch), path, stateDict)
        except UnsupportedModelError, ImportError, AttributeError:
            logging.info(f"Recorded architecture {arch} no longer loads {path}")

    descriptor = _load(ModelLoader(), path, stateDict)
    _record(path, descriptor)
    return descriptor
