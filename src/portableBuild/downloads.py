"""All-or-nothing downloads."""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path


def fetchFile(url: str, target: Path) -> None:
    """Download ``url`` so that ``target`` only ever exists complete.

    A skipped download is decided by ``target`` existing, so a truncated file
    under that name would be reused by every later build.
    """
    partial = target.with_name(target.name + ".part")
    try:
        urllib.request.urlretrieve(url, partial)
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)


def fetchIfMissing(url: str, target: Path, notice: str) -> None:
    if target.exists():
        return
    print(notice)
    fetchFile(url, target)
