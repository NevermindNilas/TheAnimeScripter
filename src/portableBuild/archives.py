"""Unpacking the interpreter archives without writing outside the destination."""

from __future__ import annotations

import shutil
import tarfile
import zipfile
from pathlib import Path


def checkMemberInside(destination: Path | str, memberName: str) -> None:
    """Raise ValueError if ``memberName`` would land outside ``destination``."""
    base = Path(destination).resolve()
    # Backslashes are separators on Windows only; check both readings.
    for name in {memberName, memberName.replace("\\", "/")}:
        landing = (base / name).resolve()
        if landing != base and base not in landing.parents:
            raise ValueError(
                f"Archive member {memberName!r} would be written outside {base}"
            )


def unpackZip(archivePath: Path, destination: Path) -> None:
    with zipfile.ZipFile(archivePath) as archive:
        for name in archive.namelist():
            checkMemberInside(destination, name)
        archive.extractall(destination)


def unpackTarball(archivePath: Path, destination: Path) -> None:
    with tarfile.open(archivePath, "r:gz") as archive:
        archive.extractall(destination, filter="data")


def liftStandaloneFolder(workDir: Path) -> None:
    """Move the entries of ``workDir/python`` up one level and drop the folder.

    The standalone tarballs hold a single ``python/`` folder. An entry left in
    ``workDir`` by an interrupted run is replaced.
    """
    nested = workDir / "python"
    if not nested.is_dir():
        return
    for entry in list(nested.iterdir()):
        landing = workDir / entry.name
        if landing.is_dir() and not landing.is_symlink():
            shutil.rmtree(landing)
        elif landing.exists() or landing.is_symlink():
            landing.unlink()
        shutil.move(str(entry), str(landing))
    nested.rmdir()
