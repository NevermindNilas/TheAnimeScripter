"""Download, unpack and pip-bootstrap the interpreter in the work directory.

The work directory survives a failed build; a later build reuses the finished
downloads and a finished unpack, and redoes anything else.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from pathlib import Path

from . import archives, commands, downloads
from .settings import BuildSettings

GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"
GET_PIP_NAME = "get-pip.py"
# Written once unpacking has fully finished. The interpreter is among the first
# archive members, so its presence alone does not prove a complete tree.
UNPACKED_MARKER = ".tas-extracted"
STANDALONE_RELEASES = (
    "https://github.com/astral-sh/python-build-standalone/releases/download"
)


@dataclass(frozen=True)
class ArchiveSource:
    url: str
    fileName: str
    notice: str


def archiveSourceFor(settings: BuildSettings) -> ArchiveSource:
    version = settings.pythonVersion
    if settings.isWindows:
        return ArchiveSource(
            url=f"https://www.python.org/ftp/python/{version}/python-{version}-embed-amd64.zip",
            fileName="python.zip",
            notice="Downloading Python embeddable package...",
        )
    tag = settings.standaloneTag
    if settings.isMacos:
        target = "aarch64-apple-darwin"
        notice = "Downloading Python standalone build for macOS (arm64)..."
    else:
        target = "x86_64-unknown-linux-gnu"
        notice = "Downloading Python standalone build for Linux..."
    return ArchiveSource(
        url=f"{STANDALONE_RELEASES}/{tag}/cpython-{version}+{tag}-{target}-install_only.tar.gz",
        fileName="python.tar.gz",
        notice=notice,
    )


def requireAppleSilicon() -> None:
    machine = platform.machine().lower()
    if machine not in ("arm64", "aarch64"):
        raise RuntimeError(
            f"Unsupported macOS architecture '{machine}'. Only Apple Silicon (arm64) is supported."
        )


def enableSitePackages(workDir: Path) -> None:
    """Uncomment ``import site`` in the embeddable distribution's ``._pth`` file.

    Without it the freshly installed pip is not importable.
    """
    pthFiles = sorted(workDir.glob("python*._pth"))
    if not pthFiles:
        return
    text = pthFiles[0].read_text()
    if "#import site" in text:
        pthFiles[0].write_text(text.replace("#import site", "import site"))


def unpackInterpreter(settings: BuildSettings, archivePath: Path) -> None:
    workDir = settings.workDir
    marker = workDir / UNPACKED_MARKER
    python = settings.interpreter
    if python.exists() and marker.exists():
        return

    print("Extracting Python...")
    marker.unlink(missing_ok=True)
    if settings.isWindows:
        archives.unpackZip(archivePath, workDir)
    else:
        archives.unpackTarball(archivePath, workDir)
        archives.liftStandaloneFolder(workDir)
        if python.exists():
            os.chmod(python, 0o755)
    marker.write_text("ok")


def installInterpreter(settings: BuildSettings) -> Path:
    """Leave a pip-equipped interpreter in the work directory and return its path."""
    print(
        f"Setting up portable Python {settings.pythonVersion} for {settings.systemName}..."
    )
    workDir = settings.workDir
    workDir.mkdir(parents=True, exist_ok=True)
    if settings.isMacos:
        requireAppleSilicon()

    source = archiveSourceFor(settings)
    archivePath = workDir / source.fileName
    downloads.fetchIfMissing(source.url, archivePath, source.notice)
    unpackInterpreter(settings, archivePath)

    getPip = workDir / GET_PIP_NAME
    downloads.fetchIfMissing(GET_PIP_URL, getPip, "Downloading get-pip.py...")
    if settings.isWindows:
        enableSitePackages(workDir)

    python = settings.interpreter
    print("Installing pip...")
    commands.runCommand([str(python), str(getPip)], cwd=workDir)

    print("Portable Python installation complete!")
    return python
