"""Put the bundle together in the output directory."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from .interpreter import GET_PIP_NAME, UNPACKED_MARKER, archiveSourceFor
from .settings import BuildSettings

DOCUMENT_NAMES = (
    "LICENSE",
    "README.md",
    "README.txt",
    "PARAMETERS.MD",
    "CHANGELOG.MD",
    "THIRD_PARTY_NOTICES.md",
)
PROFILE_SEED_NAME = ".dependencyCache.json"
MACOS_PROFILE_SEED = b'{"profile": "macos-mps"}'

LAUNCHER_NAME = "run.sh"
LAUNCHER_TEMPLATE = """#!/bin/bash
# The Anime Scripter - @LABEL@ Launcher Script

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set up the Python path
PYTHON_EXE="$SCRIPT_DIR/bin/python3"

# Check if Python executable exists
if [ ! -f "$PYTHON_EXE" ]; then
    echo "Error: Python executable not found at $PYTHON_EXE"
    echo "Please run the build script first: python build.py"
    exit 1
fi

# Run the main application
exec "$PYTHON_EXE" "$SCRIPT_DIR/main.py" "$@"
"""


def removePath(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def mergeTree(source: Path, target: Path) -> None:
    """Copy ``source`` over ``target``, keeping symbolic links as links.

    The standalone builds link bin/python3 to the versioned binary; following
    the link would ship the binary twice. An existing target is written over,
    which shutil.copytree cannot do for a link that is already there.
    """
    if target.is_symlink() or (target.exists() and not target.is_dir()):
        target.unlink()
    target.mkdir(parents=True, exist_ok=True)
    for entry in source.iterdir():
        landing = target / entry.name
        if entry.is_symlink():
            if landing.exists() or landing.is_symlink():
                removePath(landing)
            os.symlink(os.readlink(entry), landing)
        elif entry.is_dir():
            mergeTree(entry, landing)
        else:
            if landing.is_symlink() or landing.is_dir():
                removePath(landing)
            shutil.copy2(entry, landing)
    shutil.copystat(source, target)


def copyInterpreter(settings: BuildSettings, outputDir: Path) -> None:
    print("Copying Python installation...")
    for entry in settings.workDir.iterdir():
        landing = outputDir / entry.name
        if entry.is_dir() and not entry.is_symlink():
            mergeTree(entry, landing)
        elif entry.is_symlink():
            if landing.exists() or landing.is_symlink():
                removePath(landing)
            os.symlink(os.readlink(entry), landing)
        else:
            shutil.copy2(entry, landing)


def copySources(settings: BuildSettings, outputDir: Path) -> None:
    print("Copying source code...")
    sourceTarget = outputDir / "src"
    if sourceTarget.exists():
        # A develop build writes over a live install; stale modules must not survive.
        print(f"Removing existing src directory: {sourceTarget}")
        shutil.rmtree(sourceTarget)
    shutil.copytree(settings.root / "src", sourceTarget)
    shutil.copy2(settings.root / "main.py", outputDir / "main.py")


def copyRequirements(settings: BuildSettings, outputDir: Path) -> None:
    print("Copying requirements files...")
    for requirementsFile in settings.bundledRequirements:
        shutil.copy2(requirementsFile, outputDir / requirementsFile.name)


def launcherText(label: str) -> str:
    return LAUNCHER_TEMPLATE.replace("@LABEL@", label)


def writeLauncher(settings: BuildSettings, outputDir: Path) -> None:
    if settings.isWindows:
        return
    label = "macOS" if settings.isMacos else "Linux"
    launcher = outputDir / LAUNCHER_NAME
    launcher.write_bytes(launcherText(label).encode("utf-8"))
    os.chmod(launcher, 0o755)
    print(f"Created {label} launcher script: {LAUNCHER_NAME}")


def writeProfileSeed(settings: BuildSettings, outputDir: Path) -> None:
    """macOS has one runtime profile, so its bundle ships with it selected."""
    if settings.isMacos:
        (outputDir / PROFILE_SEED_NAME).write_bytes(MACOS_PROFILE_SEED)


def copyDocuments(settings: BuildSettings, targetDir: Path) -> None:
    for name in DOCUMENT_NAMES:
        document = settings.root / name
        if not document.exists():
            print(f"Skipping missing extra file: {document}")
            continue
        shutil.copy(document, targetDir / name)


def leftoverNames(settings: BuildSettings) -> tuple[str, ...]:
    return (
        archiveSourceFor(settings).fileName,
        GET_PIP_NAME,
        "license.txt",
        "wheel",
        UNPACKED_MARKER,
    )


def removeLeftovers(settings: BuildSettings, outputDir: Path) -> None:
    """Drop what came along with the interpreter copy but is not part of the bundle."""
    print("Cleaning up temporary files...")
    for name in leftoverNames(settings):
        path = outputDir / name
        if not path.exists() and not path.is_symlink():
            print(f"{path} does not exist, skipping removal.")
        elif path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
            print(f"Removed directory {path}")
        else:
            path.unlink()
            print(f"Removed {path}")


def assembleBundle(settings: BuildSettings, outputDir: Path) -> None:
    print("Creating portable bundle...")
    copyInterpreter(settings, outputDir)
    copySources(settings, outputDir)
    copyRequirements(settings, outputDir)
    writeLauncher(settings, outputDir)
    writeProfileSeed(settings, outputDir)
    print(f"Portable bundle created at {outputDir}")
    copyDocuments(settings, outputDir)
    removeLeftovers(settings, outputDir)
