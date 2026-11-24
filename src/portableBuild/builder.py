"""The build from start to finish."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from . import commands
from .assemble import assembleBundle
from .interpreter import installInterpreter
from .settings import BuildSettings

APP_FOLDER = "TheAnimeScripter"
PORTABLE_FOLDER = "TAS-Portable"
PIP_QUIET = ("--no-cache-dir", "--disable-pip-version-check")


def checkRequirementsFiles(settings: BuildSettings) -> None:
    for requirementsFile in settings.bundledRequirements:
        if not requirementsFile.exists():
            raise FileNotFoundError(f"Requirements file not found: {requirementsFile}")


def outputDirFor(settings: BuildSettings, develop: bool = False) -> Path:
    """The release directory, or the platform's development install directory."""
    if not develop:
        return settings.releaseParent / "main"
    if settings.isWindows:
        appData = os.environ.get("APPDATA")
        roaming = Path(appData) if appData else Path.home() / "AppData" / "Roaming"
        return roaming / APP_FOLDER
    if settings.isMacos:
        support = Path.home() / "Library" / "Application Support"
        return support / APP_FOLDER / PORTABLE_FOLDER
    return Path.home() / ".config" / APP_FOLDER / PORTABLE_FOLDER


def prepareOutputDir(outputDir: Path, develop: bool) -> None:
    # The develop directory is a live install (weights, dependencies): never wiped.
    if outputDir.exists() and not develop:
        print(f"Removing existing build directory: {outputDir}")
        shutil.rmtree(outputDir)
    outputDir.mkdir(parents=True, exist_ok=True)


def installCoreRequirements(settings: BuildSettings) -> None:
    """Install requirements.txt only; the runtime profile is installed on first run."""
    print("Installing core requirements into the portable Python...")
    python = settings.interpreter
    if not python.exists():
        raise FileNotFoundError(f"Portable Python executable not found: {python}")
    pip = [str(python), "-I", "-m", "pip", "install"]

    print("Bootstrapping build backend tooling...")
    commands.runCommand([*pip, "setuptools==81.0.0", "wheel", *PIP_QUIET])
    commands.runCommand(
        [
            *pip,
            "-r",
            str(settings.coreRequirements),
            "--no-build-isolation",
            *PIP_QUIET,
        ]
    )
    print("Core requirements installed successfully!")


def removeWorkDir(settings: BuildSettings) -> None:
    workDir = settings.workDir
    if workDir.exists():
        shutil.rmtree(workDir)
        print(f"Removed {workDir}")
    else:
        print(f"{workDir} does not exist, skipping removal.")


def buildPortable(settings: BuildSettings, develop: bool = False) -> Path:
    """Build the portable bundle and return the directory it was written to.

    A failure leaves the work directory in place so the next build resumes.
    """
    checkRequirementsFiles(settings)
    outputDir = outputDirFor(settings, develop)
    prepareOutputDir(outputDir, develop)
    installInterpreter(settings)
    installCoreRequirements(settings)
    assembleBundle(settings, outputDir)
    removeWorkDir(settings)
    print("Bundle process completed successfully!")
    print(f"Portable bundle is ready at {outputDir}")
    return outputDir
