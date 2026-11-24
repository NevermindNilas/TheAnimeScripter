"""What one portable build runs on: the two pins, the paths and the target system.

The interpreter version and the standalone release tag are written here and
nowhere else in the package; every download URL is composed from the settings.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass
from pathlib import Path

# Must equal the python-version the Build-* workflows pin.
BUNDLED_PYTHON_VERSION = "3.14.10"
# The python-build-standalone release that ships BUNDLED_PYTHON_VERSION.
STANDALONE_RELEASE_TAG = "20261110"

CORE_REQUIREMENTS_NAME = "requirements.txt"
BUNDLED_REQUIREMENTS_NAMES = (
    CORE_REQUIREMENTS_NAME,
    "extra-requirements-windows.txt",
    "extra-requirements-windows-lite.txt",
    "extra-requirements-windows-rocm.txt",
    "extra-requirements-linux.txt",
    "extra-requirements-linux-lite.txt",
    "extra-requirements-linux-rocm.txt",
    "extra-requirements-macos.txt",
    "extra-requirements-macos-lite.txt",
    "deprecated-requirements.txt",
)


@dataclass(frozen=True)
class BuildSettings:
    root: Path
    systemName: str
    pythonVersion: str
    standaloneTag: str
    releaseParent: Path
    workDir: Path
    coreRequirements: Path
    bundledRequirements: tuple[Path, ...]

    @property
    def isWindows(self) -> bool:
        return self.systemName == "Windows"

    @property
    def isMacos(self) -> bool:
        return self.systemName == "Darwin"

    def interpreterIn(self, directory: Path) -> Path:
        """Where the interpreter sits inside an unpacked tree or a bundle."""
        if self.isWindows:
            return directory / "python.exe"
        return directory / "bin" / "python3"

    @property
    def interpreter(self) -> Path:
        return self.interpreterIn(self.workDir)


def repositoryRoot() -> Path:
    """The checkout this package lives in (<root>/src/portableBuild/settings.py)."""
    return Path(__file__).resolve().parents[2]


def defaultSettings(
    root: Path | str | None = None,
    systemName: str | None = None,
    **overrides,
) -> BuildSettings:
    """Settings for ``root`` (default: this checkout) on ``systemName`` (default: the host).

    Any BuildSettings field may be overridden by keyword.
    """
    rootDir = Path(root) if root is not None else repositoryRoot()
    values = {
        "root": rootDir,
        "systemName": systemName if systemName is not None else platform.system(),
        "pythonVersion": BUNDLED_PYTHON_VERSION,
        "standaloneTag": STANDALONE_RELEASE_TAG,
        "releaseParent": rootDir / "dist-portable",
        "workDir": rootDir / "portable-python",
        "coreRequirements": rootDir / CORE_REQUIREMENTS_NAME,
        "bundledRequirements": tuple(
            rootDir / name for name in BUNDLED_REQUIREMENTS_NAMES
        ),
    }
    unknown = sorted(set(overrides) - set(values))
    if unknown:
        raise TypeError(f"Unknown build setting(s): {', '.join(unknown)}")
    values.update(overrides)
    values["bundledRequirements"] = tuple(
        Path(item) for item in values["bundledRequirements"]
    )
    return BuildSettings(**values)
