import platform
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BuildContext:
    base_dir: Path
    dist_path: Path
    requirements_path: Path
    requirements_files: list[Path]
    portable_python_dir: Path
    python_version: str
    standalone_release: str
    system: str


def create_build_context(base_dir: Path | None = None) -> BuildContext:
    root = base_dir or Path(__file__).resolve().parents[2]
    requirements_path = root / "requirements.txt"
    requirements_files = [
        requirements_path,
        root / "extra-requirements-windows.txt",
        root / "extra-requirements-windows-lite.txt",
        root / "extra-requirements-linux.txt",
        root / "extra-requirements-linux-lite.txt",
        root / "extra-requirements-macos.txt",
        root / "extra-requirements-macos-lite.txt",
        root / "deprecated-requirements.txt",
    ]

    return BuildContext(
        base_dir=root,
        dist_path=root / "dist-portable",
        requirements_path=requirements_path,
        requirements_files=requirements_files,
        portable_python_dir=root / "portable-python",
        python_version="3.14.5",
        # python-build-standalone release tag that ships python_version.
        standalone_release="20260510",
        system=platform.system(),
    )


def validate_requirements_files(context: BuildContext) -> None:
    for requirements_file in context.requirements_files:
        if not requirements_file.exists():
            raise FileNotFoundError(f"Requirements file not found: {requirements_file}")
