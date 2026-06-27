from tools.build_support.context import BuildContext
from tools.build_support.process import run_subprocess


def install_requirements(context: BuildContext) -> None:
    print("Installing core requirements into the portable Python...")

    python_executable = (
        context.portable_python_dir / "python.exe"
        if context.system == "Windows"
        else context.portable_python_dir / "bin" / "python3"
    )

    if not python_executable.exists():
        raise FileNotFoundError(
            f"Portable Python executable not found: {python_executable}"
        )

    print("Bootstrapping build backend tooling...")
    run_subprocess(
        [
            str(python_executable),
            "-I",
            "-m",
            "pip",
            "install",
            "setuptools==81.0.0",
            "wheel",
            "--no-cache-dir",
            "--disable-pip-version-check",
        ],
    )

    run_subprocess(
        [
            str(python_executable),
            "-I",
            "-m",
            "pip",
            "install",
            "-r",
            str(context.requirements_path),
            "--no-build-isolation",
            "--no-cache-dir",
            "--disable-pip-version-check",
        ],
    )
    print("Core requirements installed successfully!")
