from tools.build_support.context import BuildContext
from tools.build_support.process import run_subprocess

EXTRA_REQUIREMENTS_BY_SYSTEM = {
    "Darwin": "extra-requirements-macos.txt",
}


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

    extra_requirements_name = EXTRA_REQUIREMENTS_BY_SYSTEM.get(context.system)
    if extra_requirements_name is not None:
        extra_requirements_path = context.base_dir / extra_requirements_name
        print(f"Installing {context.system} runtime requirements...")
        run_subprocess(
            [
                str(python_executable),
                "-I",
                "-m",
                "pip",
                "install",
                "-c",
                str(context.requirements_path),
                "-r",
                str(extra_requirements_path),
                "--no-build-isolation",
                "--no-cache-dir",
                "--disable-pip-version-check",
            ],
        )

    print("Requirements installed successfully!")
