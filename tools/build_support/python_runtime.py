import os
import platform
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

from tools.build_support.context import BuildContext
from tools.build_support.process import run_subprocess


def download_portable_python(context: BuildContext) -> Path:
    """Download and set up Python for the target platform."""
    print(
        f"Setting up portable Python {context.python_version} for {context.system}..."
    )

    os.makedirs(context.portable_python_dir, exist_ok=True)

    if context.system == "Windows":
        return download_portable_python_windows(context)
    if context.system == "Darwin":
        return download_portable_python_macos(context)
    return download_portable_python_linux(context)


def download_portable_python_windows(context: BuildContext) -> Path:
    python_url = (
        f"https://www.python.org/ftp/python/{context.python_version}/"
        f"python-{context.python_version}-embed-amd64.zip"
    )
    get_pip_url = "https://bootstrap.pypa.io/get-pip.py"

    python_zip = context.portable_python_dir / "python.zip"
    if not python_zip.exists():
        print("Downloading Python embeddable package...")
        urllib.request.urlretrieve(python_url, python_zip)

    python_exe = context.portable_python_dir / "python.exe"
    if not python_exe.exists():
        print("Extracting Python...")
        with zipfile.ZipFile(python_zip, "r") as zip_ref:
            zip_ref.extractall(context.portable_python_dir)

    get_pip_path = context.portable_python_dir / "get-pip.py"
    if not get_pip_path.exists():
        print("Downloading get-pip.py...")
        urllib.request.urlretrieve(get_pip_url, get_pip_path)

    pth_files = list(context.portable_python_dir.glob("python*._pth"))
    if pth_files:
        pth_file = pth_files[0]
        with open(pth_file) as f:
            content = f.read()

        if "#import site" in content:
            content = content.replace("#import site", "import site")
            with open(pth_file, "w") as f:
                f.write(content)

    print("Installing pip...")
    run_subprocess(
        [str(python_exe), str(get_pip_path)], cwd=context.portable_python_dir
    )

    print("Portable Python installation complete!")
    return python_exe


def flatten_standalone_python_dir(target_dir: Path) -> None:
    """Move python-build-standalone install_only contents to target_dir."""
    nested = target_dir / "python"
    if not nested.is_dir():
        return
    for item in nested.iterdir():
        dest = target_dir / item.name
        if dest.exists():
            shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
        shutil.move(str(item), str(dest))
    nested.rmdir()


def download_portable_python_linux(context: BuildContext) -> Path:
    python_url = (
        "https://github.com/astral-sh/python-build-standalone/releases/download/"
        f"{context.standalone_release}/cpython-{context.python_version}+"
        f"{context.standalone_release}-x86_64-unknown-linux-gnu-install_only.tar.gz"
    )
    get_pip_url = "https://bootstrap.pypa.io/get-pip.py"

    python_tar = context.portable_python_dir / "python.tar.gz"
    if not python_tar.exists():
        print("Downloading Python standalone build for Linux...")
        urllib.request.urlretrieve(python_url, python_tar)

    python_exe = context.portable_python_dir / "bin" / "python3"
    if not python_exe.exists():
        print("Extracting Python...")
        with tarfile.open(python_tar, "r:gz") as tar_ref:
            tar_ref.extractall(context.portable_python_dir)

        flatten_standalone_python_dir(context.portable_python_dir)

        if python_exe.exists():
            os.chmod(python_exe, 0o755)

    get_pip_path = context.portable_python_dir / "get-pip.py"
    if not get_pip_path.exists():
        print("Downloading get-pip.py...")
        urllib.request.urlretrieve(get_pip_url, get_pip_path)

    print("Installing pip...")
    run_subprocess(
        [str(python_exe), str(get_pip_path)], cwd=context.portable_python_dir
    )

    print("Portable Python installation complete!")
    return python_exe


def download_portable_python_macos(context: BuildContext) -> Path:
    machine = platform.machine().lower()
    if machine not in ("arm64", "aarch64"):
        raise RuntimeError(
            f"Unsupported macOS architecture '{machine}'. "
            "Only Apple Silicon (arm64) is supported."
        )

    python_url = (
        "https://github.com/astral-sh/python-build-standalone/releases/download/"
        f"{context.standalone_release}/cpython-{context.python_version}+"
        f"{context.standalone_release}-aarch64-apple-darwin-install_only.tar.gz"
    )
    get_pip_url = "https://bootstrap.pypa.io/get-pip.py"

    python_tar = context.portable_python_dir / "python.tar.gz"
    if not python_tar.exists():
        print("Downloading Python standalone build for macOS (arm64)...")
        urllib.request.urlretrieve(python_url, python_tar)

    python_exe = context.portable_python_dir / "bin" / "python3"
    if not python_exe.exists():
        print("Extracting Python...")
        with tarfile.open(python_tar, "r:gz") as tar_ref:
            tar_ref.extractall(context.portable_python_dir)

        flatten_standalone_python_dir(context.portable_python_dir)

        if python_exe.exists():
            os.chmod(python_exe, 0o755)

    get_pip_path = context.portable_python_dir / "get-pip.py"
    if not get_pip_path.exists():
        print("Downloading get-pip.py...")
        urllib.request.urlretrieve(get_pip_url, get_pip_path)

    print("Installing pip...")
    run_subprocess(
        [str(python_exe), str(get_pip_path)], cwd=context.portable_python_dir
    )

    print("Portable Python installation complete!")
    return python_exe
