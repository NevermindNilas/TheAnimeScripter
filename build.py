import subprocess
import shutil
import os
import platform
import tarfile
from pathlib import Path
import urllib.request
import zipfile
import argparse

baseDir = Path(__file__).resolve().parent
distPath = baseDir / "dist-portable"
pyprojectPath = baseDir / "pyproject.toml"
requirementsPath = baseDir / "requirements.txt"
requirementsFiles = [
    requirementsPath,
    baseDir / "extra-requirements-windows.txt",
    baseDir / "extra-requirements-windows-lite.txt",
    baseDir / "extra-requirements-linux.txt",
    baseDir / "extra-requirements-linux-lite.txt",
    baseDir / "extra-requirements-macos.txt",
    baseDir / "deprecated-requirements.txt",
]

for requirementsFile in requirementsFiles:
    if not requirementsFile.exists():
        raise FileNotFoundError(f"Requirements file not found: {requirementsFile}")

portablePythonDir = baseDir / "portable-python"
pythonVersion = "3.13.12"
system = platform.system()


def runSubprocess(command, shell=False, cwd=None, stdout=None, env=None):
    try:
        subprocess.run(
            command,
            shell=shell,
            check=True,
            cwd=cwd,
            stdout=stdout,
            env=env,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error while running command {command}: {e}")
        raise


def downloadPortablePython():
    """Download and setup Python for the target platform"""
    print(f"Setting up portable Python {pythonVersion} for {system}...")

    os.makedirs(portablePythonDir, exist_ok=True)

    if system == "Windows":
        return downloadPortablePythonWindows()
    if system == "Darwin":
        return downloadPortablePythonMacos()
    return downloadPortablePythonLinux()


def downloadPortablePythonWindows():
    """Download the embeddable Python package for Windows"""
    pythonUrl = f"https://www.python.org/ftp/python/{pythonVersion}/python-{pythonVersion}-embed-amd64.zip"
    getPipUrl = "https://bootstrap.pypa.io/get-pip.py"

    pythonZip = portablePythonDir / "python.zip"
    if not pythonZip.exists():
        print("Downloading Python embeddable package...")
        urllib.request.urlretrieve(pythonUrl, pythonZip)

    if not (portablePythonDir / "python.exe").exists():
        print("Extracting Python...")
        with zipfile.ZipFile(pythonZip, "r") as zipRef:
            zipRef.extractall(portablePythonDir)

    getPipPath = portablePythonDir / "get-pip.py"
    if not getPipPath.exists():
        print("Downloading get-pip.py...")
        urllib.request.urlretrieve(getPipUrl, getPipPath)

    pthFiles = list(portablePythonDir.glob("python*._pth"))
    if pthFiles:
        pthFile = pthFiles[0]
        with open(pthFile, "r") as f:
            content = f.read()

        if "#import site" in content:
            content = content.replace("#import site", "import site")
            with open(pthFile, "w") as f:
                f.write(content)

    print("Installing pip...")
    runSubprocess(
        [str(portablePythonDir / "python.exe"), str(getPipPath)], cwd=portablePythonDir
    )

    print("Portable Python installation complete!")
    return portablePythonDir / "python.exe"


def downloadPortablePythonLinux():
    """Download and setup Python for Linux"""
    # For Linux, we'll download a portable Python build or use pyenv-like approach
    pythonUrl = f"https://github.com/indygreg/python-build-standalone/releases/download/20241016/cpython-{pythonVersion}+20241016-x86_64-unknown-linux-gnu-install_only.tar.gz"
    getPipUrl = "https://bootstrap.pypa.io/get-pip.py"

    pythonTar = portablePythonDir / "python.tar.gz"
    if not pythonTar.exists():
        print("Downloading Python standalone build for Linux...")
        urllib.request.urlretrieve(pythonUrl, pythonTar)

    pythonExe = portablePythonDir / "bin" / "python3"
    if not pythonExe.exists():
        print("Extracting Python...")
        with tarfile.open(pythonTar, "r:gz") as tarRef:
            tarRef.extractall(portablePythonDir)

        # Make Python executable
        if pythonExe.exists():
            os.chmod(pythonExe, 0o755)

    getPipPath = portablePythonDir / "get-pip.py"
    if not getPipPath.exists():
        print("Downloading get-pip.py...")
        urllib.request.urlretrieve(getPipUrl, getPipPath)

    print("Installing pip...")
    runSubprocess([str(pythonExe), str(getPipPath)], cwd=portablePythonDir)

    print("Portable Python installation complete!")
    return pythonExe


def downloadPortablePythonMacos():
    """Download and setup Python for macOS (Apple Silicon arm64)."""
    machine = platform.machine().lower()
    if machine not in ("arm64", "aarch64"):
        raise RuntimeError(
            f"Unsupported macOS architecture '{machine}'. Only Apple Silicon (arm64) is supported."
        )

    pythonUrl = (
        f"https://github.com/astral-sh/python-build-standalone/releases/download/"
        f"20260325/cpython-{pythonVersion}+20260325-aarch64-apple-darwin-install_only.tar.gz"
    )
    getPipUrl = "https://bootstrap.pypa.io/get-pip.py"

    pythonTar = portablePythonDir / "python.tar.gz"
    if not pythonTar.exists():
        print("Downloading Python standalone build for macOS (arm64)...")
        urllib.request.urlretrieve(pythonUrl, pythonTar)

    pythonExe = portablePythonDir / "bin" / "python3"
    if not pythonExe.exists():
        print("Extracting Python...")
        with tarfile.open(pythonTar, "r:gz") as tarRef:
            tarRef.extractall(portablePythonDir)

        if pythonExe.exists():
            os.chmod(pythonExe, 0o755)

    getPipPath = portablePythonDir / "get-pip.py"
    if not getPipPath.exists():
        print("Downloading get-pip.py...")
        urllib.request.urlretrieve(getPipUrl, getPipPath)

    print("Installing pip...")
    runSubprocess([str(pythonExe), str(getPipPath)], cwd=portablePythonDir)

    print("Portable Python installation complete!")
    return pythonExe


def installRequirements():
    print("Installing core requirements into the portable Python...")

    pythonExecutable = (
        portablePythonDir / "python.exe"
        if system == "Windows"
        else portablePythonDir / "bin" / "python3"
    )

    if not pythonExecutable.exists():
        raise FileNotFoundError(
            f"Portable Python executable not found: {pythonExecutable}"
        )

    print("Bootstrapping build backend tooling...")
    runSubprocess(
        [
            str(pythonExecutable),
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

    runSubprocess(
        [
            str(pythonExecutable),
            "-I",
            "-m",
            "pip",
            "install",
            "-r",
            str(requirementsPath),
            "--no-build-isolation",
            "--no-cache-dir",
            "--disable-pip-version-check",
        ],
    )
    print("Core requirements installed successfully!")


def bundleFiles(targetDir):
    print("Creating portable bundle...")

    bundleDir = targetDir

    print("Copying Python installation...")
    for item in portablePythonDir.iterdir():
        if item.is_dir():
            shutil.copytree(item, bundleDir / item.name, dirs_exist_ok=True)
        else:
            shutil.copy2(item, bundleDir / item.name)

    print("Copying source code...")
    srcDir = baseDir / "src"
    srcDest = bundleDir / "src"

    if srcDest.exists():
        print(f"Removing existing src directory: {srcDest}")
        shutil.rmtree(srcDest)

    shutil.copytree(srcDir, srcDest)

    shutil.copy2(baseDir / "main.py", bundleDir / "main.py")
    if pyprojectPath.exists():
        shutil.copy2(pyprojectPath, bundleDir / "pyproject.toml")

    print("Copying requirements files...")
    for requirementsFile in requirementsFiles:
        shutil.copy2(requirementsFile, bundleDir / requirementsFile.name)

    if system in ("Linux", "Darwin"):
        launcherScript = bundleDir / "run.sh"
        platformLabel = "Linux" if system == "Linux" else "macOS"
        with open(launcherScript, "w") as f:
            f.write(f"""#!/bin/bash
# The Anime Scripter - {platformLabel} Launcher Script

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"

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
""")
        os.chmod(launcherScript, 0o755)
        print(f"Created {platformLabel} launcher script: run.sh")

    print(f"Portable bundle created at {bundleDir}")


def moveExtras(targetDir):
    bundleDir = targetDir
    filesToCopy = [
        "LICENSE",
        "README.md",
        "README.txt",
        "PARAMETERS.MD",
        "CHANGELOG.MD",
    ]

    for fileName in filesToCopy:
        sourcePath = baseDir / fileName
        if not sourcePath.exists():
            print(f"Skipping missing extra file: {sourcePath}")
            continue

        shutil.copy(sourcePath, bundleDir)


def cleanupTempFiles(targetDir):
    """Remove temporary files created during the build process"""
    print("Cleaning up temporary files...")
    bundleDir = targetDir

    if system == "Windows":
        tempFiles = [
            "python.zip",
            "get-pip.py",
            "license.txt",
            "wheel",
        ]
    else:
        tempFiles = [
            "python.tar.gz",
            "get-pip.py",
            "license.txt",
            "wheel",
        ]

    for tempFile in tempFiles:
        tempFilePath = bundleDir / tempFile
        if tempFilePath.exists():
            if tempFilePath.is_dir():
                shutil.rmtree(tempFilePath)
                print(f"Removed directory {tempFilePath}")
            else:
                os.remove(tempFilePath)
                print(f"Removed {tempFilePath}")
        else:
            print(f"{tempFilePath} does not exist, skipping removal.")


def removePortablePython():
    """Remove the portable Python directory"""
    if portablePythonDir.exists():
        shutil.rmtree(portablePythonDir)
        print(f"Removed {portablePythonDir}")
    else:
        print(f"{portablePythonDir} does not exist, skipping removal.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build script for The Anime Scripter portable version."
    )
    parser.add_argument(
        "--develop",
        action="store_true",
        help="If active, it will overwrite the contents of F:\\TheAnimeScripter\\dist-portable\\main with the newly generated build. ONLY USE IN DEVELOPMENT!",
    )
    args = parser.parse_args()

    if args.develop:
        if system == "Windows":
            finalOutputDir = Path(
                r"D:\tastest\TheAnimeScripter"
            )
        elif system == "Darwin":
            finalOutputDir = (
                Path.home() / "Library" / "Application Support" / "TheAnimeScripter" / "TAS-Portable"
            )
        else:
            # Linux development path
            finalOutputDir = (
                Path.home() / ".config" / "TheAnimeScripter" / "TAS-Portable"
            )
    else:
        finalOutputDir = distPath / "main"

    if finalOutputDir.exists() and not args.develop:
        # Just so it doesn't randomly delete TAS-Portable
        print(f"Removing existing build directory: {finalOutputDir}")
        shutil.rmtree(finalOutputDir)

    os.makedirs(finalOutputDir.parent, exist_ok=True)
    os.makedirs(finalOutputDir, exist_ok=True)

    pythonExe = downloadPortablePython()
    installRequirements()
    bundleFiles(finalOutputDir)
    moveExtras(finalOutputDir)
    cleanupTempFiles(finalOutputDir)
    removePortablePython()
    print("Bundle process completed successfully!")
    print(f"Portable bundle is ready at {finalOutputDir}")
