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
requirementsPath = baseDir / "requirements.txt"

if not requirementsPath.exists():
    raise FileNotFoundError(f"Requirements file not found: {requirementsPath}")

portablePythonDir = baseDir / "portable-python"
pythonVersion = "3.13.7"
system = platform.system()


def runSubprocess(command, shell=False, cwd=None):
    try:
        subprocess.run(command, shell=shell, check=True, cwd=cwd)
    except subprocess.CalledProcessError as e:
        print(f"Error while running command {command}: {e}")
        raise


def downloadPortablePython():
    """Download and setup Python for the target platform"""
    print(f"Setting up portable Python {pythonVersion} for {system}...")

    os.makedirs(portablePythonDir, exist_ok=True)

    if system == "Windows":
        return downloadPortablePythonWindows()
    else:
        return downloadPortablePythonLinux()


def downloadPortablePythonWindows():
    """Download the embeddable Python package for Windows"""
    pythonUrl = f"https://www.python.org/ftp/python/{pythonVersion}/python-{pythonVersion}-embed-amd64.zip"
    getPiPUrl = "https://bootstrap.pypa.io/get-pip.py"

    pythonZip = portablePythonDir / "python.zip"
    if not pythonZip.exists():
        print("Downloading Python embeddable package...")
        urllib.request.urlretrieve(pythonUrl, pythonZip)

    if not (portablePythonDir / "python.exe").exists():
        print("Extracting Python...")
        with zipfile.ZipFile(pythonZip, "r") as zipRef:
            zipRef.extractall(portablePythonDir)

    getPiPPath = portablePythonDir / "get-pip.py"
    if not getPiPPath.exists():
        print("Downloading get-pip.py...")
        urllib.request.urlretrieve(getPiPUrl, getPiPPath)

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
        [str(portablePythonDir / "python.exe"), "get-pip.py"], cwd=portablePythonDir
    )

    print("Portable Python installation complete!")
    return portablePythonDir / "python.exe"


def downloadPortablePythonLinux():
    """Download and setup Python for Linux"""
    # For Linux, we'll download a portable Python build or use pyenv-like approach
    pythonUrl = f"https://github.com/indygreg/python-build-standalone/releases/download/20241016/cpython-{pythonVersion}+20241016-x86_64-unknown-linux-gnu-install_only.tar.gz"
    getPiPUrl = "https://bootstrap.pypa.io/get-pip.py"

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

    getPiPPath = portablePythonDir / "get-pip.py"
    if not getPiPPath.exists():
        print("Downloading get-pip.py...")
        urllib.request.urlretrieve(getPiPUrl, getPiPPath)

    print("Installing pip...")
    runSubprocess([str(pythonExe), str(getPiPPath)], cwd=portablePythonDir)

    print("Portable Python installation complete!")
    return pythonExe


def installRequirements():
    print("Installing the requirements...")

    if system == "Windows":
        pipExe = portablePythonDir / "Scripts" / "pip.exe"
    else:
        pipExe = portablePythonDir / "bin" / "pip3"
        if not pipExe.exists():
            pipExe = portablePythonDir / "bin" / "pip"

    runSubprocess([str(pipExe), "install", "-r", str(requirementsPath)])
    print("Requirements installation complete!")


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
    shutil.copytree(srcDir, bundleDir / "src", dirs_exist_ok=True)

    shutil.copy2(baseDir / "main.py", bundleDir / "main.py")

    # Create launcher script for Linux
    if system == "Linux":
        launcherScript = bundleDir / "run.sh"
        with open(launcherScript, "w") as f:
            f.write("""#!/bin/bash
# The Anime Scripter - Linux Launcher Script

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
""")
        # Make the script executable
        os.chmod(launcherScript, 0o755)
        print("Created Linux launcher script: run.sh")

    print("Copying requirements files...")
    # Copy Windows requirements files
    if (baseDir / "extra-requirements-windows.txt").exists():
        shutil.copy2(
            baseDir / "extra-requirements-windows.txt",
            bundleDir / "extra-requirements-windows.txt",
        )
    if (baseDir / "extra-requirements-windows-lite.txt").exists():
        shutil.copy2(
            baseDir / "extra-requirements-windows-lite.txt",
            bundleDir / "extra-requirements-windows-lite.txt",
        )

    # Copy Linux requirements files
    if (baseDir / "extra-requirements-linux.txt").exists():
        shutil.copy2(
            baseDir / "extra-requirements-linux.txt",
            bundleDir / "extra-requirements-linux.txt",
        )
    if (baseDir / "extra-requirements-linux-lite.txt").exists():
        shutil.copy2(
            baseDir / "extra-requirements-linux-lite.txt",
            bundleDir / "extra-requirements-linux-lite.txt",
        )

    # Copy deprecated requirements
    if (baseDir / "deprecated-requirements.txt").exists():
        shutil.copy2(
            baseDir / "deprecated-requirements.txt",
            bundleDir / "deprecated-requirements.txt",
        )

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
        try:
            shutil.copy(baseDir / fileName, bundleDir)
        except Exception as e:
            print(f"Error while copying {fileName}: {e}")


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
                "C:/Users/nilas/AppData/Roaming/TheAnimeScripter/TAS-Portable"
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
