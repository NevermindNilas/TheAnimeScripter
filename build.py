import subprocess
import shutil
import os
from pathlib import Path
import urllib.request
import zipfile

baseDir = Path(__file__).resolve().parent
distPath = baseDir / "dist-portable"
requirementsPath = baseDir / "requirements-windows.txt"
portablePythonDir = baseDir / "portable-python"
pythonVersion = "3.13.3"  # Hardcoded version


def runSubprocess(command, shell=False, cwd=None):
    try:
        subprocess.run(command, shell=shell, check=True, cwd=cwd)
    except subprocess.CalledProcessError as e:
        print(f"Error while running command {command}: {e}")
        raise


def downloadPortablePython():
    """Download the embeddable Python package"""
    print(f"Downloading portable Python {pythonVersion}...")

    os.makedirs(portablePythonDir, exist_ok=True)

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
        [str(portablePythonDir / "python.exe"), "get-pip.py"], cwd=portablePythonDir
    )

    print("Portable Python installation complete!")
    return portablePythonDir / "python.exe"


def installRequirements():
    print("Installing the requirements...")
    pipExe = portablePythonDir / "Scripts" / "pip.exe"
    runSubprocess([str(pipExe), "install", "-r", str(requirementsPath)])
    print("Requirements installation complete!")


def bundleFiles():
    print("Creating portable bundle...")

    bundleDir = distPath / "TAS-Portable"
    os.makedirs(bundleDir, exist_ok=True)

    print("Copying Python installation...")
    pythonDir = bundleDir
    if pythonDir.exists():
        shutil.rmtree(pythonDir)
    shutil.copytree(portablePythonDir, pythonDir)

    print("Copying source code...")
    srcDir = baseDir / "src"
    try:
        shutil.copytree(srcDir, bundleDir / "src")
    except FileExistsError:
        print(f"Directory {bundleDir / 'src'} already exists. Skipping copy.")
    except Exception as e:
        print(f"Error while copying {srcDir}: {e}")

    try:
        # Copy main.py
        print("Copying main.py...")
        shutil.copy(baseDir / "main.py", bundleDir / "main.py")
    except FileExistsError:
        print(f"File {bundleDir / 'main.py'} already exists. Skipping copy.")
    except Exception as e:
        print(f"Error while copying {baseDir / 'main.py'}: {e}")

    print("Copying requirements files...")
    shutil.copy(
        baseDir / "extra-requirements-windows.txt",
        bundleDir / "extra-requirements-windows.txt",
    )
    shutil.copy(
        baseDir / "extra-requirements-windows-lite.txt",
        bundleDir / "extra-requirements-windows-lite.txt",
    )

    print(f"Portable bundle created at {bundleDir}")


def moveExtras():
    bundleDir = distPath / "TAS-Portable"
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


def removePortablePython():
    """Remove the portable Python directory"""
    if portablePythonDir.exists():
        shutil.rmtree(portablePythonDir)
        print(f"Removed {portablePythonDir}")
    else:
        print(f"{portablePythonDir} does not exist, skipping removal.")


if __name__ == "__main__":
    pythonExe = downloadPortablePython()
    installRequirements()
    bundleFiles()
    moveExtras()
    removePortablePython()
    print("Bundle process completed successfully!")
