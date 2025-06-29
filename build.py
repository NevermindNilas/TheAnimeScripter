import subprocess
import shutil
import os
from pathlib import Path
import urllib.request
import zipfile
import argparse

baseDir = Path(__file__).resolve().parent
distPath = baseDir / "dist-portable"
# requirementsPath = baseDir / "requirements.txt"
reqFiles = list(baseDir.glob("requirements*.txt"))
if reqFiles:
    requirementsPath = reqFiles[0]
else:
    raise FileNotFoundError("No requirements file found in the base directory.")

portablePythonDir = baseDir / "portable-python"
pythonVersion = "3.13.5"
vapourSynthVersion = "R72"


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


def downloadAndInstallVapourSynth():
    """Download and install VapourSynth into the portable Python"""
    print(f"Downloading VapourSynth {vapourSynthVersion}...")

    vapourSynthUrl = f"https://github.com/vapoursynth/vapoursynth/releases/download/{vapourSynthVersion}/VapourSynth64-Portable-{vapourSynthVersion}.zip"
    vapourSynthZip = portablePythonDir / "vapoursynth.zip"

    if not vapourSynthZip.exists():
        print("Downloading VapourSynth portable package...")
        urllib.request.urlretrieve(vapourSynthUrl, vapourSynthZip)

    print("Extracting VapourSynth into Python directory...")
    with zipfile.ZipFile(vapourSynthZip, "r") as zipRef:
        zipRef.extractall(portablePythonDir)

    print("Installing VapourSynth wheel...")
    pipExe = portablePythonDir / "Scripts" / "pip.exe"
    wheelDir = portablePythonDir / "wheel"

    python313Wheels = list(wheelDir.glob("*cp313*.whl"))
    if python313Wheels:
        wheelFile = python313Wheels[0]
        print(f"Installing wheel: {wheelFile.name}")
        runSubprocess([str(pipExe), "install", str(wheelFile)])
    else:
        print("Warning: No Python 3.13 wheel found in the wheel directory")

    if vapourSynthZip.exists():
        os.remove(vapourSynthZip)
        print("Removed VapourSynth zip file")

    print("VapourSynth installation complete!")


def installVapourSynthPlugins():
    """Install VapourSynth plugins using vsrepo"""
    print("Installing VapourSynth plugins...")

    pythonExe = portablePythonDir / "python.exe"
    vsrepoScript = portablePythonDir / "vsrepo.py"

    if not vsrepoScript.exists():
        print("Warning: vsrepo.py not found, skipping plugin installation")
        return

    env = os.environ.copy()
    env["PYTHONPATH"] = str(portablePythonDir / "Lib" / "site-packages")

    print("Testing VapourSynth installation...")
    try:
        runSubprocess(
            [
                str(pythonExe),
                "-c",
                "import vapoursynth; print('VapourSynth detected successfully')",
            ],
            cwd=portablePythonDir,
        )
    except Exception as e:
        print(f"Warning: VapourSynth module test failed: {e}")
        print("Attempting to continue with plugin installation...")

    print("Updating vsrepo...")
    try:
        result = subprocess.run(
            [str(pythonExe), str(vsrepoScript), "update"],
            cwd=portablePythonDir,
            env=env,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("vsrepo updated successfully!")
        else:
            print(f"vsrepo update failed: {result.stderr}")
    except Exception as e:
        print(f"Error updating vsrepo: {e}")

    print("Installing bestsource plugin...")
    try:
        result = subprocess.run(
            [str(pythonExe), str(vsrepoScript), "install", "bestsource"],
            cwd=portablePythonDir,
            env=env,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("bestsource plugin installed successfully!")
        else:
            print(f"bestsource installation failed: {result.stderr}")
            print("Trying alternative installation method...")
            try:
                subprocess.run(
                    [str(pythonExe), str(vsrepoScript), "install", "-f", "bestsource"],
                    cwd=portablePythonDir,
                    env=env,
                    check=True,
                )
                print("bestsource plugin installed successfully with force flag!")
            except Exception as e2:
                print(f"Alternative installation also failed: {e2}")
    except Exception as e:
        print(f"Error installing bestsource plugin: {e}")

    print("VapourSynth plugins installation complete!")


def installRequirements():
    print("Installing the requirements...")
    pipExe = portablePythonDir / "Scripts" / "pip.exe"
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

    print("Copying requirements files...")
    shutil.copy2(
        baseDir / "extra-requirements-windows.txt",
        bundleDir / "extra-requirements-windows.txt",
    )
    shutil.copy2(
        baseDir / "extra-requirements-windows-lite.txt",
        bundleDir / "extra-requirements-windows-lite.txt",
    )
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
    tempFiles = [
        "python.zip",
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
        finalOutputDir = Path(
            "C:/Users/nilas/AppData/Roaming/TheAnimeScripter/TAS-Portable"
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
    downloadAndInstallVapourSynth()
    installRequirements()
    installVapourSynthPlugins()
    bundleFiles(finalOutputDir)
    moveExtras(finalOutputDir)
    cleanupTempFiles(finalOutputDir)
    removePortablePython()
    print("Bundle process completed successfully!")
    print(f"Portable bundle is ready at {finalOutputDir}")
