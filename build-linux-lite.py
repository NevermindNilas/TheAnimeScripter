import subprocess
import shutil
from pathlib import Path

baseDir = Path(__file__).resolve().parent
distPath = baseDir / "dist-lite"
venvPath = baseDir / "venv-lite"
venvBinPath = venvPath / "bin"


def runSubprocess(command, shell=False):
    try:
        subprocess.run(command, shell=shell, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error while running command {command}: {e}")
        raise


def createVenv():
    print("Creating the virtual environment...")
    try:
        runSubprocess(["python3.12", "-m", "venv", str(venvPath), "--without-pip"])
        print("Installing pip in the virtual environment...")
        runSubprocess([str(venvBinPath / "python3.12"), "-m", "ensurepip", "--upgrade"])
    except subprocess.CalledProcessError:
        print("Failed to create venv with python3.12. Attempting with virtualenv...")
        runSubprocess(["pip", "install", "virtualenv"])
        runSubprocess(["virtualenv", "-p", "python3.12", str(venvPath)])


def installRequirements():
    print("Installing the requirements...")
    runSubprocess(
        [
            str(venvBinPath / "python3.12"),
            "-m",
            "pip",
            "install",
            "-r",
            "requirements-linux-lite.txt",
        ]
    )


def installPyinstaller():
    print("Installing PyInstaller...")
    runSubprocess(
        [str(venvBinPath / "python3.12"), "-m", "pip", "install", "pyinstaller"]
    )


def createExecutable():
    print("Creating executable with PyInstaller...")
    srcPath = baseDir / "src"
    mainPath = baseDir / "main.py"
    iconPath = srcPath / "assets" / "icon.ico"

    commonArgs = [
        "--noconfirm",
        "--onedir",
        "--console",
        "--noupx",
        "--clean",
        "--icon",
        str(iconPath),
        "--distpath",
        str(distPath),
    ]

    cliArgs = commonArgs + [
        "--add-data",
        f"{srcPath}:src/",
        "--hidden-import",
        "rife_ncnn_vulkan_python.rife_ncnn_vulkan_wrapper",
        "--hidden-import",
        "upscale_ncnn_py.upscale_ncnn_py_wrapper",
        "--collect-all",
        "fastrlock",
        "--collect-all",
        "inquirer",
        "--collect-all",
        "readchar",
        str(mainPath),
    ]

    pyinstallerPath = shutil.which("pyinstaller", path=str(venvBinPath))
    if not pyinstallerPath:
        print("PyInstaller not found in the virtual environment")
        return

    print("Creating the CLI executable...")
    runSubprocess([pyinstallerPath] + cliArgs)
    print("Finished creating the CLI executable")


def moveExtras():
    mainDir = distPath / "main"
    filesToCopy = ["LICENSE", "README.md", "README.txt"]

    for fileName in filesToCopy:
        try:
            shutil.copy(baseDir / fileName, mainDir)
        except Exception as e:
            print(f"Error while copying {fileName}: {e}")


if __name__ == "__main__":
    createVenv()
    installRequirements()
    installPyinstaller()
    createExecutable()
    moveExtras()
