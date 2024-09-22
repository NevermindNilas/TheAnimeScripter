import subprocess
import shutil
from pathlib import Path

baseDir = Path(__file__).resolve().parent
distPath = baseDir / "dist-full"
venvPath = baseDir / "venv-full"
venvScripts = venvPath / "Scripts"


def runSubprocess(command, shell=False):
    try:
        subprocess.run(command, shell=shell, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error while running command {command}: {e}")
        raise


def createVenv():
    print("Creating the virtual environment...")
    runSubprocess(["python", "-m", "venv", str(venvPath)])


def activateVenv():
    print("Activating the virtual environment...")
    runSubprocess(str(venvScripts / "activate"), shell=True)


def installRequirements():
    print("Installing the requirements...")
    runSubprocess(
        [str(venvScripts / "pip3"), "install", "-r", "requirements-windows.txt"]
    )


def installPyinstaller():
    print("Installing PyInstaller...")
    runSubprocess([str(venvScripts / "python"), "-m", "pip", "install", "pyinstaller"])


def createExecutable():
    print("Creating executable with PyInstaller...")
    srcPath = baseDir / "src"
    mainPath = baseDir / "main.py"
    iconPath = srcPath / "assets" / "icon.ico"
    benchmarkPath = baseDir / "benchmark.py"

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
        f"{srcPath};src/",
        "--hidden-import",
        "rife_ncnn_vulkan_python.rife_ncnn_vulkan_wrapper",
        "--hidden-import",
        "upscale_ncnn_py.upscale_ncnn_py_wrapper",
        "--collect-all",
        "tensorrt",
        "--collect-all",
        "tensorrt-cu12-bindings",
        "--collect-all",
        "tensorrt_libs",
        "--collect-all",
        "fastrlock",
        "--collect-all",
        "inquirer",
        "--collect-all",
        "readchar",
        "--collect-all",
        "grapheme",
        str(mainPath),
    ]

    benchmarkArgs = commonArgs + [str(benchmarkPath)]

    print("Creating the CLI executable...")
    runSubprocess([str(venvScripts / "pyinstaller")] + cliArgs)

    print("Finished creating the CLI executable")

    print("Creating the benchmark executable...")
    runSubprocess([str(venvScripts / "pyinstaller")] + benchmarkArgs)
    print("Finished creating the benchmark executable")

    mainInternalPath = distPath / "main" / "_internal"
    benchmarkInternalPath = distPath / "benchmark" / "_internal"

    if benchmarkInternalPath.exists():
        for item in benchmarkInternalPath.iterdir():
            targetPath = mainInternalPath / item.name
            if item.is_file():
                shutil.copy2(item, targetPath)
            elif item.is_dir():
                shutil.copytree(item, targetPath, dirs_exist_ok=True)

    benchmarkExePath = distPath / "benchmark" / "benchmark.exe"
    mainExePath = distPath / "main"

    shutil.move(benchmarkExePath, mainExePath)


def compileAll():
    print("Compiling all the files...")
    mainDir = distPath / "main"
    runSubprocess([str(venvScripts / "python"), "-m", "compileall", str(mainDir)])


def moveExtras():
    mainDir = distPath / "main"
    filesToCopy = ["LICENSE", "README.md", "README.txt"]

    for fileName in filesToCopy:
        try:
            shutil.copy(baseDir / fileName, mainDir)
        except Exception as e:
            print(f"Error while copying {fileName}: {e}")


def cleanUp():
    benchmarkFolder = distPath / "benchmark"

    try:
        shutil.rmtree(benchmarkFolder)
    except Exception as e:
        print(f"Error while removing benchmark folder: {e}")

    print("Done! You can find the built executable in the dist-full folder")


if __name__ == "__main__":
    createVenv()
    activateVenv()
    installRequirements()
    installPyinstaller()
    createExecutable()
    compileAll()
    moveExtras()
    cleanUp()
