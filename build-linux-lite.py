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
    runSubprocess([str(venvBinPath / "python3.12"), "-m", "pip", "install", "-r", "requirements-linux-lite.txt"])
    
    unwanted_packages = [
        "nvidia-cuda-nvrtc-cu12",
        "nvidia-cuda-runtime-cu12",
        "nvidia-cuda-cupti-cu12",
        "nvidia-cudnn-cu12",
        "nvidia-cublas-cu12",
        "nvidia-cufft-cu12",
        "nvidia-curand-cu12",
        "nvidia-cusolver-cu12",
        "nvidia-cusparse-cu12",
        "nvidia-nccl-cu12",
        "nvidia-nvtx-cu12",
        "nvidia-nvjitlink-cu12",
        "triton"
    ]
    
    for package in unwanted_packages:
        runSubprocess([str(venvBinPath / "python3.12"), "-m", "pip", "uninstall", "-y", package])

def installPyinstaller():
    print("Installing PyInstaller...")
    runSubprocess([str(venvBinPath / "python3.12"), "-m", "pip", "install", "pyinstaller"])

def createExecutable():
    print("Creating executable with PyInstaller...")
    srcPath = baseDir / "src"
    mainPath = baseDir / "main.py"
    iconPath = srcPath / "assets" / "icon.ico"
    benchmarkPath = baseDir / "benchmark.py"

    exclude_packages = [
        "nvidia-cuda-nvrtc-cu12",
        "nvidia-cuda-runtime-cu12",
        "nvidia-cuda-cupti-cu12",
        "nvidia-cudnn-cu12",
        "nvidia-cublas-cu12",
        "nvidia-cufft-cu12",
        "nvidia-curand-cu12",
        "nvidia-cusolver-cu12",
        "nvidia-cusparse-cu12",
        "nvidia-nccl-cu12",
        "nvidia-nvtx-cu12",
        "nvidia-nvjitlink-cu12",
        "triton"
    ]

    exclude_args = []
    for package in exclude_packages:
        exclude_args.extend(["--exclude-module", package])

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
    ] + exclude_args

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
        "--collect-all",
        "grapheme",
        str(mainPath),
    ]

    pyinstallerPath = shutil.which("pyinstaller", path=str(venvBinPath))
    if not pyinstallerPath:
        print("PyInstaller not found in the virtual environment")
        return

    print("Creating the CLI executable...")
    runSubprocess([pyinstallerPath] + cliArgs)
    print("Finished creating the CLI executable")

    print("Creating the benchmark executable...")
    runSubprocess([pyinstallerPath] + commonArgs + [str(benchmarkPath)])
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

    benchmarkExePath = distPath / "benchmark" / "benchmark"
    mainExePath = distPath / "main"
    shutil.move(benchmarkExePath, mainExePath)

def moveExtras():
    mainDir = distPath / "main"
    filesToCopy = ["LICENSE", "README.md", "README.txt"]

    for fileName in filesToCopy:
        try:
            shutil.copy(baseDir / fileName, mainDir)
        except Exception as e:
            print(f"Error while copying {fileName}: {e}")

def cleanUp():
    benchmarkDir = distPath / "benchmark"
    try:
        shutil.rmtree(benchmarkDir)
    except Exception as e:
        print(f"Error while removing benchmark directory: {e}")

if __name__ == "__main__":
    createVenv()
    installRequirements()
    installPyinstaller()
    createExecutable()
    moveExtras()
    cleanUp()