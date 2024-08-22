import subprocess
import os
import shutil

base_dir = os.path.dirname(os.path.abspath(__file__))
distPath = os.path.join(base_dir, "dist")


def create_venv():
    print("Creating the virtual environment...")
    try:
        subprocess.run(
            ["python3.12", "-m", "venv", "venv", "--without-pip"], check=True
        )
        print("Installing pip in the virtual environment...")
        subprocess.run(
            ["./venv/bin/python3.12", "-m", "ensurepip", "--upgrade"], check=True
        )
    except subprocess.CalledProcessError:
        print("Failed to create venv with python3.12. Attempting with virtualenv...")
        subprocess.run(["pip", "install", "virtualenv"], check=True)
        subprocess.run(["virtualenv", "-p", "python3.12", "venv"], check=True)


def install_requirements():
    print("Installing the requirements...")
    subprocess.run(
        [
            "./venv/bin/python3.12",
            "-m",
            "pip",
            "install",
            "-r",
            "requirements-linux.txt",
        ],
        check=True,
    )


def install_pyinstaller():
    print("Installing PyInstaller...")
    subprocess.run(
        ["./venv/bin/python3.12", "-m", "pip", "install", "pyinstaller"], check=True
    )


def create_executable():
    print("Creating executable with PyInstaller...")
    src_path = os.path.join(base_dir, "src")
    main_path = os.path.join(base_dir, "main.py")
    icon_path = os.path.join(base_dir, "src", "assets", "icon.ico")

    print("Creating the CLI executable...")

    pyinstallerPath = shutil.which("pyinstaller", path="./venv/bin")
    if not pyinstallerPath:
        print("PyInstaller not found in the virtual environment")
        return

    try:
        subprocess.run(
            [
                pyinstallerPath,
                "--noconfirm",
                "--onedir",
                "--console",
                "--noupx",
                "--clean",
                "--add-data",
                f"{src_path}:src/",
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
                "--icon",
                f"{icon_path}",
                main_path,
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error during PyInstaller execution: {e}")
        return

    print("Finished creating the CLI executable")

    print("Creating the benchmark executable...")

    benchmarkPath = os.path.join(base_dir, "benchmark.py")
    try:
        subprocess.run(
            [
                pyinstallerPath,
                "--noconfirm",
                "--onedir",
                "--console",
                "--noupx",
                "--clean",
                "--icon",
                f"{icon_path}",
                benchmarkPath,
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error during PyInstaller execution: {e}")
        return

    mainInternalPath = os.path.join(base_dir, "dist", "main", "_internal")
    benchmarkInternalPath = os.path.join(base_dir, "dist", "benchmark", "_internal")

    for directory in [benchmarkInternalPath]:
        for filename in os.listdir(directory):
            sourceFilePath = os.path.join(directory, filename)
            mainFilePath = os.path.join(mainInternalPath, filename)

            if os.path.isfile(sourceFilePath):
                shutil.copy2(sourceFilePath, mainFilePath)

            elif os.path.isdir(sourceFilePath):
                shutil.copytree(sourceFilePath, mainFilePath, dirs_exist_ok=True)

    benchmarkExeFilePath = os.path.join(base_dir, "dist", "benchmark", "benchmark")
    mainExeFilePath = os.path.join(base_dir, "dist", "main")

    shutil.move(benchmarkExeFilePath, mainExeFilePath)
    mainInternalPath = os.path.join(base_dir, "dist", "main", "_internal")


def move_extras():
    dist_dir = os.path.join(base_dir, "dist")
    main_dir = os.path.join(dist_dir, "main")
    license_path = os.path.join(base_dir, "LICENSE")
    readme_path = os.path.join(base_dir, "README.md")
    readme_txt_path = os.path.join(base_dir, "README.txt")

    try:
        shutil.copy(license_path, main_dir)
        shutil.copy(readme_path, main_dir)
        shutil.copy(readme_txt_path, main_dir)
    except Exception as e:
        print("Error while copying jsx file: ", e)


if __name__ == "__main__":
    create_venv()
    install_requirements()
    install_pyinstaller()
    create_executable()
    move_extras()
