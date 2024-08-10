import subprocess
import os
import shutil

base_dir = os.path.dirname(os.path.abspath(__file__))
distPath = os.path.join(base_dir, "dist-full")


def create_venv():
    print("Creating the virtual environment...")
    subprocess.run(["python", "-m", "venv", "venv-full"], check=True)


def activate_venv():
    print("Activating the virtual environment...")
    subprocess.run(".\\venv-full\\Scripts\\activate", shell=True, check=True)


def install_requirements():
    print("Installing the requirements...")
    subprocess.run(
        [".\\venv-full\\Scripts\\pip3", "install", "-r", "requirements-windows.txt"],
        check=True,
    )


def install_pyinstaller():
    print("Installing PyInstaller...")
    subprocess.run(
        [".\\venv-full\\Scripts\\python", "-m", "pip", "install", "pyinstaller"], check=True
    )


def create_executable():
    print("Creating executable with PyInstaller...")
    src_path = os.path.join(base_dir, "src")
    mainPath = os.path.join(base_dir, "main.py")
    iconPath = os.path.join(base_dir, "src", "assets", "icon.ico")
    benchmarkPath = os.path.join(base_dir, "benchmark.py")

    print("Creating the CLI executable...")
    subprocess.run(
        [
            ".\\venv-full\\Scripts\\pyinstaller",
            "--noconfirm",
            "--onedir",
            "--console",
            "--noupx",
            "--clean",
            "--add-data",
            f"{src_path};src/",
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
            "--icon",
            f"{iconPath}",
            "--distpath",
            distPath,
            mainPath,
        ],
        check=True,
    )

    print("Finished creating the CLI executable")

    print("Creating the benchmark executable...")
    subprocess.run(
        [
            ".\\venv-full\\Scripts\\pyinstaller",
            "--noconfirm",
            "--onedir",
            "--console",
            "--noupx",
            "--clean",
            "--icon",
            f"{iconPath}",
            "--distpath",
            distPath,
            benchmarkPath,
        ],
        check=True,
    )
    print("Finished creating the benchmark executable")

    mainInternalPath = os.path.join(base_dir, "dist-full", "main", "_internal")
    benchmarkInternalPath = os.path.join(base_dir, "dist-full", "benchmark", "_internal")

    if os.path.exists(benchmarkInternalPath):
        for filename in os.listdir(benchmarkInternalPath):
            sourceFilePath = os.path.join(benchmarkInternalPath, filename)
            mainFilePath = os.path.join(mainInternalPath, filename)

            if os.path.isfile(sourceFilePath):
                shutil.copy2(sourceFilePath, mainFilePath)

            elif os.path.isdir(sourceFilePath):
                shutil.copytree(sourceFilePath, mainFilePath, dirs_exist_ok=True)

    benchmarkExeFilePath = os.path.join(base_dir, "dist-full", "benchmark", "benchmark.exe")
    mainExeFilePath = os.path.join(base_dir, "dist-full", "main")

    shutil.move(benchmarkExeFilePath, mainExeFilePath)

def move_extras():
    dist_dir = os.path.join(base_dir, "dist-full")
    main_dir = os.path.join(dist_dir, "main")
    license_path = os.path.join(base_dir, "LICENSE")
    readme_path = os.path.join(base_dir, "README.md")
    readme_txt_path = os.path.join(base_dir, "README.txt")

    try:
        shutil.copy(license_path, main_dir)
        shutil.copy(readme_path, main_dir)
        shutil.copy(readme_txt_path, main_dir)
    except Exception as e:
        print("Error while copying files: ", e)


def clean_up():
    benchmarkFolder = os.path.join(base_dir, "dist-full", "benchmark")

    try:
        shutil.rmtree(benchmarkFolder)
    except Exception as e:
        print("Error while removing benchmark folder: ", e)

    print("Done! You can find the built executable in the dist-full folder")


if __name__ == "__main__":
    create_venv()
    activate_venv()
    install_requirements()
    install_pyinstaller()
    create_executable()
    move_extras()
    clean_up()