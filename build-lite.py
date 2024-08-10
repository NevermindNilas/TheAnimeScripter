import subprocess
import os
import shutil
import sys

base_dir = os.path.dirname(os.path.abspath(__file__))
distPath = os.path.join(base_dir, "dist-lite" if "--output" in sys.argv else "dist")


def create_venv():
    print("Creating the virtual environment...")
    subprocess.run(["python", "-m", "venv", "venv-lite"], check=True)


def activate_venv():
    print("Activating the virtual environment...")
    subprocess.run(".\\venv-lite\\Scripts\\activate", shell=True, check=True)


def install_requirements():
    print("Installing the requirements...")
    subprocess.run(
        [".\\venv-lite\\Scripts\\pip3", "install", "-r", "requirements-windows-lite.txt"],
        check=True,
    )


def install_pyinstaller():
    print("Installing PyInstaller...")
    subprocess.run(
        [".\\venv-lite\\Scripts\\python", "-m", "pip", "install", "pyinstaller"], check=True
    )


def create_executable():
    print("Creating executable with PyInstaller...")
    src_path = os.path.join(base_dir, "src")
    main_path = os.path.join(base_dir, "main.py")
    iconPath = os.path.join(base_dir, "src", "assets", "icon.ico")
    benchmarkPath = os.path.join(base_dir, "benchmark.py")

    print("Creating the CLI executable...")
    subprocess.run(
        [
            ".\\venv-lite\\Scripts\\pyinstaller",
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
            main_path,
        ],
        check=True,
    )

    print("Finished creating the CLI executable")

    print("Creating the benchmark executable...")
    subprocess.run(
        [
            ".\\venv-lite\\Scripts\\pyinstaller",
            "--noconfirm",
            "--onedir",
            "--console",
            "--noupx",
            "--clean",
            "--icon",
            f"{iconPath}",
            benchmarkPath,
        ],
        check=True,
    )
    print("Finished creating the benchmark executable")

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

    benchmarkExeFilePath = os.path.join(base_dir, "dist", "benchmark", "benchmark.exe")
    mainExeFilePath = os.path.join(base_dir, "dist", "main")

    shutil.move(benchmarkExeFilePath, mainExeFilePath)


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
        print("Error while copying files: ", e)


def clean_up():
    benchmarkFolder = os.path.join(base_dir, distPath, "benchmark")

    try:
        shutil.rmtree(benchmarkFolder)
    except Exception as e:
        print("Error while removing folders: ", e)

    print("Done! You can find the built executable in the dist folder")


if __name__ == "__main__":
    create_venv()
    activate_venv()
    install_requirements()
    install_pyinstaller()
    create_executable()
    move_extras()
    clean_up()