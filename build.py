import subprocess
import os
import shutil
from importlib.metadata import distribution

base_dir = os.path.dirname(os.path.abspath(__file__))
distPath = os.path.join(base_dir, "dist")


def create_venv():
    print("Creating the virtual environment...")
    subprocess.run(["python", "-m", "venv", "venv"], check=True)


def activate_venv():
    print("Activating the virtual environment...")
    subprocess.run(".\\venv\\Scripts\\activate", shell=True, check=True)


def install_requirements():
    print("Installing the requirements...")
    subprocess.run(
        [".\\venv\\Scripts\\pip3", "install", "-r", "requirements-windows.txt"],
        check=True,
    )


def install_pyinstaller():
    print("Installing PyInstaller...")
    subprocess.run(
        [".\\venv\\Scripts\\python", "-m", "pip", "install", "pyinstaller"], check=True
    )


def create_executable():
    print("Creating executable with PyInstaller...")
    src_path = os.path.join(base_dir, "src")
    main_path = os.path.join(base_dir, "main.py")
    # guiPath = os.path.join(base_dir, "gui.py")
    iconPath = os.path.join(base_dir, "src", "assets", "icon.ico")
    updaterPath = os.path.join(base_dir, "updater.py")
    benchmarkPath = os.path.join(base_dir, "benchmark.py")

    universal_ncnn_models_path = os.path.join(
        distribution("upscale_ncnn_py").locate_file("upscale_ncnn_py"),
        "models",
    )

    print("Creating the CLI executable...")
    subprocess.run(
        [
            ".\\venv\\Scripts\\pyinstaller",
            "--noconfirm",
            "--onedir",
            "--console",
            "--noupx",
            "--clean",
            "--add-data",
            f"{src_path};src/",
            "--add-data",
            f"{universal_ncnn_models_path};upscale_ncnn_py/models",
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
            "cupy",
            "--collect-all",
            "cupyx",
            "--collect-all",
            "cupy_backends",
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
    # print("Creating the GUI executable...")

    # subprocess.run(
    #    [
    #        ".\\venv\\Scripts\\pyinstaller",
    #        "--noconfirm",
    #        "--onedir",
    #        "--noconsole",
    #        "--noupx",
    #        "--clean",
    #        "--debug=all",
    #        "--icon",
    #        f"{iconPath}",
    #        guiPath,
    #    ],
    #    check=True,
    # )

    # print("Finished creating the GUI executable")
    print("Creating the benchmark executable...")

    subprocess.run(
        [
            ".\\venv\\Scripts\\pyinstaller",
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
    print("Creating the updater executable...")

    subprocess.run(
        [
            ".\\venv\\Scripts\\pyinstaller",
            "--noconfirm",
            "--onedir",
            "--console",
            "--noupx",
            "--clean",
            "--icon",
            f"{iconPath}",
            updaterPath,
        ],
        check=True,
    )
    print("Finished creating the updater executable")

    mainInternalPath = os.path.join(base_dir, "dist", "main", "_internal")
    # guiInternalPath = os.path.join(base_dir, "dist", "gui", "_internal")
    benchmarkInternalPath = os.path.join(base_dir, "dist", "benchmark", "_internal")
    updaterInternalPath = os.path.join(base_dir, "dist", "updater", "_internal")

    for directory in [benchmarkInternalPath, updaterInternalPath]:
        for filename in os.listdir(directory):
            sourceFilePath = os.path.join(directory, filename)
            mainFilePath = os.path.join(mainInternalPath, filename)

            if os.path.isfile(sourceFilePath):
                shutil.copy2(sourceFilePath, mainFilePath)

            elif os.path.isdir(sourceFilePath):
                shutil.copytree(sourceFilePath, mainFilePath, dirs_exist_ok=True)

    # guiExeFilePath = os.path.join(base_dir, "dist", "gui", "gui.exe")
    benchmarkExeFilePath = os.path.join(base_dir, "dist", "benchmark", "benchmark.exe")
    updaterExeFilePath = os.path.join(base_dir, "dist", "updater", "updater.exe")
    mainExeFilePath = os.path.join(base_dir, "dist", "main")

    # shutil.move(guiExeFilePath, mainExeFilePath)
    shutil.move(benchmarkExeFilePath, mainExeFilePath)
    shutil.move(updaterExeFilePath, mainExeFilePath)


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
    # Seems like pyinstaller duplicates some dll files due to cupy cuda
    # Removing them in order to fit the <2GB limit
    dllFiles = [
        "cublasLt64_12.dll",
        "cusparse64_12.dll",
        "cufft64_11.dll",
        "cusolver64_11.dll",
        "cublas64_12.dll",
        "curand64_10.dll",
        "nvrtc64_120_0.dll",
        "nvJitLink_120_0.dll",
    ]
    dllPath = os.path.join(base_dir, "dist", "main", "_internal")

    for file in dllFiles:
        try:
            os.remove(os.path.join(dllPath, file))
        except Exception as e:
            print("Error while removing dll file: ", e)

    # guiFolder = os.path.join(base_dir, "dist", "gui")
    benchmarkFolder = os.path.join(base_dir, "dist", "benchmark")
    updaterFolder = os.path.join(base_dir, "dist", "updater")

    try:
        # shutil.rmtree(guiFolder)
        shutil.rmtree(benchmarkFolder)
        shutil.rmtree(updaterFolder)

    except Exception as e:
        print("Error while removing gui folder: ", e)

    print("Done!, you can find the built executable in the dist folder")


if __name__ == "__main__":
    create_venv()
    activate_venv()
    install_requirements()
    install_pyinstaller()
    create_executable()
    move_extras()
    clean_up()
