import subprocess
import os
import shutil
from importlib.metadata import distribution

from main import scriptVersion

base_dir = os.path.dirname(os.path.abspath(__file__))
outputName = "TAS" + scriptVersion + ".7z"
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
        [".\\venv\\Scripts\\pip3", "install", "-r", "requirements.txt"],
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
    gui_path = os.path.join(base_dir, "gui.py")
    icon_path = os.path.join(base_dir, "src", "assets", "icon.ico")

    rife_ncnn_models_path = os.path.join(
        distribution("rife_ncnn_vulkan_python").locate_file("rife_ncnn_vulkan_python"),
        "models",
    )

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
            f"{rife_ncnn_models_path};rife_ncnn_vulkan_python/models",
            "--add-data",
            f"{universal_ncnn_models_path};upscale_ncnn_py/models",
            "--hidden-import",
            "rife_ncnn_vulkan_python.rife_ncnn_vulkan_wrapper",
            "--hidden-import",
            "upscale_ncnn_py.upscale_ncnn_py_wrapper",
            "--collect-all",
            "cupy",
            "--collect-all",
            "cupyx",
            "--collect-all",
            "cupy_backends",
            "--collect-all",
            "fastrlock",
            "--icon",
            f"{icon_path}",
            main_path,
        ],
        check=True,
    )

    print("Finished creating the CLI executable")
    print("Creating the GUI executable...")

    subprocess.run(
        [
            ".\\venv\\Scripts\\pyinstaller",
            "--noconfirm",
            "--onedir",
            "--noconsole",
            "--noupx",
            "--clean",
            "--icon",
            f"{icon_path}",
            gui_path,
        ],
        check=True,
    )

    print("Finished creating the GUI executable")

    guiInternalPath = os.path.join(base_dir, "dist", "gui", "_internal")
    mainInternalPath = os.path.join(base_dir, "dist", "main", "_internal")

    for filename in os.listdir(guiInternalPath):
        guiFilePath = os.path.join(guiInternalPath, filename)
        mainFilePath = os.path.join(mainInternalPath, filename)

        if os.path.isfile(guiFilePath):
            shutil.copy2(guiFilePath, mainFilePath)

        elif os.path.isdir(guiFilePath):
            shutil.copytree(guiFilePath, mainFilePath, dirs_exist_ok=True)

    guiExeFilePath = os.path.join(base_dir, "dist", "gui", "gui.exe")
    mainExeFilePath = os.path.join(base_dir, "dist", "main")

    shutil.move(guiExeFilePath, mainExeFilePath)
    mainInternalPath = os.path.join(base_dir, "dist", "main", "_internal")


def move_extras():
    dist_dir = os.path.join(base_dir, "dist")
    main_dir = os.path.join(dist_dir, "main")
    jsx_path = os.path.join(base_dir, "TheAnimeScripter.jsx")
    license_path = os.path.join(base_dir, "LICENSE")
    readme_path = os.path.join(base_dir, "README.md")
    readme_txt_path = os.path.join(base_dir, "README.txt")
    target_path = os.path.join(main_dir, os.path.basename(jsx_path))

    try:
        shutil.copy(jsx_path, target_path)
        shutil.copy(license_path, main_dir)
        shutil.copy(readme_path, main_dir)
        shutil.copy(readme_txt_path, main_dir)
    except Exception as e:
        print("Error while copying jsx file: ", e)


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

    guiFolder = os.path.join(base_dir, "dist", "gui")

    try:
        shutil.rmtree(guiFolder)
    except Exception as e:
        print("Error while removing gui folder: ", e)

    print("\n")
    answer = input("Do you want to clean up the residual files? (y/n): ")

    if answer.lower() == "y":
        print("Cleaning up...")
        try:
            spec_file = os.path.join(base_dir, "main.spec")
            if os.path.exists(spec_file):
                os.remove(spec_file)
        except Exception as e:
            print("Error while removing spec file: ", e)

        try:
            venv_file = os.path.join(base_dir, "venv")
            if os.path.exists(venv_file):
                shutil.rmtree(venv_file)
        except Exception as e:
            print("Error while removing the venv: ", e)

        try:
            build_dir = os.path.join(base_dir, "build")
            if os.path.exists(build_dir):
                shutil.rmtree(build_dir)
        except Exception as e:
            print("Error while removing the build directory: ", e)

        try:
            pycache_dir = os.path.join(base_dir, "__pycache__")
            if os.path.exists(pycache_dir):
                shutil.rmtree(pycache_dir)
        except Exception as e:
            print("Error while removing the pycache directory: ", e)

    else:
        print("Skipping Cleanup...")

    print("Done!, you can find the built executable in the dist folder")


def compress_dist():
    print("Compressing the dist folder...")
    answer = input(
        "Do you want to compress the file with 7z? NOTE: It requires 7z to be installed and on path. (y/n): "
    )

    if answer.lower() == "y":
        print("Compressing the dist folder, this can take a while...")
        tempDistPath = os.path.join(distPath, "main")
        subprocess.run(
            ["7z", "a", "-mx9", os.path.join(distPath, outputName), tempDistPath],
            shell=True,
            check=True,
        )
    else:
        print("Skipping Compression...")

    print("Done!, you can find the compressed file in the root folder")


if __name__ == "__main__":
    create_venv()
    activate_venv()
    install_requirements()
    install_pyinstaller()
    create_executable()
    move_extras()
    clean_up()
    compress_dist()
