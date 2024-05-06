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

def install_polygraphy():
    print("Installing Polygraphy...")
    tensorrt_dir = os.path.join(base_dir, "TensorRT")
    polygraphy_dir = os.path.join(tensorrt_dir, "Tools", "Polygraphy")
    install_script = os.path.join(polygraphy_dir, "install.ps1")
    subprocess.run(["powershell", "-File", install_script], check=True)


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
    print("Creating the benchmark executable...")

    benchmarkPath = os.path.join(base_dir, "benchmark.py")
    subprocess.run(
        [
            ".\\venv\\Scripts\\pyinstaller",
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

    guiInternalPath = os.path.join(base_dir, "dist", "gui", "_internal")
    mainInternalPath = os.path.join(base_dir, "dist", "main", "_internal")
    benchmarkInternalPath = os.path.join(base_dir, "dist", "benchmark", "_internal")

    for directory in [guiInternalPath, benchmarkInternalPath]:
        for filename in os.listdir(directory):
            sourceFilePath = os.path.join(directory, filename)
            mainFilePath = os.path.join(mainInternalPath, filename)

            if os.path.isfile(sourceFilePath):
                shutil.copy2(sourceFilePath, mainFilePath)

            elif os.path.isdir(sourceFilePath):
                shutil.copytree(sourceFilePath, mainFilePath, dirs_exist_ok=True)

    guiExeFilePath = os.path.join(base_dir, "dist", "gui", "gui.exe")
    benchmarkExeFilePath = os.path.join(base_dir, "dist", "benchmark", "benchmark.exe")
    mainExeFilePath = os.path.join(base_dir, "dist", "main")

    shutil.move(guiExeFilePath, mainExeFilePath)
    shutil.move(benchmarkExeFilePath, mainExeFilePath)
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

    print("Done!, you can find the built executable in the dist folder")

if __name__ == "__main__":
    create_venv()
    activate_venv()
    install_polygraphy()
    install_requirements()
    install_pyinstaller()
    create_executable()
    move_extras()
    clean_up()
    
