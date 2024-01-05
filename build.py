import subprocess
import os
import platform
import shutil

base_dir = os.path.dirname(os.path.abspath(__file__))


def create_venv():
    print("Creating the virtual environment...")
    subprocess.run(["python", "-m", "venv", "venv"], check=True)


def activate_venv():
    print("Activating the virtual environment...")
    if platform.system() == "Windows":
        os.system(".\\venv\\Scripts\\activate")
    else:
        os.system("source ./venv/bin/activate")


def install_requirements():
    print("Installing the requirements...")
    if platform.system() == "Windows":
        subprocess.run([".\\venv\\Scripts\\python", "-m", "pip",
                       "install", "-r", "requirements.txt"], check=True)
    else:
        subprocess.run(["./venv/bin/python", "-m", "pip",
                       "install", "-r", "requirements.txt"], check=True)


def install_pyinstaller():
    print("Installing PyInstaller...")
    if platform.system() == "Windows":
        subprocess.run([".\\venv\\Scripts\\python", "-m", "pip",
                       "install", "pyinstaller"], check=True)
    else:
        subprocess.run(["./venv/bin/python", "-m", "pip",
                       "install", "pyinstaller"], check=True)


def create_executable():
    print("Creating executable with PyInstaller...")
    src_path = os.path.join(base_dir, "src")
    bat_path = os.path.join(base_dir, "get_ffmpeg.bat")
    jsx_path = os.path.join(base_dir, "TheAnimeScripter.jsx")
    main_path = os.path.join(base_dir, "main.py")
    subprocess.run([
        "./venv/bin/pyinstaller" if platform.system() != "Windows" else ".\\venv\\Scripts\\pyinstaller",
        "--noconfirm",
        "--onedir",
        "--console",
        "--noupx",
        "--add-data", f"{src_path};src/",
        "--add-data", f"{bat_path};.",
        "--add-data", f"{jsx_path};.",
        main_path
    ], check=True)
    move_jsx_file(jsx_path)


def move_jsx_file(jsx_path):
    dist_dir = os.path.join(base_dir, "dist")
    main_dir = os.path.join(dist_dir, "main")
    target_path = os.path.join(main_dir, os.path.basename(jsx_path))
    try:
        shutil.copy(jsx_path, target_path)
    except Exception as e:
        print("Error while copying jsx file: ", e)


def clean_up():
    clean_permission = input(
        "Do you want to clean up the residue files? (y/n) ")

    if clean_permission.lower() == "y":
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

    else:
        print("Skipping clean up...")

    print("Done!, you can find the built executable in the dist folder")


if __name__ == "__main__":
    create_venv()
    activate_venv()
    install_requirements()
    install_pyinstaller()
    create_executable()
    clean_up()
