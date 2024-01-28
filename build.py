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
        subprocess.run([".\\venv\\Scripts\\pip3", "install",
                       "--pre", "-r", "requirements.txt"], check=True)
    else:
        subprocess.run(["./venv/bin/pip3", "install", "--pre",
                       "-r", "requirements.txt"], check=True)


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
    jsx_path = os.path.join(base_dir, "TheAnimeScripter.jsx")
    license_path = os.path.join(base_dir, "LICENSE")
    main_path = os.path.join(base_dir, "main.py")
    icon_path = os.path.join(base_dir, "demos", "icon.ico")

    subprocess.run([
        "./venv/bin/pyinstaller" if platform.system() != "Windows" else ".\\venv\\Scripts\\pyinstaller",
        "--noconfirm",
        "--onedir",
        "--console",
        "--noupx",
        "--clean",
        "--add-data", f"{src_path};src/",
        "--collect-all", "cupy",
        "--collect-all", "cupyx",
        "--collect-all", "cupy_backends",
        "--collect-all", "fastrlock",
        "--icon", f"{icon_path}",
        main_path
    ], check=True)

    move_extras(jsx_path, license_path)


def move_extras(jsx_path, license_path):
    dist_dir = os.path.join(base_dir, "dist")
    main_dir = os.path.join(dist_dir, "main")
    target_path = os.path.join(main_dir, os.path.basename(jsx_path))
    try:
        shutil.copy(jsx_path, target_path)
        shutil.copy(license_path, main_dir)
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