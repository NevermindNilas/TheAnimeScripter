import argparse
import subprocess
import os
import platform

def create_venv():
    print("Creating virtual environment...")
    subprocess.run(["python", "-m", "venv", "venv"], check=True)

def activate_venv():
    print("Activating virtual environment...")
    if platform.system() == "Windows":
        os.system(".\\venv\\Scripts\\activate")
    else:
        os.system("source ./venv/bin/activate")

def install_requirements():
    print("Installing requirements...")
    if platform.system() == "Windows":
        subprocess.run([".\\venv\\Scripts\\python", "-m", "pip", "install", "-r", "requirements.txt"], check=True)
    else:
        subprocess.run(["./venv/bin/python", "-m", "pip", "install", "-r", "requirements.txt"], check=True)

def install_pyinstaller():
    print("Installing PyInstaller...")
    if platform.system() == "Windows":
        subprocess.run([".\\venv\\Scripts\\python", "-m", "pip", "install", "pyinstaller"], check=True)
    else:
        subprocess.run(["./venv/bin/python", "-m", "pip", "install", "pyinstaller"], check=True)

def create_executable():
    print("Creating executable with PyInstaller...")
    subprocess.run([
        "./venv/bin/pyinstaller" if platform.system() != "Windows" else ".\\venv\\Scripts\\pyinstaller",
        "--noconfirm",
        "--onedir",
        "--console",
        "--add-data", "H:/TheAnimeScripter/src;src/",
        "--add-data", "H:/TheAnimeScripter/src/get_ffmpeg.bat;.",
        "H:/TheAnimeScripter/main.py"
    ], check=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="Build the executable")
    args = parser.parse_args()

    create_venv()
    activate_venv()
    install_requirements()

    if args.build:
        #install_pyinstaller()
        #create_executable()
        pass

if __name__ == "__main__":
    main()