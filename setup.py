import subprocess
import os
import platform

def create_venv():
    print("Creating virtual environment...")
    subprocess.run(["python", "-m", "venv", "venv"], check=True)

def activate_venv():
    print({"Activating virtual environment..."})
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

create_venv()
activate_venv()
install_requirements()