import subprocess
import os
import platform

def create_venv():
    subprocess.run(["python", "-m", "venv", "venv"], check=True)

def activate_venv():
    if platform.system() == "Windows":
        os.system(".\\venv\\Scripts\\activate")
    else:
        os.system("source ./venv/bin/activate")

def install_requirements():
    if platform.system() == "Windows":
        subprocess.run([".\\venv\\Scripts\\python", "-m", "pip", "install", "-r", "requirements.txt"], check=True)
    else:
        subprocess.run(["./venv/bin/python", "-m", "pip", "install", "-r", "requirements.txt"], check=True)

def download_models():
    print("Do you want to download the models for offline use? (y/n)")
    answer = input()
    if answer == "y":
        if platform.system() == "Windows":
            subprocess.run([".\\venv\\Scripts\\python", ".\\src\\download_models.py"], check=True)
        else:
            subprocess.run(["./venv/bin/python", "./src/download_models.py"], check=True)
    else:
        print("The model(s) will be downloaded on runtime")

create_venv()
activate_venv()
install_requirements()
download_models()