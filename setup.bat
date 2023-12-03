@echo off
echo ------------------------------------
echo    DOWNLOADING PYTHON INSTALLER
echo ------------------------------------
curl https://www.python.org/ftp/python/3.11.0/python-3.11.0-amd64.exe --output %cd%\python-3.11.0-amd64.exe

echo ------------------------------------
echo      AUTO INSTALLING PYTHON
echo ------------------------------------
start /wait "" "%cd%\python-3.11.0-amd64.exe" 

set /p UserInput=Do you agree with adding Python to system PATH? (Y/N):
if /I "%UserInput%" EQU "Y" (
    echo ------------------------------------
    echo      ADDING PYTHON TO PATH
    echo ------------------------------------
    setx path "%path%;%localappdata%\Programs\Python\Python311"
) else (
    echo Installation has been cancelled this can be ran again otherwise,
    echo Please refer to manual installation: https://github.com/NevermindNilas/TheAnimeScripter/tree/main#manual-installation
    pause
    exit /b
)

set /p UserInput=Do you want to download the models? After Effects users will have to. (Y/N):
if /I "%UserInput%" EQU "Y" (
    echo ------------------------------------
    echo         DOWNLOADING MODELS
    echo ------------------------------------
    python .\download_models.py
) else (
    echo The models will not be downloaded, you can manually run python download_models.py
    echo After Effects users will have to download them if they want to use the script.
)

echo ------------------------------------
echo      DOWNLOADING DEPENDENCIES
echo ------------------------------------
python -m pip install --upgrade pip
pip install -r requirements.txt
pause