@echo off
if "%~1"=="post-install" goto :post_install

echo ------------------------------------
echo    DOWNLOADING PYTHON INSTALLER
echo ------------------------------------
curl https://www.python.org/ftp/python/3.11.0/python-3.11.0-amd64.exe --output %cd%\python-3.11.0-amd64.exe

echo ------------------------------------
echo         INSTALLING PYTHON
echo ------------------------------------
set /p consent="Do you agree with installing Python 3.11 and adding it to System Path? This is necessary for the functionality of the script. (Y/N): "
if /i "%consent%"=="Y" (
    start /wait "" "%cd%\python-3.11.0-amd64.exe" /quiet InstallAllUsers=1 PrependPath=1
) else (
    echo Python installation cancelled.
    echo Feel free to run the script again or refer to manual installation on the github page
    pause
    exit /b
)

echo ------------------------------------
echo         RESTARTING SCRIPT
echo ------------------------------------
start cmd /k "%~dpnx0" post-install
exit /b

:post_install
echo ------------------------------------
echo      DOWNLOADING DEPENDENCIES
echo ------------------------------------
call "%ProgramFiles%\Python311\Scripts\pip" install --no-warn-script-location -r requirements.txt

echo ------------------------------------
echo         DOWNLOADING MODELS
echo ------------------------------------
call "%ProgramFiles%\Python311\python" download_models.py

echo ------------------------------------
echo CHECKING STATIC_FFMPEG INSTALLATION
echo ------------------------------------
static_ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo static_ffmpeg is not installed properly. Please run: static_ffmpeg in the terminal.
    exit /b
) else (
    echo static_ffmpeg is installed properly.
)

pause
exit /b