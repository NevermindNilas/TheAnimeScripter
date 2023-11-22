@echo off
echo ------------------------------------
echo    DOWNLOADING PYTHON INSTALLER
echo ------------------------------------
bitsadmin /transfer myDownloadJob /download /priority normal https://www.python.org/ftp/python/3.11.0/python-3.11.0-amd64.exe %cd%\python-3.11.0-amd64.exe >nul
echo Python installer downloaded. Waiting for 2 seconds before starting the installer...
echo !!!Be sure to add Python to system PATH!!!
timeout /t 2 /nobreak >nul
start /wait "" "%cd%\python-3.11.0-amd64.exe"

echo ------------------------------------
echo      DOWNLOADING DEPENDENCIES
echo ------------------------------------
python -m pip install --upgrade pip
pip install -r requirements.txt
pause