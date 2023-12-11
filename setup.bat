@echo off

:: Honestly whoever came up with this is insane
:: Venv approach coming in the future, it's just a pain overall

NET SESSION >nul 2>&1
IF %ERRORLEVEL% EQU 0 (
    echo Running with admin perms. 
) ELSE (
    echo Run the script with admin perms.
    pause
    exit /b
)

cd /d %~dp0

set PYTHON_URL=https://www.python.org/ftp/python/3.11.0/python-3.11.0-amd64.exe

set /p consent="Do you agree with installing Python 3.11 and adding it to System Path? This is necessary for the functionality of the script. (Y/N): "
if /i "%consent%"=="Y" (
    powershell -Command "& { iwr '%PYTHON_URL%' -OutFile python-installer.exe }"
    python-installer.exe /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
) else (
    echo Python installation cancelled.
    echo Feel free to run the script again or refer to manual installation on the github page
    pause
    exit /b
)

echo Installing requirements...

:: Installing requirements
start cmd /c "pip install -r requirements.txt && echo Requirements installation succeeded! || echo Requirements installation failed!"

start cmd /c "python download_models.py"

echo handling ffmpeg...

mkdir src\ffmpeg

:: Handling ffmpeg
set FFMPEG_URL=https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip
powershell -Command "& { iwr '%FFMPEG_URL%' -OutFile src\ffmpeg\ffmpeg.zip }"

powershell -Command "& { Expand-Archive -Path 'src\ffmpeg\ffmpeg.zip' -DestinationPath 'src\ffmpeg\tmp'; }"

powershell -Command "& { Move-Item -Path 'src\ffmpeg\tmp\ffmpeg-*-win64-gpl\bin\ffmpeg.exe' -Destination 'src\ffmpeg'; }"

powershell -Command "& { Remove-Item -Path 'src\ffmpeg\tmp' -Recurse -Force; }"

powershell -Command "& { Remove-Item -Path 'src\ffmpeg\ffmpeg.zip'; }"

:: Removing the python installer
powershell -Command "& { Remove-Item -Path 'python-installer.exe'; }"