@echo off
set "INSTALLER_PATH=C:\Program Files (x86)\Microsoft Visual Studio\Installer"
set "PATH=%INSTALLER_PATH%;%PATH%"
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 (
    echo vcvars64.bat failed
    exit /b 1
)
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
set "CUDA_PATH=%CUDA_HOME%"
set "PATH=%CUDA_HOME%\bin;%PATH%"
where cl
where nvcc
python "%~dp0build_softsplat.py"
