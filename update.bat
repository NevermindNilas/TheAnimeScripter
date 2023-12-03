@echo off
set /p UserInput=Do you want to download the models? After Effects users will have to. (Y/N):
if /I "%UserInput%" EQU "Y" (
    echo ------------------------------------
    echo         DOWNLOADING MODELS
    echo ------------------------------------
    python .\src\download_models.py
) else (
    echo The models will not be downloaded, models will be automatically downloaded on runtime or can be downloaded manually,
    echo using 'python .\download_models.py' in the command line.
    echo After Effects users will have to download them if they want to use the script.
)

echo ------------------------------------
echo      DOWNLOADING DEPENDENCIES
echo ------------------------------------
python -m pip install --upgrade pip
pip install -r requirements.txt
pause