@echo off
echo ------------------------------------
echo DOWNLOADING DEPENDENCIES
echo ------------------------------------
python -m pip install --upgrade pip
pip install -r requirements.txt
pause