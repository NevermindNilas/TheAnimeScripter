@echo off
echo ------------------------------------
echo         DOWNLOADING MODELS
echo ------------------------------------
python .\download_models.py

echo ------------------------------------
echo      DOWNLOADING DEPENDENCIES
echo ------------------------------------
pip install -r requirements.txt
pause