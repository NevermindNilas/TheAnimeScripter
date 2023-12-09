@echo off
echo ------------------------------------
echo      DOWNLOADING DEPENDENCIES
echo ------------------------------------
pip install -r requirements.txt

echo ------------------------------------
echo         DOWNLOADING MODELS
echo ------------------------------------
python .\download_models.py
pause