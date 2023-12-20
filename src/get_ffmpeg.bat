:: Handling ffmpeg
@echo off

echo Downloading FFMPEG binaries

mkdir "%~dp0ffmpeg"

set FFMPEG_URL=https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip
powershell -Command "& { (New-Object System.Net.WebClient).DownloadFile('%FFMPEG_URL%', '%~dp0ffmpeg\ffmpeg.zip') }"

powershell -Command "& { Expand-Archive -Path '%~dp0ffmpeg\ffmpeg.zip' -DestinationPath '%~dp0ffmpeg\tmp'; }"

powershell -Command "& { Move-Item -Path '%~dp0ffmpeg\tmp\ffmpeg-*-win64-gpl\bin\ffmpeg.exe' -Destination '%~dp0ffmpeg'; }"

powershell -Command "& { Remove-Item -Path '%~dp0ffmpeg\tmp' -Recurse -Force; }"

powershell -Command "& { Remove-Item -Path '%~dp0ffmpeg\ffmpeg.zip'; }"