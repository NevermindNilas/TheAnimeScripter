:: Handling ffmpeg
@echo off

echo Downloading FFMPEG binaries

mkdir "%~dp0src\ffmpeg"

set FFMPEG_URL=https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip
powershell -Command "& { (New-Object System.Net.WebClient).DownloadFile('%FFMPEG_URL%', '%~dp0src\ffmpeg\ffmpeg.zip') }"

powershell -Command "& { Expand-Archive -Path '%~dp0src\ffmpeg\ffmpeg.zip' -DestinationPath '%~dp0src\ffmpeg\tmp'; }"

powershell -Command "& { Move-Item -Path '%~dp0src\ffmpeg\tmp\ffmpeg-*-win64-gpl\bin\ffmpeg.exe' -Destination '%~dp0src\ffmpeg'; }"

powershell -Command "& { Remove-Item -Path '%~dp0src\ffmpeg\tmp' -Recurse -Force; }"

powershell -Command "& { Remove-Item -Path '%~dp0src\ffmpeg\ffmpeg.zip'; }"