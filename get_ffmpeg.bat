:: Handling ffmpeg
@echo off

echo Downloading FFMPEG binaries

mkdir src\ffmpeg

set FFMPEG_URL=https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip
powershell -Command "& { iwr '%FFMPEG_URL%' -OutFile src\ffmpeg\ffmpeg.zip }"

powershell -Command "& { Expand-Archive -Path 'src\ffmpeg\ffmpeg.zip' -DestinationPath 'src\ffmpeg\tmp'; }"

powershell -Command "& { Move-Item -Path 'src\ffmpeg\tmp\ffmpeg-*-win64-gpl\bin\ffmpeg.exe' -Destination 'src\ffmpeg'; }"

powershell -Command "& { Remove-Item -Path 'src\ffmpeg\tmp' -Recurse -Force; }"

powershell -Command "& { Remove-Item -Path 'src\ffmpeg\ffmpeg.zip'; }"