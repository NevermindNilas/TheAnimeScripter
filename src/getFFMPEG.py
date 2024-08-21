import shutil
import requests
import zipfile
import tarfile
import logging
import os
import platform
import glob

from alive_progress import alive_bar

# Determine paths and URLs based on the operating system
if os.name == "nt":
    mainPath = os.path.join(os.getenv("APPDATA"), "TheAnimeScripter")
    FFMPEGURL = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
else:
    mainPath = os.path.join(
        os.getenv("XDG_CONFIG_HOME", os.path.expanduser("~/config")), "TheAnimeScripter"
    )
    FFMPEGURL = (
        "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
    )

os.makedirs(mainPath, exist_ok=True)


def getFFMPEG():
    ffmpegPath = shutil.which("ffmpeg")
    if ffmpegPath is None:
        ffmpegPath = os.path.join(
            mainPath,
            "ffmpeg",
            "ffmpeg.exe" if platform.system() == "Windows" else "ffmpeg",
        )
        logging.info(f"FFMPEG Path: {ffmpegPath}")
        if not os.path.exists(ffmpegPath):
            ffmpegPath = downloadAndExtractFFMPEG(ffmpegPath)
    else:
        logging.info(f"FFMPEG was found in System Path: {ffmpegPath}")
    return str(ffmpegPath)


def downloadAndExtractFFMPEG(ffmpegPath):
    logging.info("Getting FFMPEG")
    extractFunc = (
        extractFFMPEGZip if platform.system() == "Windows" else extractFFMPEGTar
    )
    ffmpegDir = os.path.dirname(ffmpegPath)
    ffmpegArchivePath = os.path.join(
        ffmpegDir, "ffmpeg.zip" if platform.system() == "Windows" else "ffmpeg.tar.xz"
    )

    os.makedirs(ffmpegDir, exist_ok=True)

    response = requests.get(FFMPEGURL, stream=True)
    totalSizeInBytes = int(response.headers.get("content-length", 0))
    totalSizeInMB = totalSizeInBytes // (1024 * 1024)

    with alive_bar(
        total=totalSizeInMB + 1, title="Downloading FFmpeg", bar="smooth", unit="MB"
    ) as bar, open(ffmpegArchivePath, "wb") as file:
        for data in response.iter_content(chunk_size=1024 * 1024):
            file.write(data)
            bar(len(data) // (1024 * 1024))

    extractFunc(ffmpegArchivePath, ffmpegDir)
    return str(ffmpegPath)


def extractFFMPEGZip(ffmpegZipPath, ffmpegDir):
    with zipfile.ZipFile(ffmpegZipPath, "r") as zipRef:
        zipRef.extractall(ffmpegDir)
    os.rename(
        os.path.join(ffmpegDir, "ffmpeg-master-latest-win64-gpl", "bin", "ffmpeg.exe"),
        os.path.join(ffmpegDir, "ffmpeg.exe"),
    )
    os.remove(ffmpegZipPath)
    shutil.rmtree(os.path.join(ffmpegDir, "ffmpeg-master-latest-win64-gpl"))


def extractFFMPEGTar(ffmpegTarPath, ffmpegDir):
    with tarfile.open(ffmpegTarPath, "r:xz") as tarRef:
        tarRef.extractall(ffmpegDir)
    for directory in glob.glob(os.path.join(ffmpegDir, "ffmpeg-*-static")):
        os.rename(os.path.join(directory, "ffmpeg"), os.path.join(ffmpegDir, "ffmpeg"))
        shutil.rmtree(directory)
    os.remove(ffmpegTarPath)