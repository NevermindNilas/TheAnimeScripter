import shutil
import requests
import glob
import zipfile
import tarfile
import logging
import os
import platform

from alive_progress import alive_bar

if os.name == "nt":
    appdata = os.getenv("APPDATA")
    mainPath = os.path.join(appdata, "TheAnimeScripter")

    if not os.path.exists(mainPath):
        os.makedirs(mainPath)
else:
    dirPath = os.path.dirname(os.path.realpath(__file__))

def getFFMPEG():
    ffmpegPath = shutil.which("ffmpeg")

    if ffmpegPath is None:
        if platform.system() == "Windows":
            ffmpegPath = os.path.join(mainPath, "ffmpeg", "ffmpeg.exe")
        else:
            ffmpegPath = os.path.join(mainPath, "ffmpeg", "ffmpeg")

        logging.info(f"FFMPEG Path: {ffmpegPath}")

        if not os.path.exists(ffmpegPath):
            ffmpegPath = downloadAndExtractFFMPEG(ffmpegPath)
    else:
        logging.info(f"FFMPEG was found in System Path: {ffmpegPath}")

    return str(ffmpegPath)

def downloadAndExtractFFMPEG(ffmpegPath):
    logging.info("Getting FFMPEG")
    if platform.system() == "Windows":
        FFMPEGURL = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
        extractFunc = extractFFMPEGZip
    else:
        FFMPEGURL = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
        extractFunc = extractFFMPEGTar

    ffmpegDir = os.path.dirname(ffmpegPath)
    ffmpegArchivePath = os.path.join(ffmpegDir, "ffmpeg.tar.xz" if platform.system() != "Windows" else "ffmpeg.zip")

    if not os.path.exists(ffmpegDir):
        os.makedirs(ffmpegDir)

    response = requests.get(FFMPEGURL, stream=True)
    totalSizeInBytes = int(response.headers.get("content-length", 0))
    totalSizeInMB = int(totalSizeInBytes / 1024 / 1024)

    with alive_bar(
        total=totalSizeInMB + 1,
        title="Downloading FFmpeg",
        bar="smooth",
        unit="MB",
        spinner=True,
        enrich_print=True,
        receipt=True,
        monitor=True,
        elapsed=True,
        stats=False,
        dual_line=False,
        force_tty=True,
    ) as bar, open(ffmpegArchivePath, "wb") as file:
        for data in response.iter_content(chunk_size=1024 * 1024):
            file.write(data)
            bar(int(len(data) / (1024 * 1024)))

    extractFunc(ffmpegArchivePath, ffmpegDir)
    return str(ffmpegPath)
def extractFFMPEGZip(ffmpegZipPath, ffmpegDir):
    with zipfile.ZipFile(ffmpegZipPath, "r") as zipRef:
        zipRef.extractall(ffmpegDir)

    os.rename(
        os.path.join(ffmpegDir, "ffmpeg-master-latest-win64-gpl", "bin", "ffmpeg.exe"),
        os.path.join(ffmpegDir, "ffmpeg.exe")
    )

    os.remove(ffmpegZipPath)
    for directory in glob.glob(os.path.join(ffmpegDir, "ffmpeg-*-win64-gpl")):
        shutil.rmtree(directory)

def extractFFMPEGTar(ffmpegTarPath, ffmpegDir):
    with tarfile.open(ffmpegTarPath, "r:xz") as tarRef:
        tarRef.extractall(ffmpegDir)

    for directory in glob.glob(os.path.join(ffmpegDir, "ffmpeg-*-static")):
        os.rename(
            os.path.join(directory, "ffmpeg"),
            os.path.join(ffmpegDir, "ffmpeg")
        )
        break

    os.remove(ffmpegTarPath)
    for directory in glob.glob(os.path.join(ffmpegDir, "ffmpeg-*-static")):
        shutil.rmtree(directory)