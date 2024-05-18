import shutil
import requests
import glob
import zipfile
import tarfile
import logging
from tqdm import tqdm
from pathlib import Path
import platform


def getFFMPEG():
    ffmpegPath = shutil.which("ffmpeg")

    if ffmpegPath is None:
        if platform.system() == "Windows":
            ffmpegPath = Path(__file__).resolve().parent / "ffmpeg" / "ffmpeg.exe"
        else:
            ffmpegPath = Path(__file__).resolve().parent / "ffmpeg" / "ffmpeg"

        logging.info(f"FFMPEG Path: {ffmpegPath}")

        if not ffmpegPath.is_file():
            ffmpegPath = downloadAndExtractFFMPEG(ffmpegPath)
    else:
        logging.info(f"FFMPEG was found in System Path: {ffmpegPath}")

    return str(ffmpegPath)


def downloadAndExtractFFMPEG(ffmpegPath):
    logging.info("Getting FFMPEG")
    print(
        "Couldn't find FFMPEG, downloading it now. This will take a few seconds on the first run, but will be cached for future runs."
    )

    if platform.system() == "Windows":
        FFMPEGURL = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
        extractFunc = extractFFMPEGZip
    else:
        FFMPEGURL = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
        extractFunc = extractFFMPEGTar

    ffmpegDir = ffmpegPath.parent
    ffmpegArchivePath = (
        ffmpegDir / "ffmpeg.tar.xz"
        if platform.system() != "Windows"
        else ffmpegDir / "ffmpeg.zip"
    )

    ffmpegDir.mkdir(parents=True, exist_ok=True)

    response = requests.get(FFMPEGURL, stream=True)
    totalSizeInBytes = int(response.headers.get("content-length", 0))

    with tqdm(
        total=totalSizeInBytes, unit="iB", unit_scale=True, colour="green"
    ) as progress_bar, open(ffmpegArchivePath, "wb") as file:
        for data in response.iter_content(chunk_size=1024):
            progress_bar.update(len(data))
            file.write(data)

    extractFunc(ffmpegArchivePath, ffmpegDir)

    print("\n")
    return str(ffmpegPath)


def extractFFMPEGZip(ffmpegZipPath, ffmpegDir):
    with zipfile.ZipFile(ffmpegZipPath, "r") as zipRef:
        zipRef.extractall(ffmpegDir)

    (ffmpegDir / "ffmpeg-master-latest-win64-gpl" / "bin" / "ffmpeg.exe").rename(
        ffmpegDir / "ffmpeg.exe"
    )

    ffmpegZipPath.unlink()
    for directory in glob.glob(str(ffmpegDir / "ffmpeg-*-win64-gpl")):
        shutil.rmtree(directory)


def extractFFMPEGTar(ffmpegTarPath, ffmpegDir):
    with tarfile.open(ffmpegTarPath, "r:xz") as tarRef:
        tarRef.extractall(ffmpegDir)

    (ffmpegDir / "ffmpeg-*-static" / "ffmpeg").rename(ffmpegDir / "ffmpeg")

    ffmpegTarPath.unlink()
    for directory in glob.glob(str(ffmpegDir / "ffmpeg-*-static")):
        shutil.rmtree(directory)
