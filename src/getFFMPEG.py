import shutil
import requests
import glob
import zipfile
import tarfile
import logging
from alive_progress import alive_bar
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
        dual_line=True,
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

    (ffmpegDir / "ffmpeg-master-latest-win64-gpl" / "bin" / "ffmpeg.exe").rename(
        ffmpegDir / "ffmpeg.exe"
    )

    ffmpegZipPath.unlink()
    for directory in glob.glob(str(ffmpegDir / "ffmpeg-*-win64-gpl")):
        shutil.rmtree(directory)


def extractFFMPEGTar(ffmpegTarPath, ffmpegDir):
    with tarfile.open(ffmpegTarPath, "r:xz") as tarRef:
        tarRef.extractall(ffmpegDir)

    for directory in ffmpegDir.glob("ffmpeg-*-static"):
        (directory / "ffmpeg").rename(ffmpegDir / "ffmpeg")
        break

    ffmpegTarPath.unlink()
    for directory in glob.glob(str(ffmpegDir / "ffmpeg-*-static")):
        shutil.rmtree(directory)
