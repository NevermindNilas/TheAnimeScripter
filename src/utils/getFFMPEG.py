import shutil
import requests
import logging
import os

from src.utils.progressBarLogic import progressBarDownloadLogic


def getFFMPEG(mainPath, sysUsed, path):
    ffmpegPath = shutil.which("ffmpeg")
    if ffmpegPath is None:
        ffmpegPath = downloadAndExtractFFMPEG(path, sysUsed)
    else:
        logging.info(f"FFMPEG was found in System Path: {ffmpegPath}")
    return str(ffmpegPath)


def downloadAndExtractFFMPEG(ffmpegPath, sysUsed):
    logging.info("Getting FFMPEG")
    extractFunc = extractFFMPEGZip if sysUsed == "Windows" else extractFFMPEGTar
    ffmpegDir = os.path.dirname(ffmpegPath)
    ffmpegArchivePath = os.path.join(
        ffmpegDir, "ffmpeg.zip" if sysUsed == "Windows" else "ffmpeg.tar.xz"
    )

    os.makedirs(ffmpegDir, exist_ok=True)

    FFMPEGURL = (
        "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
        if sysUsed == "Windows"
        else "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
    )

    response = requests.get(FFMPEGURL, stream=True)
    totalSizeInBytes = int(response.headers.get("content-length", 0))
    totalSizeInMB = totalSizeInBytes // (1024 * 1024)

    with progressBarDownloadLogic(totalSizeInMB + 1, "Downloading FFmpeg") as bar, open(
        ffmpegArchivePath, "wb"
    ) as file:
        for data in response.iter_content(chunk_size=1024 * 1024):
            file.write(data)
            bar(len(data) // (1024 * 1024))

    extractFunc(ffmpegArchivePath, ffmpegDir)
    return str(ffmpegPath)


def extractFFMPEGZip(ffmpegZipPath, ffmpegDir):
    import zipfile

    with zipfile.ZipFile(ffmpegZipPath, "r") as zipRef:
        zipRef.extractall(ffmpegDir)
    os.rename(
        os.path.join(ffmpegDir, "ffmpeg-master-latest-win64-gpl", "bin", "ffmpeg.exe"),
        os.path.join(ffmpegDir, "ffmpeg.exe"),
    )
    os.remove(ffmpegZipPath)
    shutil.rmtree(os.path.join(ffmpegDir, "ffmpeg-master-latest-win64-gpl"))


def extractFFMPEGTar(ffmpegTarPath, ffmpegDir):
    import tarfile

    with tarfile.open(ffmpegTarPath, "r:xz") as tarRef:
        tarRef.extractall(ffmpegDir)
    for item in os.listdir(ffmpegDir):
        full_path = os.path.join(ffmpegDir, item)
        if (
            os.path.isdir(full_path)
            and item.startswith("ffmpeg-")
            and item.endswith("-static")
        ):
            os.rename(
                os.path.join(full_path, "ffmpeg"), os.path.join(ffmpegDir, "ffmpeg")
            )
            shutil.rmtree(full_path)
    os.remove(ffmpegTarPath)
