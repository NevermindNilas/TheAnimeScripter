import shutil
import requests
import logging
import os

from src.utils.progressBarLogic import ProgressBarDownloadLogic


def getFFMPEG(mainPath, sysUsed, path):
    ffmpegPath = shutil.which("ffmpeg")
    if ffmpegPath is None:
        ffmpegPath = downloadAndExtractFfmpeg(path, sysUsed)
    else:
        logging.info(f"FFMPEG found in System Path: {ffmpegPath}")
    return str(ffmpegPath)


def downloadAndExtractFfmpeg(ffmpegPath, sysUsed):
    logging.info("Downloading FFMPEG")
    extractFunc = extractFfmpegZip if sysUsed == "Windows" else extractFfmpegTar
    ffmpegDir = os.path.dirname(ffmpegPath)
    archiveExtension = "ffmpeg.zip" if sysUsed == "Windows" else "ffmpeg.tar.xz"
    ffmpegArchivePath = os.path.join(ffmpegDir, archiveExtension)

    os.makedirs(ffmpegDir, exist_ok=True)

    ffmpegUrl = (
        "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
        if sysUsed == "Windows"
        else "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
    )

    try:
        response = requests.get(ffmpegUrl, stream=True)
        response.raise_for_status()
        totalSizeInBytes = int(response.headers.get("content-length", 0))
        totalSizeInMB = totalSizeInBytes // (1024 * 1024)

        with ProgressBarDownloadLogic(
            totalSizeInMB + 1, "Downloading FFmpeg"
        ) as bar, open(ffmpegArchivePath, "wb") as file:
            for data in response.iter_content(chunk_size=1024 * 1024):
                if data:
                    file.write(data)
                    bar(len(data) // (1024 * 1024))
    except requests.RequestException as e:
        logging.error(f"Failed to download FFMPEG: {e}")
        raise

    extractFunc(ffmpegArchivePath, ffmpegDir)
    return str(ffmpegPath)


def extractFfmpegZip(ffmpegZipPath, ffmpegDir):
    import zipfile

    try:
        with zipfile.ZipFile(ffmpegZipPath, "r") as zipRef:
            zipRef.extractall(ffmpegDir)
        os.rename(
            os.path.join(
                ffmpegDir, "ffmpeg-master-latest-win64-gpl", "bin", "ffmpeg.exe"
            ),
            os.path.join(ffmpegDir, "ffmpeg.exe"),
        )
    except zipfile.BadZipFile as e:
        logging.error(f"Failed to extract ZIP: {e}")
        raise
    finally:
        os.remove(ffmpegZipPath)
        shutil.rmtree(os.path.join(ffmpegDir, "ffmpeg-master-latest-win64-gpl"))


def extractFfmpegTar(ffmpegTarPath, ffmpegDir):
    import tarfile

    try:
        with tarfile.open(ffmpegTarPath, "r:xz") as tarRef:
            tarRef.extractall(ffmpegDir)
        for item in os.listdir(ffmpegDir):
            fullPath = os.path.join(ffmpegDir, item)
            if (
                os.path.isdir(fullPath)
                and item.startswith("ffmpeg-")
                and item.endswith("-static")
            ):
                os.rename(
                    os.path.join(fullPath, "ffmpeg"),
                    os.path.join(ffmpegDir, "ffmpeg"),
                )
                shutil.rmtree(fullPath)
    except tarfile.TarError as e:
        logging.error(f"Failed to extract TAR: {e}")
        raise
    finally:
        os.remove(ffmpegTarPath)
