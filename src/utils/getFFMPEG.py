import shutil
import requests
import logging
import os

import src.constants as cs
from src.utils.progressBarLogic import ProgressBarDownloadLogic


def getFFMPEG():
    ffmpegPath = None
    ffprobePath = None
    if ffmpegPath is None or ffprobePath is None:
        ffmpegPath = downloadAndExtractFfmpeg(cs.FFMPEGPATH)
    else:
        logging.info(f"FFMPEG found in System Path: {ffmpegPath}")
        logging.info(f"FFPROBE found in System Path: {ffprobePath}")

    cs.FFMPEGPATH = ffmpegPath
    ffProbeExe = "ffprobe.exe" if cs.SYSTEM == "Windows" else "ffprobe"
    cs.FFPROBEPATH = os.path.join(os.path.dirname(ffmpegPath), ffProbeExe)


def downloadAndExtractFfmpeg(ffmpegPath):
    logging.info("Downloading FFMPEG")
    extractFunc = extractFfmpegZip if cs.SYSTEM == "Windows" else extractFfmpegTar
    ffmpegDir = os.path.dirname(ffmpegPath)
    archiveExtension = "ffmpeg.zip" if cs.SYSTEM == "Windows" else "ffmpeg.tar.xz"
    ffmpegArchivePath = os.path.join(ffmpegDir, archiveExtension)

    os.makedirs(ffmpegDir, exist_ok=True)

    ffmpegUrl = (
        "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
        if cs.SYSTEM == "Windows"
        else "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
    )

    try:
        response = requests.get(ffmpegUrl, stream=True)
        response.raise_for_status()
        totalSizeInBytes = int(response.headers.get("content-length", 0))
        totalSizeInMB = totalSizeInBytes // (1024 * 1024)

        with (
            ProgressBarDownloadLogic(totalSizeInMB + 1, "Downloading FFmpeg") as bar,
            open(ffmpegArchivePath, "wb") as file,
        ):
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
        ffmpeg_src = os.path.join(
            ffmpegDir, "ffmpeg-master-latest-win64-gpl", "bin", "ffmpeg.exe"
        )
        ffmpeg_dst = os.path.join(ffmpegDir, "ffmpeg.exe")
        if not os.path.exists(ffmpeg_dst):
            os.rename(ffmpeg_src, ffmpeg_dst)

        ffprobe_src = os.path.join(
            ffmpegDir, "ffmpeg-master-latest-win64-gpl", "bin", "ffprobe.exe"
        )
        ffprobe_dst = os.path.join(ffmpegDir, "ffprobe.exe")
        if not os.path.exists(ffprobe_dst):
            os.rename(ffprobe_src, ffprobe_dst)

    except zipfile.BadZipFile as e:
        logging.error(f"Failed to extract ZIP: {e}")
        raise
    finally:
        os.remove(ffmpegZipPath)
        shutil.rmtree(os.path.join(ffmpegDir, "ffmpeg-master-latest-win64-gpl"))


def extractFfmpegTar(ffmpegTarPath, ffmpegDir):
    import tarfile
    import stat

    try:
        with tarfile.open(ffmpegTarPath, "r:xz") as tarRef:
            tarRef.extractall(ffmpegDir)
        
        # Find the extracted directory
        extracted_dir = None
        for item in os.listdir(ffmpegDir):
            fullPath = os.path.join(ffmpegDir, item)
            if (
                os.path.isdir(fullPath)
                and item.startswith("ffmpeg-")
                and item.endswith("-static")
            ):
                extracted_dir = fullPath
                break
        
        if extracted_dir:
            # Move ffmpeg binary
            ffmpeg_src = os.path.join(extracted_dir, "ffmpeg")
            ffmpeg_dst = os.path.join(ffmpegDir, "ffmpeg")
            if os.path.exists(ffmpeg_src) and not os.path.exists(ffmpeg_dst):
                os.rename(ffmpeg_src, ffmpeg_dst)
                # Make executable
                os.chmod(ffmpeg_dst, os.stat(ffmpeg_dst).st_mode | stat.S_IEXEC)
            
            # Move ffprobe binary
            ffprobe_src = os.path.join(extracted_dir, "ffprobe")
            ffprobe_dst = os.path.join(ffmpegDir, "ffprobe")
            if os.path.exists(ffprobe_src) and not os.path.exists(ffprobe_dst):
                os.rename(ffprobe_src, ffprobe_dst)
                # Make executable
                os.chmod(ffprobe_dst, os.stat(ffprobe_dst).st_mode | stat.S_IEXEC)
            
            # Clean up extracted directory
            shutil.rmtree(extracted_dir)
        
    except tarfile.TarError as e:
        logging.error(f"Failed to extract TAR: {e}")
        raise
    finally:
        if os.path.exists(ffmpegTarPath):
            os.remove(ffmpegTarPath)
