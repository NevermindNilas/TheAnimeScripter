import shutil
import requests
import logging
import os

from src.utils.progressBarLogic import ProgressBarDownloadLogic


def getFFMPEG(sysUsed, path, realtime: bool = False):
    ffmpegPath = shutil.which("ffmpeg")
    ffprobePath = shutil.which("ffprobe")
    mpvPath = shutil.which("mpv") if realtime else None
    if ffmpegPath is None or (realtime and mpvPath is None) or ffprobePath is None:
        ffmpegPath, mpvPath = downloadAndExtractFfmpeg(path, sysUsed, realtime)
    else:
        logging.info(f"FFMPEG found in System Path: {ffmpegPath}")
        if realtime:
            logging.info(f"FFPLAY found in System Path: {mpvPath}")
        logging.info(f"FFPROBE found in System Path: {ffprobePath}")
    return ffmpegPath, mpvPath, ffprobePath


def downloadAndExtractFfmpeg(ffmpegPath, sysUsed, realtime):
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

    mpvURL = r"https://github.com/NevermindNilas/TAS-Modes-Host/releases/download/main/mpv.exe"

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

    if realtime:
        try:
            response = requests.get(mpvURL, stream=True)
            response.raise_for_status()
            totalSizeInBytes = int(response.headers.get("content-length", 0))
            totalSizeInMB = totalSizeInBytes // (1024 * 1024)

            mpvPath = os.path.join(ffmpegDir, "mpv.exe")
            with (
                ProgressBarDownloadLogic(totalSizeInMB + 1, "Downloading MPV") as bar,
                open(mpvPath, "wb") as file,
            ):
                for data in response.iter_content(chunk_size=1024 * 1024):
                    if data:
                        file.write(data)
                        bar(len(data) // (1024 * 1024))
        except requests.RequestException as e:
            logging.error(f"Failed to download MPV: {e}")
            raise

    extractFunc(ffmpegArchivePath, ffmpegDir)
    return str(ffmpegPath), str(ffmpegPath).replace(
        "ffmpeg", "mpv"
    ) if realtime else None


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


def extractFfmpegTar(ffmpegTarPath, ffmpegDir, realtime):
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
                ffmpeg_src = os.path.join(fullPath, "ffmpeg")
                ffmpeg_dst = os.path.join(ffmpegDir, "ffmpeg")
                if not os.path.exists(ffmpeg_dst):
                    os.rename(ffmpeg_src, ffmpeg_dst)
                if realtime:
                    ffplay_src = os.path.join(fullPath, "ffplay")
                    ffplay_dst = os.path.join(ffmpegDir, "ffplay")
                    if not os.path.exists(ffplay_dst):
                        os.rename(ffplay_src, ffplay_dst)
                shutil.rmtree(fullPath)
    except tarfile.TarError as e:
        logging.error(f"Failed to extract TAR: {e}")
        raise
    finally:
        os.remove(ffmpegTarPath)
