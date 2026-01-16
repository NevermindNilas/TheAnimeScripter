import shutil
import requests
import logging
import os
import subprocess

import src.constants as cs
from src.utils.progressBarLogic import ProgressBarDownloadLogic


def getFFMPEG():
    ffmpegPath = downloadAndExtractFfmpeg(cs.FFMPEGPATH)
    cs.FFMPEGPATH = ffmpegPath
    ffProbeExe = "ffprobe.exe" if cs.SYSTEM == "Windows" else "ffprobe"
    cs.FFPROBEPATH = os.path.join(os.path.dirname(ffmpegPath), ffProbeExe)
    logFfmpegVersion(cs.FFMPEGPATH)


def downloadAndExtractFfmpeg(ffmpegPath):
    logging.info("Downloading FFMPEG")
    extractFunc = extractFfmpegZip if cs.SYSTEM == "Windows" else extractFfmpegTar
    ffmpegDir = os.path.dirname(ffmpegPath)
    archiveExtension = "ffmpeg.zip" if cs.SYSTEM == "Windows" else "ffmpeg.tar.xz"
    ffmpegArchivePath = os.path.join(ffmpegDir, archiveExtension)

    os.makedirs(ffmpegDir, exist_ok=True)

    ffmpegUrl = (
        "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl-shared.zip"
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


def logFfmpegVersion(ffmpegPath: str) -> None:
    try:
        if not ffmpegPath or not os.path.exists(ffmpegPath):
            logging.warning("FFmpeg path not found to log version.")
            return
        proc = subprocess.run(
            [ffmpegPath, "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        output = proc.stdout.splitlines()
        if output:
            logging.info(f"FFmpeg version: {output[0]}")
            if len(output) > 1:
                logging.debug(f"FFmpeg build: {output[1]}")
        else:
            logging.info("FFmpeg version: <no output>")
    except Exception as e:
        logging.warning(f"Unable to retrieve FFmpeg version: {e}")


def extractFfmpegZip(ffmpegZipPath, ffmpegDir):
    import zipfile

    try:
        with zipfile.ZipFile(ffmpegZipPath, "r") as zipRef:
            zipRef.extractall(ffmpegDir)

        extractedRoot = os.path.join(ffmpegDir, "ffmpeg-master-latest-win64-gpl-shared")
        bin_dir = os.path.join(extractedRoot, "bin")

        if os.path.exists(bin_dir):
            for item in os.listdir(bin_dir):
                s = os.path.join(bin_dir, item)
                d = os.path.join(ffmpegDir, item)
                if os.path.exists(d):
                    try:
                        if os.path.isdir(d):
                            shutil.rmtree(d)
                        else:
                            os.remove(d)
                    except Exception as e:
                        logging.warning(f"Failed to remove existing file {d}: {e}")

                shutil.move(s, d)

        if os.path.exists(extractedRoot):
            shutil.rmtree(extractedRoot, onerror=remove_readonly)

    except zipfile.BadZipFile as e:
        logging.error(f"Failed to extract ZIP: {e}")
        raise
    finally:
        if os.path.exists(ffmpegZipPath):
            os.remove(ffmpegZipPath)


def extractFfmpegTar(ffmpegTarPath, ffmpegDir):
    import tarfile
    import stat

    try:
        with tarfile.open(ffmpegTarPath, "r:xz") as tarRef:
            tarRef.extractall(ffmpegDir)

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
            ffmpeg_src = os.path.join(extracted_dir, "ffmpeg")
            ffmpeg_dst = os.path.join(ffmpegDir, "ffmpeg")
            if os.path.exists(ffmpeg_src) and not os.path.exists(ffmpeg_dst):
                os.rename(ffmpeg_src, ffmpeg_dst)
                os.chmod(ffmpeg_dst, os.stat(ffmpeg_dst).st_mode | stat.S_IEXEC)

            ffprobe_src = os.path.join(extracted_dir, "ffprobe")
            ffprobe_dst = os.path.join(ffmpegDir, "ffprobe")
            if os.path.exists(ffprobe_src) and not os.path.exists(ffprobe_dst):
                os.rename(ffprobe_src, ffprobe_dst)
                os.chmod(ffprobe_dst, os.stat(ffprobe_dst).st_mode | stat.S_IEXEC)

            shutil.rmtree(extracted_dir, onerror=remove_readonly)

    except tarfile.TarError as e:
        logging.error(f"Failed to extract TAR: {e}")
        raise
    finally:
        if os.path.exists(ffmpegTarPath):
            os.remove(ffmpegTarPath)


def remove_readonly(func, path, excinfo):
    import stat
    import time
    import logging

    try:
        os.chmod(path, stat.S_IWRITE)
    except Exception:
        pass

    try:
        func(path)
    except Exception:
        time.sleep(1)
        try:
            func(path)
        except Exception as e:
            logging.warning(f"Failed to remove {path}: {e}")
