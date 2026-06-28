import logging
import os
import shutil
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import src.constants as cs

_ffmpegDllDirectoryHandle = None


def addFfmpegToDllSearchPath(ffmpegPath: str | None = None) -> None:
    global _ffmpegDllDirectoryHandle

    if cs.SYSTEM != "Windows":
        return

    ffmpegPath = ffmpegPath or cs.FFMPEGPATH
    ffmpeg_dir = os.path.dirname(ffmpegPath)
    if not ffmpeg_dir or not os.path.exists(ffmpeg_dir):
        return

    try:
        _ffmpegDllDirectoryHandle = os.add_dll_directory(ffmpeg_dir)
        logging.info(f"Added FFmpeg directory to DLL search path: {ffmpeg_dir}")
    except Exception as e:
        logging.warning(f"Failed to add FFmpeg to DLL search path: {e}")


def getFFMPEG():
    ffmpegPath = downloadAndExtractFfmpeg(cs.FFMPEGPATH)
    cs.FFMPEGPATH = ffmpegPath
    if not cs.FFPROBEPATH or not os.path.exists(cs.FFPROBEPATH):
        ffProbeExe = "ffprobe.exe" if cs.SYSTEM == "Windows" else "ffprobe"
        cs.FFPROBEPATH = os.path.join(os.path.dirname(ffmpegPath), ffProbeExe)
    addFfmpegToDllSearchPath(cs.FFMPEGPATH)


def findSystemFfmpeg() -> tuple[str, str] | None:
    """Return system ffmpeg/ffprobe when both are available."""
    searchPaths = [
        None,
        "/opt/homebrew/bin",
        "/usr/local/bin",
        "/usr/bin",
    ]

    ffmpegPath = None
    ffprobePath = None
    for searchPath in searchPaths:
        ffmpegCandidate = shutil.which("ffmpeg", path=searchPath)
        ffprobeCandidate = shutil.which("ffprobe", path=searchPath)
        if ffmpegCandidate and ffprobeCandidate:
            ffmpegPath = ffmpegCandidate
            ffprobePath = ffprobeCandidate
            break

    if ffmpegPath and ffprobePath:
        return ffmpegPath, ffprobePath
    return None


def _downloadFile(url: str, destination: str, label: str) -> None:
    from src.infra.progressBarLogic import ProgressBarDownloadLogic

    with urlopen(url) as response:
        # Check for HTTP errors manually (like raise_for_status)
        if response.getcode() != 200:
            raise HTTPError(url, response.getcode(), None, None, None)

        totalSizeInBytes = int(response.headers.get("content-length", 0))

        with (
            ProgressBarDownloadLogic(totalSizeInBytes or 1, label) as bar,
            open(destination, "wb") as file,
        ):
            while True:
                data = response.read(1024 * 1024)
                if not data:
                    break
                file.write(data)
                bar(len(data))


def downloadAndExtractFfmpeg(ffmpegPath):
    logging.info("Downloading FFMPEG")
    ffmpegDir = os.path.dirname(ffmpegPath)
    if cs.SYSTEM == "Darwin":
        systemFfmpeg = findSystemFfmpeg()
        if systemFfmpeg is not None:
            cs.FFPROBEPATH = systemFfmpeg[1]
            logging.info(f"Using system FFmpeg: {systemFfmpeg[0]}")
            return systemFfmpeg[0]

        raise RuntimeError(
            "FFmpeg and FFprobe are required on macOS. Install native Apple "
            "Silicon FFmpeg with `brew install ffmpeg`, or place ffmpeg and "
            "ffprobe on PATH."
        )

    extractFunc = extractFfmpegZip if cs.SYSTEM == "Windows" else extractFfmpegTar
    archiveExtension = "ffmpeg.zip" if cs.SYSTEM == "Windows" else "ffmpeg.tar.xz"
    ffmpegArchivePath = os.path.join(ffmpegDir, archiveExtension)

    os.makedirs(ffmpegDir, exist_ok=True)

    ffmpegUrl = (
        "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl-shared.zip"
        if cs.SYSTEM == "Windows"
        else "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
    )

    try:
        _downloadFile(ffmpegUrl, ffmpegArchivePath, "Downloading FFmpeg")
    except (URLError, HTTPError) as e:
        logging.error(f"Failed to download FFMPEG: {e}")
        raise

    extractFunc(ffmpegArchivePath, ffmpegDir)
    return str(ffmpegPath)


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
    import stat
    import tarfile

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
    import logging
    import stat
    import time

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
