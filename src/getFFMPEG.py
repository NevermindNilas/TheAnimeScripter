import shutil
import requests
import glob
import zipfile
import logging
from tqdm import tqdm
from pathlib import Path

def getFFMPEG():
    ffmpegPath = shutil.which("ffmpeg")

    # Just making sure here.
    if ffmpegPath is None:
        ffmpegPath = shutil.which("FFMPEG")
    else:
        logging.info(f"FFMPEG was found in System Path: {ffmpegPath}")

    if ffmpegPath is None:
        ffmpegPath = Path(__file__).resolve().parent / "ffmpeg" / "ffmpeg.exe"
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

    FFMPEGURL = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
    ffmpegDir = ffmpegPath.parent

    ffmpegDir.mkdir(parents=True, exist_ok=True)
    ffmpegZipPath = ffmpegDir / "ffmpeg.zip"

    response = requests.get(FFMPEGURL, stream=True)
    total_size_in_bytes = int(response.headers.get("content-length", 0))

    with tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True, colour="green") as progress_bar, open(ffmpegZipPath, "wb") as file:
        for data in response.iter_content(chunk_size=1024):
            progress_bar.update(len(data))
            file.write(data)

    extractFFMPEG(ffmpegZipPath, ffmpegDir)

    print("\n")
    return str(ffmpegPath)


def extractFFMPEG(ffmpegZipPath, ffmpegDir):
    with zipfile.ZipFile(ffmpegZipPath, "r") as zipRef:
        zipRef.extractall(ffmpegDir)

    (ffmpegDir / "ffmpeg-master-latest-win64-gpl" / "bin" / "ffmpeg.exe").rename(ffmpegDir / "ffmpeg.exe")

    ffmpegZipPath.unlink()
    for directory in glob.glob(str(ffmpegDir / "ffmpeg-*-win64-gpl")):
        shutil.rmtree(directory)