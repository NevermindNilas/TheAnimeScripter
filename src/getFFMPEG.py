import os
import shutil
import requests
import glob
import zipfile
import logging

from tqdm import tqdm

def getFFMPEG():
    ffmpegPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "ffmpeg", "ffmpeg.exe")
    logging.info(f"FFMPEG Path: {ffmpegPath}")

    if not os.path.isfile(ffmpegPath):
        downloadAndExtractFFMPEG(ffmpegPath)

    return ffmpegPath

def downloadAndExtractFFMPEG(ffmpegPath):
    logging.info("Getting FFMPEG")
    print("Couldn't find FFMPEG, downloading it now. This will take a few seconds on the first run, but will be cached for future runs.")

    FFMPEG_URL = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
    ffmpegDir = os.path.dirname(ffmpegPath)

    os.makedirs(ffmpegDir, exist_ok=True)
    ffmpegZipPath = os.path.join(ffmpegDir, "ffmpeg.zip")

    response = requests.get(FFMPEG_URL, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

    with open(ffmpegZipPath, 'wb') as file:
        for data in response.iter_content(chunk_size=1024):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

    if progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")

    extractFFMPEG(ffmpegZipPath, ffmpegDir)

    print("\n")
    return ffmpegPath

def extractFFMPEG(ffmpegZipPath, ffmpegDir):
    with zipfile.ZipFile(ffmpegZipPath, "r") as zipRef:
        zipRef.extractall(ffmpegDir)

    for root, dirs, files in os.walk(ffmpegDir):
        for file in files:
            if file == "ffmpeg.exe":
                shutil.move(os.path.join(root, file), ffmpegDir)

    os.remove(ffmpegZipPath)
    for directory in glob.glob(os.path.join(ffmpegDir, "ffmpeg-*-win64-gpl")):
        shutil.rmtree(directory)