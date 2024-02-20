import os
import shutil
import wget
import glob
import zipfile
import logging

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

    wget.download(FFMPEG_URL, out=ffmpegZipPath)
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