import os
import shutil
import wget
import glob
import zipfile
import picologging as logging

#https://jeremylee.sh/bins/ffmpeg.7z

def get_ffmpeg():
    """
    This script will download and extract the latest ffmpeg.exe binary for windows.
    I do not need any other files from the ffmpeg build, so anything unnecessary is removed.
    """
    
    logging.info(
        "Getting FFMPEG")
    
    print(
        "Couldn't find FFMPEG, downloading it now, this will add a few seconds onto the first run, but it will be cached for future runs.")
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    ffmpeg_path = os.path.join(dir_path, "ffmpeg", "ffmpeg.exe")
    
    logging.info(
        f"FFMPEG path: {ffmpeg_path}")
    
    FFMPEG_URL = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
    ffmpeg_dir = os.path.dirname(ffmpeg_path)

    os.makedirs(ffmpeg_dir, exist_ok=True)
    ffmpeg_zip_path = os.path.join(ffmpeg_dir, "ffmpeg.zip")

    wget.download(FFMPEG_URL, out=ffmpeg_zip_path)

    with zipfile.ZipFile(ffmpeg_zip_path, 'r') as zip_ref:
        zip_ref.extractall(ffmpeg_dir)

    for root, dirs, files in os.walk(ffmpeg_dir):
        for file in files:
            if file == "ffmpeg.exe":
                shutil.move(os.path.join(root, file), ffmpeg_dir)

    os.remove(ffmpeg_zip_path)
    for directory in glob.glob(os.path.join(ffmpeg_dir, "ffmpeg-*-win64-gpl")):
        shutil.rmtree(directory)
    
    # Force a new line
    print("\n")
        
    return ffmpeg_path
