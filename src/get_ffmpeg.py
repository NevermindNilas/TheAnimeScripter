import os
import shutil
import wget
import glob
import zipfile

def get_ffmpeg():
    FFMPEG_URL = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"

    ffmpeg_dir = os.path.join(os.getcwd(), "ffmpeg")

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
