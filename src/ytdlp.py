import os
import logging
import subprocess

from yt_dlp import YoutubeDL
from .ffmpegSettings import encodeYTDLP


class VideoDownloader:
    def __init__(self, video_link, output, quality, encodeMethod, customEncoder, ffmpeg_path:str = None):
        self.link = video_link
        self.output = output
        self.quality = quality
        self.encodeMethod = encodeMethod
        self.customEncoder = customEncoder
        self.ffmpeg_path = ffmpeg_path

        self.downloadVideo()
        if self.quality:
            self.encodeVideo()
            self.cleanup()

    def downloadVideo(self):
        ydl_opts = self.getOptions()
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([self.link])

    def getOptions(self):
        if not self.quality:
            return {
                "format": "bestvideo[ext=mp4][vcodec^=avc1]+bestaudio[ext=m4a]/best[ext=mp4]",
                "outtmpl": self.output,
                "ffmpeg_location": os.path.dirname(self.ffmpeg_path),
            }
        else:
            self.temp_name = os.path.splitext(self.output)[0] + ".webm"
            logging.info(f"Downloading video in webm format to: {self.temp_name}")
            return {
                "format": "bestvideo+bestaudio",
                "outtmpl": self.temp_name,
                "ffmpeg_location": os.path.dirname(self.ffmpeg_path),
            }

    def encodeVideo(self):
        command = encodeYTDLP(
            self.temp_name,
            self.output,
            self.ffmpeg_path,
            self.encodeMethod,
            self.customEncoder,
        )
        
        subprocess.run(command)

    def cleanup(self):
        os.remove(self.temp_name)
        logging.info(f"Removing residual webm file: {self.temp_name}")