import os
import logging
import subprocess

from yt_dlp import YoutubeDL
from .get_ffmpeg import get_ffmpeg
from .ffmpegSettings import encodeYTDLP

class VideoDownloader():
    def __init__(self, video_link, output, quality, encode_method, custom_encoder):
        self.link = video_link
        self.output = output
        self.quality = quality
        self.encode_method = encode_method
        self.custom_encoder = custom_encoder
        
        self.setup_ffmpeg()
        self.download_video()
        if self.quality:
            self.encode_video()
            self.cleanup()

    def setup_ffmpeg(self):
        ffmpeg_path = os.path.join(os.path.dirname(__file__), "ffmpeg", "ffmpeg.exe")
        if not os.path.isfile(ffmpeg_path):
            self.ffmpeg_path = get_ffmpeg()
        else:
            self.ffmpeg_path = ffmpeg_path

    def download_video(self):
        ydl_opts = self.get_ydl_opts()
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([self.link])

    def get_ydl_opts(self):
        if not self.quality:
            return {
                'format': 'bestvideo[ext=mp4][vcodec^=avc1]+bestaudio[ext=m4a]/best[ext=mp4]',
                'outtmpl': self.output,
                'ffmpeg_location': os.path.dirname(self.ffmpeg_path),
            }
        else:
            self.temp_name = os.path.splitext(self.output)[0] + '.webm'
            logging.info(f"Downloading video in webm format to: {self.temp_name}")
            return {
                'format': 'bestvideo+bestaudio',
                'outtmpl': self.temp_name,
                'ffmpeg_location': os.path.dirname(self.ffmpeg_path),
            }

    def encode_video(self):
        command = encodeYTDLP(self.temp_name, self.output, self.ffmpeg_path, self.encode_method, self.custom_encoder)
        subprocess.run(command)

    def cleanup(self):
        os.remove(self.temp_name)
        logging.info(f"Removing residual webm file: {self.temp_name}")

    def log_success(self):
        logging.info("Downloaded video from: " + self.link)
        logging.info(f"Saved video to: {self.output}")