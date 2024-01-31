import os
import logging
import subprocess

from yt_dlp import YoutubeDL


class ytdlp():
    def __init__(self, ytdlp, output, ytdlp_quality, encode_method):
        self.link = ytdlp
        self.output = output
        self.ytdlp_quality = ytdlp_quality
        self.encode_method = encode_method
        self.run()

    def run(self):
        from .get_ffmpeg import get_ffmpeg

        ffmpeg_path = os.path.join(os.path.dirname(
            __file__), "ffmpeg", "ffmpeg.exe")

        if not os.path.isfile(ffmpeg_path):
            ffmpeg_path = get_ffmpeg()

        if not self.ytdlp_quality:
            ydl_opts = {
                'format': 'bestvideo[ext=mp4][vcodec^=avc1]+bestaudio[ext=m4a]/best[ext=mp4]',
                'outtmpl': self.output,
                'ffmpeg_location': os.path.dirname(ffmpeg_path),
            }

        else:
            temp_name = os.path.splitext(self.output)[0] + '.webm'

            logging.info(
                f"Downloading video in webm format to: {temp_name}")

            ydl_opts = {
                'format': 'bestvideo+bestaudio',
                'outtmpl': temp_name,
                'ffmpeg_location': os.path.dirname(ffmpeg_path),
            }

        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([self.link])

        if self.ytdlp_quality:
            from .ffmpegSettings import encodeYTDLP
            command = encodeYTDLP(temp_name, self.output,
                                  ffmpeg_path, self.encode_method)
            subprocess.run(command)
            os.remove(temp_name)

            logging.info(
                f"Removing residual webm file: {temp_name}")

        logging.info(
            "Downloaded video from: " + self.link)

        logging.info(
            f"Saved video to: {self.output}")

        return
