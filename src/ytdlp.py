from yt_dlp import YoutubeDL
import os
import logging

class ytdlp():
    def __init__(self, ytdlp, output):
        self.link = ytdlp
        self.output = output
        self.run()
    
    def run(self):
        from .get_ffmpeg import get_ffmpeg
        
        ffmpeg_path = os.path.join(os.path.dirname(__file__), "ffmpeg", "ffmpeg.exe")
        
        if not os.path.isfile(ffmpeg_path):
            logging.info("Couldn't find FFMPEG, downloading it now...")
            print("Couldn't find FFMPEG, downloading it now..., this will add a few aditional seconds to the process.")
            get_ffmpeg(ffmpeg_path)
        
        ydl_opts = {
            'format': 'bestvideo[ext=mp4][vcodec^=avc1]+bestaudio[ext=m4a]/best[ext=mp4]',
            'outtmpl': self.output,
            'ffmpeg_location': os.path.dirname(ffmpeg_path),
        }
        
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([self.link])

        logging.info(
            "Downloaded video from: " + self.link)
        
        logging.info(
            f"Saved video to: {self.output}")
        
        return