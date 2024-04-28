import os
import logging
import subprocess
import inquirer

from yt_dlp import YoutubeDL
from .ffmpegSettings import encodeYTDLP


class VideoDownloader:
    def __init__(self, video_link, output, encodeMethod, customEncoder, ffmpeg_path:str = None):
        self.link = video_link
        self.output = output
        self.encodeMethod = encodeMethod
        self.customEncoder = customEncoder
        self.ffmpeg_path = ffmpeg_path

        resolutions = self.listResolutions()

        questions = [
            inquirer.List('resolution',
                          message="Select the resolution you want to download, use up and down arrow keys to navigate and press enter to select:",
                          choices=resolutions,
                          ),
        ]

        answers = inquirer.prompt(questions)
        print('Selected resolution:', answers['resolution'])

        self.quality = answers['resolution']

        self.downloadVideo()


    def listResolutions(self):
        ydl_opts = {
            'listformats': False,
            'quiet': True,
            'no_warnings': True,
        }
        with YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(self.link, download=False)
            formats = info_dict.get('formats', [])
            resolutions = [f.get('height') for f in formats if f.get('height') and f.get('height') >= 360]
            return sorted(set(resolutions))
    

    def downloadVideo(self):
        ydl_opts = self.getOptions()
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([self.link])

    def getOptions(self):
        return {
            "format": f"bestvideo[height<={self.quality}][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
            "outtmpl": self.output,
            "ffmpeg_location": os.path.dirname(self.ffmpeg_path),
            "quiet": True,
            "noplaylist": True,
            "no_warnings": True,
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