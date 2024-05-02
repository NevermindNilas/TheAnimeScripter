import os
import logging
import inquirer

from yt_dlp import YoutubeDL
from .ffmpegSettings import matchEncoder


class VideoDownloader:
    def __init__(self, video_link, output, encodeMethod, customEncoder, ffmpegPath:str = None):
        self.link = video_link
        self.output = output
        self.encodeMethod = encodeMethod
        self.customEncoder = customEncoder
        self.ffmpegPath = ffmpegPath

        resolutions = self.listResolutions()

        questions = [
            inquirer.List('resolution',
                          message="Select the resolution you want to download, use up and down arrow keys to navigate and press enter to select:",
                          choices=resolutions,
                          ),
        ]

        answers = inquirer.prompt(questions)
        if not answers:
            logging.error('No resolution selected, exiting')
            exit(1)
        

        self.resolution = answers['resolution']

        if self.resolution > 1080:
            toPrint = f"The selected resolution {self.resolution} is higher than 1080p, this will require an aditional step of encoding the video for compatibility with After Effects, [WIP]"
            logging.warning(toPrint)
            print(toPrint)
        else:
            toPrint = f"Selected resolution: {self.resolution}"
            logging.info(toPrint)
            print(toPrint)

        self.downloadVideo()


    def listResolutions(self):
        options = {
            'listformats': False,
            'quiet': True,
            'no_warnings': True,
        }
        with YoutubeDL(options) as ydl:
            info_dict = ydl.extract_info(self.link, download=False)
            formats = info_dict.get('formats', [])
            resolutions = [f.get('height') for f in formats if f.get('height') and f.get('height') >= 240]
            return sorted(set(resolutions), reverse=True)
    

    def downloadVideo(self):
        options = self.getOptions()
        with YoutubeDL(options) as ydl:
            ydl.download([self.link])

    def getOptions(self):
        if self.resolution > 1080:
            return {
                "format": f"bestvideo[height<={self.resolution}]+bestaudio[ext=m4a]/best[ext=mp4]",
                "outtmpl": os.path.splitext(self.output)[0],
                "ffmpeg_location": os.path.dirname(self.ffmpegPath),
                "quiet": True,
                "noplaylist": True,
                "no_warnings": True,
                "postprocessors": [{
                    "key": "FFmpegVideoConvertor",
                    "preferedformat": "mp4",
                    #self.customEncoder if self.customEncoder else matchEncoder(self.encodeMethod),
                }],
            }
        else:
            return {
                "format": f"bestvideo[height<={self.resolution}][vcodec^=avc1]+bestaudio[ext=m4a]/best[ext=mp4]",
                "outtmpl": self.output,
                "ffmpeg_location": os.path.dirname(self.ffmpegPath),
                "quiet": True,
                "noplaylist": True,
                "no_warnings": True,
            }
