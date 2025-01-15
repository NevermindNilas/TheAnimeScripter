import os
import logging

from inquirer import List, prompt
from yt_dlp import YoutubeDL


class VideoDownloader:
    def __init__(
        self,
        video_link,
        output,
        encodeMethod,
        customEncoder,
        ffmpegPath: str = None,
        ae: bool = False,
    ):
        self.link = video_link
        self.output = output
        self.encodeMethod = encodeMethod
        self.customEncoder = customEncoder
        self.ffmpegPath = ffmpegPath
        self.ae = ae

        try:
            resolutions = self.listResolutions()
        except Exception as e:
            logging.error(f"Error while fetching video resolutions: {e}")
            exit(1)

        questions = [
            List(
                "resolution",
                message="Select the resolution you want to download (width x height), use up and down arrow keys to navigate and press enter to select",
                choices=[f"{w}x{h}" for w, h in resolutions],
            ),
        ]

        answers = prompt(questions)
        if not answers:
            logging.error("No resolution selected, exiting")
            exit(1)

        self.resolution = answers["resolution"]
        self.width, self.height = map(int, self.resolution.split("x"))

        if self.height > 1080 and self.ae:
            toPrint = f"The selected resolution {self.resolution} is higher than 1080p, this will require an additional step of encoding the video for compatibility with After Effects"
            logging.warning(toPrint)
            print(toPrint)
        else:
            toPrint = f"Selected resolution: {self.resolution}"
            logging.info(toPrint)
            print(toPrint)

        self.downloadVideo()

    def listResolutions(self):
        options = {
            "listformats": False,
            "quiet": True,
            "no_warnings": True,
        }
        with YoutubeDL(options) as ydl:
            info_dict = ydl.extract_info(self.link, download=False)
            formats = info_dict.get("formats", [])
            resolutions = [
                (f.get("width"), f.get("height"))
                for f in formats
                if f.get("width") and f.get("height") and f.get("height") >= 240
            ]

            return sorted(set(resolutions), key=lambda x: x[1], reverse=True)

    def downloadVideo(self):
        options = self.getOptions()
        try:
            with YoutubeDL(options) as ydl:
                ydl.download([self.link])
        except Exception as e:
            logging.error(f"Failed to download video: {e}")
            raise

    def getOptions(self):
        if self.height > 1080 and self.ae:
            return {
                "format": f"bestvideo[height<={self.height}][width<={self.width}]+bestaudio[ext=m4a]/best[ext=mp4]",
                "outtmpl": os.path.splitext(self.output)[0],
                "ffmpeg_location": os.path.dirname(self.ffmpegPath),
                "quiet": True,
                "noplaylist": True,
                "no_warnings": True,
                "postprocessors": [
                    {
                        "key": "FFmpegVideoConvertor",
                        "preferedformat": "mp4",
                        # self.customEncoder if self.customEncoder else matchEncoder(self.encodeMethod),
                    }
                ],
                "nocookies": True,
            }
        else:
            return {
                "format": f"bestvideo[height<={self.height}][width<={self.width}][vcodec^=avc1]+bestaudio[ext=m4a]/best[ext=mp4]",
                "outtmpl": self.output,
                "ffmpeg_location": os.path.dirname(self.ffmpegPath),
                "quiet": True,
                "noplaylist": True,
                "no_warnings": True,
                "nocookies": True,
            }
