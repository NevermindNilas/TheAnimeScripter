import os
import logging

from inquirer import List, prompt
from yt_dlp import YoutubeDL
from src.constants import ADOBE, FFMPEGPATH


class VideoDownloader:
    def __init__(
        self,
        video_link,
        output,
        encodeMethod,
        customEncoder,
    ):
        self.link = video_link
        self.output = output
        self.encodeMethod = encodeMethod
        self.customEncoder = customEncoder

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

        if self.height > 1080 and ADOBE:
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
        self.downloaded_file = None

        def hook(d):
            if d.get("status") == "finished":
                self.downloaded_file = d.get("filename")

        options = self.getOptions()
        # Add progress_hooks to options
        options["progress_hooks"] = [hook]
        try:
            with YoutubeDL(options) as ydl:
                ydl.download([self.link])
        except Exception as e:
            logging.error(f"Failed to download video: {e}")
            raise (f"Error downloading video: {e}")

    def getOptions(self):
        if self.height > 1080 and ADOBE:
            # Always ensure .mp4 extension is present
            outtmpl = self.output
            return {
                "format": f"bestvideo[height<={self.height}]+bestaudio/best[height<={self.height}]/best",
                "outtmpl": outtmpl,
                "ffmpeg_location": os.path.dirname(FFMPEGPATH),
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
                "merge_output_format": "mp4",
                "nocookies": True,
            }
        else:
            outtmpl = self.output
            return {
                "format": f"bestvideo[height<={self.height}]+bestaudio/best[height<={self.height}]/best",
                "outtmpl": outtmpl,
                "ffmpeg_location": os.path.dirname(FFMPEGPATH),
                "quiet": True,
                "noplaylist": True,
                "no_warnings": True,
                "merge_output_format": "mp4",
                "nocookies": True,
            }
