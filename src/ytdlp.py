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
            videoInfo = self.getVideoInfo()
            resolutions = self.listResolutions(videoInfo)

            if not resolutions:
                logging.error("No valid resolutions found for this video")
                exit(1)

        except Exception as e:
            logging.error(f"Error while fetching video information: {e}")
            exit(1)

        title = videoInfo.get("title", "Unknown")
        duration = videoInfo.get("duration", 0)
        durationStr = f"{duration // 60}m {duration % 60}s" if duration else "Unknown"

        print(f"\n{'=' * 60}")
        try:
            print(f"Video: {title}")
        except UnicodeEncodeError:
            print(f"Video: {title.encode('ascii', 'replace').decode('ascii')}")
        print(f"Duration: {durationStr}")
        print(f"{'=' * 60}\n")

        choices = []
        for resData in resolutions:
            width, height, fps, vcodec, filesize = resData
            fpsStr = f"{fps}fps" if fps else ""
            codecStr = self._formatCodec(vcodec)
            sizeStr = self._formatFilesize(filesize)

            display = (
                f"{width:4}x{height:<4}  {fpsStr:>6}  {codecStr:>8}  {sizeStr:>10}"
            )
            choices.append(display)

        questions = [
            List(
                "resolution",
                message="Select download quality (Up/Down to navigate, Enter to select)",
                choices=choices,
            ),
        ]

        answers = prompt(questions)
        if not answers:
            logging.error("No resolution selected, exiting")
            exit(1)

        selected = answers["resolution"]
        self.width, self.height = map(int, selected.split()[0].split("x"))

        if self.height > 1080 and ADOBE:
            toPrint = f"(!) Resolution {self.width}x{self.height} >1080p requires re-encoding for After Effects compatibility"
            logging.warning(toPrint)
            print(f"\n{toPrint}\n")
        else:
            toPrint = f"(*) Selected: {self.width}x{self.height}"
            logging.info(toPrint)
            print(f"\n{toPrint}\n")

        self.downloadVideo()

    def _formatCodec(self, vcodec):
        if not vcodec or vcodec == "none":
            return ""
        if "avc" in vcodec.lower() or "h264" in vcodec.lower():
            return "h264"
        elif "vp9" in vcodec.lower():
            return "vp9"
        elif "av01" in vcodec.lower():
            return "av1"
        elif "hevc" in vcodec.lower() or "h265" in vcodec.lower():
            return "h265"
        return vcodec[:8]

    def _formatFilesize(self, size):
        if not size:
            return ""
        if size > 1024 * 1024 * 1024:
            return f"~{size / (1024**3):.1f}GB"
        elif size > 1024 * 1024:
            return f"~{size / (1024**2):.0f}MB"
        return f"~{size / 1024:.0f}KB"

    def getVideoInfo(self):
        options = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
        }
        with YoutubeDL(options) as ydl:
            return ydl.extract_info(self.link, download=False)

    def listResolutions(self, infoDict):
        formats = infoDict.get("formats", [])

        resolutionData = []
        for f in formats:
            width = f.get("width")
            height = f.get("height")
            fps = f.get("fps")
            vcodec = f.get("vcodec", "")
            filesize = f.get("filesize") or f.get("filesize_approx")

            if width and height and height >= 240 and vcodec and vcodec != "none":
                resolutionData.append((width, height, fps, vcodec, filesize))

        uniqueResolutions = {}
        for width, height, fps, vcodec, filesize in resolutionData:
            key = (width, height)
            if key not in uniqueResolutions:
                uniqueResolutions[key] = (width, height, fps, vcodec, filesize)
            else:
                existing = uniqueResolutions[key]
                if (fps or 0) > (existing[2] or 0) or (filesize or 0) > (
                    existing[4] or 0
                ):
                    uniqueResolutions[key] = (width, height, fps, vcodec, filesize)

        return sorted(uniqueResolutions.values(), key=lambda x: x[1], reverse=True)

    def downloadVideo(self):
        self.downloadedFile = None

        def hook(d):
            status = d.get("status")
            if status == "downloading":
                downloaded = d.get("downloaded_bytes", 0)
                total = d.get("total_bytes") or d.get("total_bytes_estimate", 0)
                speed = d.get("speed", 0)

                if total > 0:
                    percent = (downloaded / total) * 100
                    speedStr = f"{speed / (1024**2):.1f}MB/s" if speed else "..."
                    print(
                        f"\rDownloading: {percent:.1f}% ({speedStr})",
                        end="",
                        flush=True,
                    )

            elif status == "finished":
                self.downloadedFile = d.get("filename")
                print("\r(*) Download complete!           ")

        options = self.getOptions()
        options["progress_hooks"] = [hook]

        try:
            with YoutubeDL(options) as ydl:
                ydl.download([self.link])
        except Exception as e:
            logging.error(f"Failed to download video: {e}")
            raise Exception(f"Error downloading video: {e}")

    def getOptions(self):
        if self.height > 1080 and ADOBE:
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
