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
            command = [ffmpeg_path, '-i', temp_name]
            match self.encode_method:
                # I know that superfast isn't exactly the best preset for x264, but I fear that someone will try to convert a 4k 24 min video
                # On a i7 4770k and it will take 3 business days to finish
                case "x264":
                    command.extend(
                        ['-c:v', 'libx264', '-preset', 'superfast', '-crf', '14'])
                case "x264_animation":
                    command.extend(
                        ['-c:v', 'libx264', '-preset', 'superfast', '-tune', 'animation', '-crf', '14'])
                case "nvenc_h264":
                    command.extend(
                        ['-c:v', 'h264_nvenc', '-preset', 'p1', '-cq', '14'])
                case "nvenc_h265":
                    command.extend(
                        ['-c:v', 'hevc_nvenc', '-preset', 'p1', '-cq', '14'])
                case "qsv_h264":
                    command.extend(
                        ['-c:v', 'h264_qsv', '-preset', 'veryfast', '-global_quality', '14'])
                case "qsv_h265":
                    command.extend(
                        ['-c:v', 'hevc_qsv', '-preset', 'veryfast', '-global_quality', '14'])
                case "nvenc_av1":
                    command.extend(
                        ['-c:v', 'av1_nvenc', '-preset', 'p1', '-cq', '14'])
                case "av1":
                    command.extend(
                        ['-c:v', 'libsvtav1', '-preset', '8', '-crf', '14'])
                case "h264_amf":
                    command.extend(
                        ['-c:v', 'h264_amf', '-quality', 'speed', '-rc', 'cqp', '-qp', '14'])
                case "hevc_amf":
                    command.extend(
                        ['-c:v', 'hevc_amf', '-quality', 'speed', '-rc', 'cqp', '-qp', '14'])
                    
            command.append(self.output)
            subprocess.run(command)

            os.remove(temp_name)

            logging.info(
                f"Removing residual webm file: {temp_name}")

        logging.info(
            "Downloaded video from: " + self.link)

        logging.info(
            f"Saved video to: {self.output}")

        return
