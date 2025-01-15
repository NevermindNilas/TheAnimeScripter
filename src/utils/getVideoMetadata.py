import logging
import textwrap
import json
import os
import subprocess


def saveMetadata(metadata, mainPath):
    with open(os.path.join(mainPath, "metadata.json"), "w") as jsonFile:
        json.dump(metadata, jsonFile, indent=4)


def getVideoMetadata(
    inputPath: str, inPoint: float, outPoint: float, mainPath: str, ffprobePath: str
):
    try:
        cmd = [
            ffprobePath,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            inputPath,
        ]
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        metadataJson = json.loads(result.stdout)

        if "streams" not in metadataJson or not metadataJson["streams"]:
            logging.error(f"No streams found: {result.stderr}")
            raise KeyError("streams")

        videoStream = next(
            (s for s in metadataJson["streams"] if s["codec_type"] == "video"), None
        )
        audioStream = next(
            (s for s in metadataJson["streams"] if s["codec_type"] == "audio"), None
        )

        if not videoStream:
            logging.error("No video stream found.")
            raise KeyError("videoStream")

        width = videoStream["width"]
        height = videoStream["height"]
        fps = eval(videoStream["r_frame_rate"])
        nFrames = int(videoStream.get("nb_frames", 0))
        duration = float(videoStream.get("duration", 0))
        if nFrames == 0 and fps != 0:
            nFrames = int(duration * fps)
        codec = videoStream["codec_name"]
        pixFmt = videoStream["pix_fmt"]
        hasAudio = audioStream is not None

        duration = round(nFrames / fps, 2) if fps else 0
        totalFramesToBeProcessed = (
            int((outPoint - inPoint) * fps) if outPoint != 0 else nFrames
        )

        metadata = {
            "Width": width,
            "Height": height,
            "AspectRatio": round(width / height, 2),
            "FPS": round(fps, 2),
            "NumberOfTotalFrames": nFrames,
            "Codec": codec,
            "Duration": duration,
            "Inpoint": inPoint,
            "Outpoint": outPoint,
            "TotalFramesToBeProcessed": totalFramesToBeProcessed,
            "PixelFormat": pixFmt,
            "HasAudio": hasAudio,
        }

        logging.info(
            textwrap.dedent(f"""
        ============== Video Metadata ==============
        Width: {width}
        Height: {height}
        AspectRatio: {round(width / height, 2)}
        FPS: {round(fps, 2)}
        Number of total frames: {nFrames}
        Codec: {codec}
        Duration: {duration} seconds
        Inpoint: {inPoint}
        Outpoint: {outPoint}
        Total frames to be processed: {totalFramesToBeProcessed}
        Pixel Format: {pixFmt}
        Has Audio: {hasAudio}""")
        )

        saveMetadata(metadata, mainPath)
        return width, height, fps, totalFramesToBeProcessed, hasAudio

    except Exception as e:
        logging.error(f"ffprobe failed: {e}")
        raise
