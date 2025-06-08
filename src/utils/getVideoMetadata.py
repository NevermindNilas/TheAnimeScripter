import logging
import textwrap
import json
import os
import subprocess
import src.constants as cs


def saveMetadata(metadata, videoDataDump=None):
    metadataPath = os.path.join(cs.MAINPATH, "metadata.json")
    with open(metadataPath, "w") as jsonFile:
        data = {
            "metadata": metadata,
            "FFPROBE DUMP": videoDataDump if videoDataDump else None,
        }
        json.dump(data, jsonFile, indent=4)

    cs.METADATAPATH = metadataPath


def getVideoMetadata(inputPath, inPoint, outPoint):
    """
    Get metadata from a video file using ffprobe.

    Parameters:
    inputPath (str): The path to the video file
    inPoint (float): Start time of clip
    outPoint (float): End time of clip
    ffprobePath (str): Path to ffprobe executable

    Returns:
    tuple: (width, height, fps, totalFramesToProcess, hasAudio)
    """
    try:
        if not os.path.exists(cs.FFMPEGPATH):
            logging.error("ffprobe not found")
            raise FileNotFoundError("ffprobe path not found")
        if not os.path.exists(inputPath):
            logging.error("Video file not found")
            raise FileNotFoundError("Video file not found")

        cmd = [
            cs.FFPROBEPATH,
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            "-count_packets",
            inputPath,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )

        if not result.stdout:
            raise Exception("No output received from ffprobe")

        probeData = json.loads(result.stdout)

        # Get video stream
        videoStream = next(
            stream for stream in probeData["streams"] if stream["codec_type"] == "video"
        )

        # Check for audio streams
        # If the condition is true, then we can check for audio streams, otherwise we can skip this step
        if cs.AUDIO:
            hasAudio = any(
                stream["codec_type"] == "audio" for stream in probeData["streams"]
            )
            # Is there any audio stream in the video to begin with?
            # If yes, then set the global variable accordingly
            cs.AUDIO = hasAudio
        else:
            hasAudio = False

        # Extract metadata
        width = int(videoStream["width"])
        height = int(videoStream["height"])
        fpsParts = videoStream["r_frame_rate"].split("/")
        fps = float(fpsParts[0]) / float(fpsParts[1])
        duration = float(probeData["format"]["duration"])
        totalFrames = int(videoStream.get("nb_read_packets", 0))
        colorFormat = videoStream.get("pix_fmt", "unknown")
        pixelFormat = videoStream.get("color_space", "unknown")
        colorSpace = videoStream.get("color_primaries", "unknown")
        ColorTRT = videoStream.get("color_transfer", "unknown")
        ColorRange = videoStream.get("color_range", "unknown")

        if outPoint != 0:
            totalFramesToProcess = int((outPoint - inPoint) * fps)
        else:
            totalFramesToProcess = totalFrames

        metadata = {
            "Width": width,
            "Height": height,
            "AspectRatio": round(width / height, 2),
            "FPS": round(fps, 2),
            "Codec": videoStream["codec_name"],
            "ColorRange": ColorRange,
            "ColorFormat": colorFormat,
            "ColorSpace": colorSpace,
            "ColorTRT": ColorTRT,
            "PixelFormat": pixelFormat,
            "Duration": duration,
            "Inpoint": inPoint,
            "Outpoint": outPoint,
            "NumberOfTotalFrames": totalFrames,
            "TotalFramesToBeProcessed": totalFramesToProcess,
            "HasAudio": hasAudio,
        }

        logging.info(
            textwrap.dedent(f"""
        ============== Video Metadata ==============
        Width: {width}
        Height: {height}
        AspectRatio: {metadata["AspectRatio"]}
        FPS: {round(fps, 2)}
        Codec: {metadata["Codec"]}
        ColorRange: {ColorRange}
        ColorFormat: {colorFormat},
        ColorSpace: {colorSpace},
        ColorTRTR: {ColorTRT},
        Duration: {duration} seconds
        Inpoint: {inPoint}
        Outpoint: {outPoint}
        Number of total frames: {totalFrames}
        Total frames to be processed: {totalFramesToProcess}
        Has Audio: {hasAudio}""")
        )

        saveMetadata(metadata, videoStream)
        return metadata

    except Exception as e:
        logging.error(f"Error getting metadata with ffprobe: {e}")
        raise
