import logging
import textwrap
import json
import os
import subprocess


def saveMetadata(metadata, mainPath):
    with open(os.path.join(mainPath, "metadata.json"), "w") as jsonFile:
        json.dump(metadata, jsonFile, indent=4)


def getVideoMetadata(inputPath, inPoint, outPoint, mainPath, ffprobePath):
    """
    Get metadata from a video file using ffprobe.

    Parameters:
    inputPath (str): The path to the video file
    inPoint (float): Start time of clip
    outPoint (float): End time of clip
    mainPath (str): Path to save metadata
    ffprobePath (str): Path to ffprobe executable

    Returns:
    tuple: (width, height, fps, totalFramesToProcess, hasAudio)
    """
    try:
        cmd = [
            ffprobePath,
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            "-count_packets",
            inputPath,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        probeData = json.loads(result.stdout)

        # Get video stream
        videoStream = next(
            stream for stream in probeData["streams"] if stream["codec_type"] == "video"
        )

        # Check for audio streams
        hasAudio = any(
            stream["codec_type"] == "audio" for stream in probeData["streams"]
        )

        # Extract metadata
        width = int(videoStream["width"])
        height = int(videoStream["height"])
        fpsParts = videoStream["r_frame_rate"].split("/")
        fps = float(fpsParts[0]) / float(fpsParts[1])
        duration = float(probeData["format"]["duration"])
        totalFrames = int(videoStream.get("nb_read_packets", 0))

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
        Duration: {duration} seconds
        Inpoint: {inPoint}
        Outpoint: {outPoint}
        Number of total frames: {totalFrames}
        Total frames to be processed: {totalFramesToProcess}
        Has Audio: {hasAudio}""")
        )

        saveMetadata(metadata, mainPath)
        return metadata

    except Exception as e:
        logging.error(f"Error getting metadata with ffprobe: {e}")
        raise
