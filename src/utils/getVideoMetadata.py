import celux
import logging
import textwrap
import json
import os

def saveMetadata(metadata, mainPath):
    with open(os.path.join(mainPath, "metadata.json"), "w") as jsonFile:
        json.dump(metadata, jsonFile, indent=4)

def getVideoMetadata(inputPath: str, inPoint: float, outPoint: float, mainPath: str):
    """
    Get metadata from a video file.

    Parameters:
    inputPath (str): The path to the video file.
    inPoint (float): The start time of the video clip.
    outPoint (float): The end time of the video clip.

    Returns:
    tuple: A tuple containing the width, height, fps, total frames to be processed, and a boolean indicating if the video has audio.
    """
    metadata = {}
    try:
        video = celux.VideoReader(inputPath, device="cpu")
        properties = video.get_properties()
    except ValueError as ve:
        if "Unknown pixel format for bit depth inference:" in str(ve):
            logging.info(f"ValueError encountered: {ve}. Switching to PyMediaInfo.")
            return videoMetadataPyMedia(inputPath, inPoint, outPoint, mainPath)
        else:
            raise
    except Exception as e:
        logging.info(f"Failed to get Metadata with Celux, switching to PyMediaInfo {e}")
        return videoMetadataPyMedia(inputPath, inPoint, outPoint, mainPath)
    
    width = properties["width"]
    height = properties["height"]
    fps = properties["fps"]
    nFrames = properties["total_frames"]
    hasAudio = properties["has_audio"]

    duration = round(nFrames / fps, 2) if fps else 0

    if outPoint != 0:
        totalFramesToBeProcessed = int((outPoint - inPoint) * fps)
    else:
        totalFramesToBeProcessed = nFrames

    metadata = {
        "Width": width,
        "Height": height,
        "AspectRatio": properties["aspect_ratio"],
        "MinFPS": properties["min_fps"],
        "MaxFPS": properties["max_fps"],
        "Codec": properties["codec"],
        "VideoLength": duration,
        "Inpoint": inPoint,
        "Outpoint": outPoint,
        "NumberOfTotalFrames": nFrames,
        "TotalFramesToBeProcessed": totalFramesToBeProcessed,
        "HasAudio": hasAudio
    }

    try:
        del video
    except Exception as e:
        logging.error(f"Error while deleting video object: {e}")

    logging.info(
        textwrap.dedent(f"""
    ============== Video Metadata ==============
    Width: {width}
    Height: {height}
    AspectRatio: {properties["aspect_ratio"]:0.2f}
    MinFPS: {properties["min_fps"]:0.2f}
    MaxFPS: {properties["max_fps"]:0.2f}
    Codec: {properties["codec"]}
    Video Length: {duration} seconds
    Inpoint: {inPoint}
    Outpoint: {outPoint}
    Number of total frames: {nFrames}
    Total frames to be processed: {totalFramesToBeProcessed}
    Has Audio: {hasAudio}""")
    )

    saveMetadata(metadata, mainPath)
    return width, height, fps, totalFramesToBeProcessed, hasAudio

def videoMetadataPyMedia(inputPath, inPoint, outPoint, mainPath):
    """
    Get metadata from a video file.

    Parameters:
    inputPath (str): The path to the video file.
    inPoint (float): The start time of the video clip.
    outPoint (float): The end time of the video clip.

    Returns:
    tuple: A tuple containing the width, height, fps, total frames to be processed, pixel format of the video, and a boolean indicating if the video has audio.
    """

    # Conditional import cuz this takes too long to import
    from pymediainfo import MediaInfo
    mediaInfo = MediaInfo.parse(inputPath)
    videoTrack = next(
        (track for track in mediaInfo.tracks if track.track_type == "Video"), None
    )
    audioTrack = next(
        (track for track in mediaInfo.tracks if track.track_type == "Audio"), None
    )

    if videoTrack is None:
        logging.error("No video stream found in the file.")
        exit(1)

    width = videoTrack.width
    height = videoTrack.height
    fps = float(videoTrack.frame_rate)
    nframes = int(videoTrack.frame_count)
    codec = videoTrack.codec
    pixFmt = videoTrack.pixel_format

    duration = round(nframes / fps, 2) if fps else 0
    inOutDuration = round((outPoint - inPoint) / fps, 2) if fps else 0

    # Calculate total frames from inPoint to outPoint
    if outPoint != 0:
        totalFramesToBeProcessed = int((outPoint - inPoint) * fps)
    else:
        totalFramesToBeProcessed = nframes

    hasAudio = audioTrack is not None

    metadata = {
        "Width": width,
        "Height": height,
        "AspectRatio": round(width/height, 2),
        "FPS": round(fps, 2),
        "NumberOfTotalFrames": nframes,
        "Codec": codec,
        "Duration": duration,
        "InOutDuration": inOutDuration,
        "TotalFramesToBeProcessed": totalFramesToBeProcessed,
        "PixelFormat": pixFmt,
        "HasAudio": hasAudio
    }

    logging.info(
        textwrap.dedent(f"""
    ============== Video Metadata ==============
    Width: {width}
    Height: {height}
    AspectRatio: {round(width/height, 2)}
    FPS: {round(fps, 2)}
    Number of total frames: {nframes}
    Codec: {codec}
    Duration: {duration} seconds
    In-Out Duration: {inOutDuration} seconds
    Total frames to be processed: {totalFramesToBeProcessed}
    Pixel Format: {pixFmt}
    Has Audio: {hasAudio}""")
    )

    saveMetadata(metadata, mainPath)
    return width, height, fps, totalFramesToBeProcessed, hasAudio