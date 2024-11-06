import celux
import logging
import textwrap


def getVideoMetadata(inputPath, inPoint, outPoint):
    """
    Get metadata from a video file.

    Parameters:
    inputPath (str): The path to the video file.
    inPoint (float): The start time of the video clip.
    outPoint (float): The end time of the video clip.

    Returns:
    tuple: A tuple containing the width, height, fps, total frames to be processed, and a boolean indicating if the video has audio.
    """
    video = celux.VideoReader(inputPath, device="cpu")
    properties = video.get_properties()

    width = properties["width"]
    height = properties["height"]
    fps = properties["fps"]
    nFrames = properties["total_frames"]
    codec = properties["codec"]
    hasAudio = properties["has_audio"]
    aspectRatio = properties["aspect_ratio"]

    duration = round(nFrames / fps, 2) if fps else 0

    # Calculate total frames from inPoint to outPoint
    if outPoint != 0:
        totalFramesToBeProcessed = int((outPoint - inPoint) * fps)
    else:
        totalFramesToBeProcessed = nFrames

    logging.info(
        textwrap.dedent(f"""
    ============== Video Metadata ==============
    Width: {width}
    Height: {height}
    AspectRatio: {aspectRatio:.2f}
    FPS: {round(fps, 2)}
    Codec: {codec}
    Video Lenght: {duration} seconds
    Inpoint: {inPoint}
    Outpoint: {outPoint}
    Number of total frames: {nFrames}
    Total frames to be processed: {totalFramesToBeProcessed}
    Has Audio: {hasAudio}""")
    )

    return width, height, fps, totalFramesToBeProcessed, hasAudio
