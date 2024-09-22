from pymediainfo import MediaInfo
import logging
import textwrap


def getVideoMetadata(inputPath, inPoint, outPoint):
    """
    Get metadata from a video file.

    Parameters:
    inputPath (str): The path to the video file.
    inPoint (float): The start time of the video clip.
    outPoint (float): The end time of the video clip.
    audio (bool): A boolean indicating if the video has audio.

    Returns:
    tuple: A tuple containing the width, height, fps, total frames to be processed, pixel format of the video, and a boolean indicating if the video has audio.
    """
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

    return width, height, fps, totalFramesToBeProcessed, pixFmt, hasAudio
