import logging
import textwrap

import celux


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
    """

    video = celux.VideoReader(inputPath)
    videoMetadata = video.get_properties()

    print(videoMetadata)

    inOutDuration = round((outPoint - inPoint), 2)

    if inPoint != 0 or outPoint != 0:
        totalFramesToBeProcessed = int(inOutDuration * videoMetadata["fps"])
    else:
        totalFramesToBeProcessed = videoMetadata["total_frames"]

    logging.info(
        textwrap.dedent(f"""
    ============== Video Metadata ==============
    Width: {videoMetadata['width']}
    Height: {videoMetadata['height']}
    AspectRatio: {round(videoMetadata['width'] / videoMetadata['height'], 2)}
    FPS: {round(videoMetadata['fps'], 2)}
    Number of total frames: {videoMetadata['total_frames']}
    Duration: {videoMetadata['duration']} seconds
    In-Out Duration: {inOutDuration} seconds
    Total frames to be processed: {totalFramesToBeProcessed}
    Pixel Format: {videoMetadata['pixel_format']}
    Has Audio: {videoMetadata['has_audio']}""")
    )

    return (
        videoMetadata["width"],
        videoMetadata["height"],
        videoMetadata["fps"],
        totalFramesToBeProcessed,
        videoMetadata["has_audio"],
    )
