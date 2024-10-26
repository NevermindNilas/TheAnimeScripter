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

    Returns:
    tuple: A tuple containing the width, height, fps, total frames to be processed, and a boolean indicating if the video has audio.
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
    nFrames = int(videoTrack.frame_count)
    codec = videoTrack.format
    chromaSubsampling = videoTrack.chroma_subsampling
    bitDepth = videoTrack.bit_depth
    colorSpace = videoTrack.color_space

    supportedPixFMTs = ["yuv420p8le", "yuv420p10le", "yuv422p8le"]
    if colorSpace == "RGB":
        pixFmt = f"rgb{bitDepth}le" if bitDepth else "rgb8le"
    elif colorSpace == "YUV":
        if bitDepth == "8":
            pixFmt = f"yuv{chromaSubsampling.replace(':', '')}p"
        else:
            pixFmt = (
                f"yuv{chromaSubsampling.replace(':', '')}p{bitDepth}le"
                if bitDepth
                else f"yuv{chromaSubsampling.replace(':', '')}p8le"
            )
    else:
        pixFmt = "unknown"

    if pixFmt not in supportedPixFMTs:
        logging.warning(
            f"Unsupported pixel format. The pixel format {pixFmt} is not officially supported, falling back to yuv420p8le."
        )

    duration = round(nFrames / fps, 2) if fps else 0
    inOutDuration = round((outPoint - inPoint) / fps, 2) if fps else 0

    # Calculate total frames from inPoint to outPoint
    if outPoint != 0:
        totalFramesToBeProcessed = int((outPoint - inPoint) * fps)
    else:
        totalFramesToBeProcessed = nFrames

    hasAudio = audioTrack is not None

    logging.info(
        textwrap.dedent(f"""
    ============== Video Metadata ==============
    Width: {width}
    Height: {height}
    AspectRatio: {round(width/height, 2)}
    FPS: {round(fps, 2)}
    Codec: {codec}
    Pixel Format: {pixFmt}
    Duration: {duration} seconds
    In-Out Duration: {inOutDuration} seconds
    Number of total frames: {nFrames}
    Total frames to be processed: {totalFramesToBeProcessed}
    Has Audio: {hasAudio}""")
    )

    return width, height, fps, totalFramesToBeProcessed, hasAudio, pixFmt
