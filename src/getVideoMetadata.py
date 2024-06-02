import cv2
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
    tuple: A tuple containing the width, height, and fps of the video.
    """
    try:
        cap = cv2.VideoCapture(inputPath)
    except Exception as e:
        logging.error(f"Error opening video file: {e}")
        exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width == 0 or height == 0:
        logging.error(
            "Width or height cannot be zero. Please check the input video file and make sure that it was put in quotation marks."
        )
        exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

    codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

    duration = round(nframes / fps, 2) if fps else 0
    inOutDuration = round((outPoint - inPoint) / fps, 2) if fps else 0

    # Calculate total frames from inPoint to outPoint
    totalFramesToBeProcessed = int((outPoint - inPoint) * fps)

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
    Total frames to be processed: {totalFramesToBeProcessed}""")
    )

    cap.release()

    return width, height, fps
