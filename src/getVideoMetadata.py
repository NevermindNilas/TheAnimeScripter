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
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width == 0 or height == 0:
        logging.error("Width or height cannot be zero. Please check the input video file and make sure that it was put in quotation marks.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

    codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

    duration = nframes / fps if fps else 0
    inOutDuration = (outPoint - inPoint) / fps if fps else 0

    logging.info(textwrap.dedent(f"""
    ============== Video Metadata ==============
    Width: {width}
    Height: {height}
    AspectRatio: {width/height}
    FPS: {fps}
    Number of frames: {nframes}
    Codec: {codec}
    Duration: {duration} seconds
    In-Out Duration: {inOutDuration} seconds"""))

    cap.release()

    return width, height, fps