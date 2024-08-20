import subprocess
import json
import logging
import textwrap


def getVideoMetadata(inputPath, inPoint, outPoint, ffprobePath):
    """
    Get metadata from a video file using ffprobe.

    Parameters:
    inputPath (str): The path to the video file.
    inPoint (float): The start time of the video clip.
    outPoint (float): The end time of the video clip.

    Returns:
    tuple: A tuple containing the width, height, fps, and total frames to be processed.
    """

    def run_ffprobe(inputPath):
        cmd = [
            ffprobePath,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,duration,nb_frames,codec_name",
            "-of",
            "json",
            inputPath,
        ]
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        return json.loads(result.stdout)

    try:
        metadata = run_ffprobe(inputPath)
        stream = metadata["streams"][0]
    except Exception as e:
        logging.error(f"Error getting video metadata: {e}")
        exit(1)

    width = int(stream["width"])
    height = int(stream["height"])
    fps = eval(stream["r_frame_rate"])  # Convert '30/1' to 30.0
    nframes = int(stream.get("nb_frames", 0))
    codec = stream["codec_name"].upper()
    duration = float(stream["duration"])

    inOutDuration = round((outPoint - inPoint), 2)
    totalFramesToBeProcessed = (
        round((outPoint - inPoint) * fps) if outPoint != 0 else nframes
    )

    logging.info(
        textwrap.dedent(f"""
    ============== Video Metadata ==============
    Width: {width}
    Height: {height}
    AspectRatio: {round(width/height, 2)}
    FPS: {round(fps, 2)}
    Number of total frames: {nframes}
    Codec: {codec}
    Total Duration of Video: {duration} seconds
    In-Out Duration: {inOutDuration} seconds
    Total frames to be processed: {totalFramesToBeProcessed}""")
    )

    return width, height, fps, totalFramesToBeProcessed
