import os
import logging
import random

from src.utils.coloredPrints import yellow

EXTENSIONS = [".mp4", ".mkv", ".webm", ".avi", ".mov", ".gif"]


def genOutputHandler(video, output, outputPath, args):
    if video.endswith(".gif"):
        if output is None or os.path.isdir(output):
            return os.path.join(output or outputPath, outputNameGenerator(args, video))
    else:
        if output is None:
            return os.path.join(outputPath, outputNameGenerator(args, video))
        elif output.endswith(tuple(EXTENSIONS)):
            return output
        elif not output.endswith("\\"):
            tempOutput = output + "\\"
            if os.path.isdir(tempOutput):
                return os.path.join(tempOutput, outputNameGenerator(args, video))
        elif os.path.isdir(output):
            return os.path.join(output, outputNameGenerator(args, video))
        else:
            raise FileNotFoundError(f"File {output} does not exist")
    return output


def encoderChecker(video, encodeMethod, customEncoder):
    if (
        video.endswith(".webm")
        and not customEncoder
        and encodeMethod not in ["vp9", "qsv_vp9", "av1"]
    ):
        print(
            yellow(
                f"Video {video} is a Webm file, encode method was not set to ['vp9', 'qsv_vp9', 'av1'] and `--custom_encoder` is None, defaulting to 'vp9'."
            )
        )
        return "vp9"
    return encodeMethod


def handleInputOutputs(args, isFrozen, outputPath):
    """
    Handles input and output paths for video processing.

    Args:
        outputPath (str): The path to the output directory
        isFrozen (bool): The frozen state of the application, Pyinstaller or Python
        args (argparse.Namespace): The arguments passed to the application

    Returns:
        dict: A dictionary containing videoPath, outputPath, encodeMethod, and customEncoder.
    """
    videos = os.path.abspath(args.input)
    output = args.output
    encodeMethod = args.encode_method
    customEncoder = args.custom_encoder

    os.makedirs(outputPath, exist_ok=True)

    if output and not output.endswith(tuple(EXTENSIONS)):
        if not output.endswith("\\"):
            output += "\\"
        os.makedirs(output, exist_ok=True)

    result = {}
    index = 1

    if os.path.isdir(videos):
        videoFiles = [
            os.path.join(videos, f)
            for f in os.listdir(videos)
            if os.path.splitext(f)[1] in EXTENSIONS
        ]
    elif os.path.isfile(videos) and not videos.endswith(".txt"):
        videoFiles = [videos]
    else:
        if videos.endswith(".txt"):
            with open(videos, "r") as file:
                videoFiles = [line.strip().strip('"') for line in file.readlines()]
        else:
            videoFiles = videos.split(";")

    for video in videoFiles:
        if not os.path.exists(video):
            raise FileNotFoundError(f"File {video} does not exist")
        result[index] = {
            "videoPath": video,
            "outputPath": genOutputHandler(video, output, outputPath, args),
            "encodeMethod": encoderChecker(video, encodeMethod, customEncoder),
            "customEncoder": customEncoder,
        }
        index += 1

    return result


def outputNameGenerator(args, videoInput):
    argMap = {
        "resize": f"-Resize{getattr(args, 'resize_factor', '')}"
        if getattr(args, "resize", False)
        else "",
        "dedup": f"-Dedup{getattr(args, 'dedup_sens', '')}"
        if getattr(args, "dedup", False)
        else "",
        "interpolate": f"-Int{getattr(args, 'interpolate_factor', '')}"
        if getattr(args, "interpolate", False)
        else "",
        "upscale": f"-Up{getattr(args, 'upscale_factor', '')}"
        if getattr(args, "upscale", False)
        else "",
        "sharpen": f"-Sh{getattr(args, 'sharpen_sens', '')}"
        if getattr(args, "sharpen", False)
        else "",
        "restore": f"-Restore{getattr(args, 'restore_method', '')}"
        if getattr(args, "restore", False)
        else "",
        "segment": "-Segment" if getattr(args, "segment", False) else "",
        "depth": "-Depth" if getattr(args, "depth", False) else "",
        "ytdlp": "-YTDLP" if getattr(args, "ytdlp", False) else "",
    }

    try:
        # Special case for input "anime"
        if videoInput == "anime":
            name = f"anime-{random.randint(0, 10000)}.mp4"
            logging.debug(f"Generated name for 'anime' input: {name}")
            return name

        # Check if videoInput is a URL
        if "https://" in videoInput or "http://" in videoInput:
            name = "TAS" + "-YTDLP" + f"-{random.randint(0, 1000)}" + ".mp4"
            logging.debug(f"Generated name for URL input: {name}")
            return name
        else:
            parts = [
                os.path.splitext(os.path.basename(videoInput))[0]
                if videoInput
                else "TAS"
            ]

        for arg, formatStr in argMap.items():
            if formatStr:
                parts.append(formatStr)

        parts.append(f"-{random.randint(0, 1000)}")

        if getattr(args, "segment", False) or getattr(args, "encode_method", "") in [
            "prores"
        ]:
            extension = ".mov"
        elif videoInput:
            extension = os.path.splitext(videoInput)[1]
        else:
            extension = ".mp4"

        outputName = "".join(parts) + extension
        logging.debug(f"Generated output name: {outputName}")

        return outputName

    except AttributeError as e:
        logging.error(f"AttributeError: {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise
