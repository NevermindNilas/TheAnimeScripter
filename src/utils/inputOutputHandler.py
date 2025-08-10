import os
import logging
import random
from src.utils.logAndPrint import logAndPrint

EXTENSIONS = [".mp4", ".mkv", ".webm", ".avi", ".mov", ".gif"]


def generateOutputPath(video, output, outputPath, args):
    """Generates appropriate output path based on input parameters."""
    # Handle GIF files
    if video.endswith(".gif"):
        if output is None or os.path.isdir(output):
            return os.path.join(output or outputPath, generateOutputName(args, video))
        return output

    # Handle output not specified
    if output is None:
        if args.encode_method == "png":
            return handlePngOutput(args, video, outputPath)
        return os.path.join(outputPath, generateOutputName(args, video))

    # Handle direct file output
    if output.endswith(tuple(EXTENSIONS)):
        return output

    # Handle directory output
    outputDir = output
    if not outputDir.endswith(os.path.sep):
        outputDir += os.path.sep

    if os.path.isdir(outputDir):
        if args.encode_method == "png":
            return handlePngOutput(args, video, outputDir)
        return os.path.join(outputDir, generateOutputName(args, video))

    raise FileNotFoundError(f"Output directory {output} does not exist")


def handlePngOutput(args, video, outputPath):
    """Handles PNG output for video processing."""
    outputName = generateOutputName(args, video)
    outputFolder = os.path.join(outputPath, outputName)
    os.makedirs(outputFolder, exist_ok=True)

    return os.path.join(outputFolder, "frames_%05d.png")


def validateEncoder(video, encodeMethod, customEncoder):
    """Validates and potentially adjusts the encoder method based on file type."""
    if (
        video.endswith(".webm")
        and not customEncoder
        and encodeMethod not in ["vp9", "qsv_vp9", "av1"]
    ):
        logAndPrint(
            f"Video {video} is a Webm file, encode method was not set to ['vp9', 'qsv_vp9', 'av1'] and `--custom_encoder` is None, defaulting to 'vp9'.",
            colorFunc="yellow",
        )
        return "vp9"
    return encodeMethod


def processInputOutputPaths(args, outputPath):
    """Processes input and output paths for video processing."""
    videos = os.path.abspath(args.input)
    output = args.output
    encodeMethod = args.encode_method
    customEncoder = args.custom_encoder

    os.makedirs(outputPath, exist_ok=True)

    if output and not output.endswith(tuple(EXTENSIONS)):
        if not output.endswith(os.path.sep):
            output += os.path.sep
        os.makedirs(output, exist_ok=True)

    videoFiles = getVideoFiles(videos)

    result = {}
    for index, video in enumerate(videoFiles, 1):
        if not os.path.exists(video):
            raise FileNotFoundError(f"File {video} does not exist")

        result[index] = {
            "videoPath": video,
            "outputPath": generateOutputPath(video, output, outputPath, args),
            "encodeMethod": validateEncoder(video, encodeMethod, customEncoder),
            "customEncoder": customEncoder,
        }

    return result


def getVideoFiles(videosInput):
    """Extract list of video files from input specification."""
    if os.path.isdir(videosInput):
        return [
            os.path.join(videosInput, f)
            for f in os.listdir(videosInput)
            if os.path.splitext(f)[1].lower() in EXTENSIONS
        ]
    elif os.path.isfile(videosInput):
        if videosInput.endswith(".txt"):
            with open(videosInput, "r") as file:
                return [line.strip().strip('"') for line in file.readlines()]
        return [videosInput]
    else:
        return videosInput.split(";")


def generateOutputName(args, videoInput):
    """Generates output filename based on input and processing arguments."""
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
        # Handle URL input
        if "https://" in videoInput or "http://" in videoInput:
            return f"TAS-YTDLP-{random.randint(0, 1000)}.mp4"

        # Start with base name
        baseName = (
            os.path.splitext(os.path.basename(videoInput))[0] if videoInput else "TAS"
        )

        # Add processing indicators
        suffixes = [suffix for suffix in argMap.values() if suffix]

        # Add random number to prevent overwrites
        suffixes.append(f"-{random.randint(0, 1000)}")

        # Determine extension
        if (
            getattr(args, "segment", False)
            or getattr(args, "encode_method", "") == "prores"
        ):
            extension = ".mov"
        elif args.encode_method == "png":
            extension = ""
        elif videoInput:
            extension = os.path.splitext(videoInput)[1]
        else:
            extension = ".mp4"

        return baseName + "".join(suffixes) + extension

    except AttributeError as e:
        logging.error(f"AttributeError in generateOutputName: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error in generateOutputName: {e}")
        raise
