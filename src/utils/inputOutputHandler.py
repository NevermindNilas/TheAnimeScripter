import os
import logging
import random
from src.utils.logAndPrint import logAndPrint

EXTENSIONS = [".mp4", ".mkv", ".webm", ".avi", ".mov", ".gif"]


def generateOutputName(args, videoInput):
    """Generates output filename based on input and processing arguments."""
    if any(proto in str(videoInput) for proto in ["https://", "http://"]):
        return f"TAS-YTDLP-{random.randint(0, 1000)}.mp4"

    baseName = os.path.splitext(os.path.basename(videoInput))[0] if videoInput else "TAS"

    features = [
        ("resize", "Resize", "resize_factor"),
        ("dedup", "Dedup", "dedup_sens"),
        ("interpolate", "Int", "interpolate_factor"),
        ("upscale", "Up", "upscale_factor"),
        ("sharpen", "Sh", "sharpen_sens"),
        ("restore", "Restore", "restore_method"),
        ("segment", "Segment", None),
        ("depth", "Depth", None),
        ("ytdlp", "YTDLP", None),
    ]

    suffixes = []
    for arg, label, val_attr in features:
        if getattr(args, arg, False):
            val = getattr(args, val_attr, "") if val_attr else ""
            suffixes.append(f"-{label}{val}")

    suffixes.append(f"-{random.randint(0, 1000)}")

    if getattr(args, "segment", False) or getattr(args, "encode_method", "") == "prores":
        extension = ".mov"
    elif getattr(args, "encode_method", "") == "png":
        extension = ""
    elif videoInput:
        extension = os.path.splitext(videoInput)[1]
    else:
        extension = ".mp4"

    return f"{baseName}{''.join(suffixes)}{extension}"


def generateOutputPath(video, output, defaultOutputPath, args):
    """Generates appropriate output path based on input parameters."""
    if output and output.endswith(tuple(EXTENSIONS)):
        return output

    baseDir = output if output and os.path.isdir(output) else defaultOutputPath

    if getattr(args, "encode_method", "") == "png":
        outputName = generateOutputName(args, video)
        outputFolder = os.path.join(baseDir, outputName)
        os.makedirs(outputFolder, exist_ok=True)
        return os.path.join(outputFolder, "frames_%05d.png")

    return os.path.join(baseDir, generateOutputName(args, video))


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


def getVideoFiles(videosInput):
    """Extract list of video files from input specification."""
    # Handle semicolon separated paths
    if ";" in str(videosInput):
        paths = [v.strip() for v in str(videosInput).split(";") if v.strip()]
        all_files = []
        for p in paths:
            all_files.extend(getVideoFiles(p))
        return all_files

    # Handle URL input
    if any(proto in str(videosInput) for proto in ["https://", "http://"]):
        return [videosInput]

    # Handle directory or file
    absPath = os.path.abspath(videosInput)
    if os.path.isdir(absPath):
        return [
            os.path.join(absPath, f)
            for f in os.listdir(absPath)
            if os.path.splitext(f)[1].lower() in EXTENSIONS
        ]

    if os.path.isfile(absPath):
        if absPath.endswith(".txt"):
            with open(absPath, "r") as f:
                return [
                    os.path.abspath(line.strip().strip('"'))
                    for line in f
                    if line.strip()
                ]
        return [absPath]

    # Fallback
    return [absPath]


def processInputOutputPaths(args, defaultOutputPath):
    """Processes input and output paths for video processing."""
    os.makedirs(defaultOutputPath, exist_ok=True)

    output = args.output
    if output:
        output = os.path.abspath(output)
        if not output.endswith(tuple(EXTENSIONS)):
            os.makedirs(output, exist_ok=True)
        else:
            parent_dir = os.path.dirname(output)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)

    videoFiles = getVideoFiles(args.input)

    results = {}
    for index, video in enumerate(videoFiles, 1):
        if not any(proto in str(video) for proto in ["https://", "http://"]):
            if not os.path.exists(video):
                raise FileNotFoundError(f"File {video} does not exist")

        results[index] = {
            "videoPath": video,
            "outputPath": generateOutputPath(video, output, defaultOutputPath, args),
            "encodeMethod": validateEncoder(
                video, args.encode_method, args.custom_encoder
            ),
            "customEncoder": args.custom_encoder,
        }

    return results
