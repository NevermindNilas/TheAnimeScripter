import logging
import os
from urllib.parse import urlparse

import src.constants as cs
from src.io.inputOutputHandler import SEQUENCE_EXTENSIONS


class InputNormalizationError(ValueError):
    pass


def processUrlInput(args, outputPath, processingEnabled):
    from src.infra.getFFMPEG import ensureFFMPEG
    from src.infra.logAndPrint import logAndPrint
    from src.io.inputOutputHandler import generateOutputName, titleFileName

    result = urlparse(args.input)

    if result.netloc.lower() not in ["www.youtube.com", "youtube.com", "youtu.be"]:
        raise InputNormalizationError(
            "URL is invalid or not a YouTube URL, please check the URL and try again"
        )

    logging.info("URL is valid and will be used for processing")

    # yt-dlp is the one thing left that runs FFmpeg (stream merge, Adobe mp4
    # conversion), so it is installed on the first URL input, before src.ytdlp
    # reads cs.FFMPEGPATH at import.
    ensureFFMPEG()
    from src.ytdlp import VideoDownloader

    downloader = VideoDownloader(args.input, args.encode_method)
    title = downloader.title

    if args.output is None:
        outputFolder = os.path.join(outputPath, "output")
        os.makedirs(os.path.join(outputFolder), exist_ok=True)
        args.output = os.path.join(
            outputFolder, generateOutputName(args, args.input, title)
        )
    elif os.path.isdir(args.output):
        outputFolder = args.output
        os.makedirs(os.path.join(outputFolder), exist_ok=True)
        if not processingEnabled:
            # The rename below needs a file in that folder, not the folder.
            args.output = os.path.join(
                outputFolder, generateOutputName(args, args.input, title)
            )
    else:
        outputFolder = os.path.dirname(args.output)
        os.makedirs(os.path.join(outputFolder), exist_ok=True)

    # The download is the processing run's input, so it is named after the
    # video alone (always .mp4, yt-dlp's merge format): with the run's
    # suffixes it was the same path as the output it is processed into.
    stem = titleFileName(title) or "TAS-YTDLP"
    tempOutput = os.path.join(outputFolder, f"{stem}.mp4")
    if processingEnabled and os.path.abspath(tempOutput) == os.path.abspath(
        args.output
    ):
        # Stages that add no suffix (stabilize, motion blur, ...).
        tempOutput = os.path.join(outputFolder, f"{stem}-YTDLP.mp4")

    downloader.download(tempOutput)
    logAndPrint(
        f"Video downloaded successfully to {tempOutput}",
        "green",
    )

    if not processingEnabled:
        if tempOutput != args.output:
            os.rename(tempOutput, args.output)
            logging.info(f"Renamed output to: {args.output}")
        return False

    args.input = str(tempOutput)
    logging.info(f"New input path: {args.input}")
    return True


def normalizeImageInput(args, processingEnabled):
    if "%" in args.input:
        logging.info(f"Image sequence pattern detected: {args.input}")
        args.input = os.path.abspath(args.input)
        cs.AUDIO = False
        return

    if args.input.lower().endswith(".png"):
        args.input = os.path.abspath(args.input)
        cs.AUDIO = False
        args.single_image_input = True
        args.png_passthrough = True
        logging.info("Single PNG input detected, enabling PNG passthrough mode")

        if processingEnabled and args.encode_method not in SEQUENCE_EXTENSIONS:
            logging.info(
                "Single PNG with processing detected; forcing --encode_method png for valid image output"
            )
            args.encode_method = "png"
        return

    raise InputNormalizationError(
        "Single image input is not supported for this format. For image sequences, use a pattern like 'frames_%05d.png' or provide a folder containing PNG files."
    )


def normalizeInputArgs(args, outputPath, processingEnabled):
    if not args.input:
        raise InputNormalizationError("No input specified")

    if args.input.startswith(("http", "www")):
        return processUrlInput(args, outputPath, processingEnabled)

    if args.input.lower().endswith((".png", ".jpg", ".jpeg")):
        normalizeImageInput(args, processingEnabled)
        return True

    if args.input.lower().endswith(".gif"):
        if args.encode_method != "gif":
            logging.error(
                "GIF input detected but encoding method is not set to GIF, defaulting to GIF encoding"
            )
            args.encode_method = "gif"
        return True

    try:
        args.input = os.path.abspath(args.input)
    except Exception as e:
        raise InputNormalizationError("Error processing input") from e

    return True
