import logging
import os
from urllib.parse import urlparse

import src.constants as cs
from src.io.inputOutputHandler import (
    OUTPUT_FILE_EXTENSIONS,
    SEQUENCE_EXTENSIONS,
    isSingleImageInput,
    isUrlInput,
)


class InputNormalizationError(ValueError):
    """The input cannot be used as given; the message is shown to the user."""


def processUrlInput(args, outputPath, processingEnabled):
    from src.infra.getFFMPEG import ensureFFMPEG
    from src.infra.logAndPrint import logAndPrint
    from src.io.inputOutputHandler import generateOutputName, titleFileName

    # isUrlInput also takes a scheme-less `www.` link, whose host urlparse
    # reads as a path: every one was refused as "not a YouTube URL".
    url = args.input.strip()
    if "://" not in url:
        url = f"https://{url}"
    host = (urlparse(url).hostname or "").lower()

    # m.youtube.com is what phones share, music.youtube.com a YouTube Music link.
    if host not in ("youtube.com", "youtu.be") and not host.endswith(".youtube.com"):
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
        os.makedirs(outputFolder, exist_ok=True)
        args.output = os.path.join(
            outputFolder, generateOutputName(args, args.input, title)
        )
    elif os.path.isdir(args.output):
        outputFolder = args.output
        os.makedirs(outputFolder, exist_ok=True)
        if not processingEnabled:
            # The rename below needs a file in that folder, not the folder.
            args.output = os.path.join(
                outputFolder, generateOutputName(args, args.input, title)
            )
    else:
        # A bare `--output clip.mp4` has no folder part, and os.makedirs("")
        # raises before anything is downloaded.
        args.output = os.path.abspath(args.output)
        outputFolder = os.path.dirname(args.output)
        os.makedirs(outputFolder, exist_ok=True)

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
            # os.replace: os.rename refuses an existing --output on Windows,
            # and an explicit output is overwritten everywhere else.
            os.replace(tempOutput, args.output)
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

    if isSingleImageInput(args.input):
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

    # A URL may end in an image extension, so it is tested first.
    if isUrlInput(args.input):
        return processUrlInput(args, outputPath, processingEnabled)

    lowered = args.input.lower()
    if lowered.endswith((".png", ".jpg", ".jpeg")):
        normalizeImageInput(args, processingEnabled)
        return True

    if lowered.endswith(".gif"):
        # An --output naming another container keeps the method asked for:
        # gif forced into `--output clip.mp4` processed every frame and then
        # failed at the muxer.
        output = str(getattr(args, "output", None) or "").lower()
        namesOtherContainer = output.endswith(
            OUTPUT_FILE_EXTENSIONS
        ) and not output.endswith(".gif")
        if args.encode_method != "gif" and not namesOtherContainer:
            logging.error(
                "GIF input detected but encoding method is not set to GIF, defaulting to GIF encoding"
            )
            args.encode_method = "gif"
        return True

    try:
        args.input = os.path.abspath(args.input)
    except Exception as error:
        raise InputNormalizationError("Error processing input") from error
    return True
