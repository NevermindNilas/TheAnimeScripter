import logging
import os
from urllib.parse import urlparse

import src.constants as cs


class InputNormalizationError(ValueError):
    pass


def processUrlInput(args, outputPath, processingEnabled):
    from src.infra.logAndPrint import logAndPrint
    from src.io.inputOutputHandler import generateOutputName
    from src.ytdlp import VideoDownloader

    result = urlparse(args.input)

    if result.netloc.lower() not in ["www.youtube.com", "youtube.com", "youtu.be"]:
        raise InputNormalizationError(
            "URL is invalid or not a YouTube URL, please check the URL and try again"
        )

    logging.info("URL is valid and will be used for processing")

    if args.output is None:
        outputFolder = os.path.join(outputPath, "output")
        os.makedirs(os.path.join(outputFolder), exist_ok=True)
        args.output = os.path.join(outputFolder, generateOutputName(args, args.input))
    elif os.path.isdir(args.output):
        outputFolder = args.output
        os.makedirs(os.path.join(outputFolder), exist_ok=True)
    else:
        outputFolder = os.path.dirname(args.output)
        os.makedirs(os.path.join(outputFolder), exist_ok=True)

    tempOutput = os.path.join(outputFolder, generateOutputName(args, args.input))

    VideoDownloader(args.input, tempOutput, args.encode_method, args.custom_encoder)
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

        if processingEnabled and args.encode_method != "png":
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
