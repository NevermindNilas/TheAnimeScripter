import os
import logging
import sys
from urllib.parse import urlparse

from .generateOutput import outputNameGenerator
from .checkSpecs import checkSystem
from .getFFMPEG import getFFMPEG
from src.ytdlp import VideoDownloader


def argumentChecker(args, mainPath, scriptVersion):
    if args.version:
        print(scriptVersion)
        sys.exit()
    else:
        args.version = scriptVersion

    # Simplify boolean conversion
    boolArgs = [
        "ytdlp_quality",
        "interpolate",
        "scenechange",
        "ensemble",
        "denoise",
        "sharpen",
        "upscale",
        "segment",
        "resize",
        "dedup",
        "depth",
        "audio",
        "half",
    ]
    for arg in boolArgs:
        setattr(args, arg, getattr(args, arg) == 1)

    args.sharpen_sens /= 100
    args.scenechange_sens = 100 - args.scenechange_sens

    logging.info("============== Arguments ==============")

    argsDict = vars(args)
    for arg in argsDict:
        logging.info(f"{arg.upper()}: {argsDict[arg]}")

    checkSystem()

    logging.info("\n============== Arguments Checker ==============")
    args.ffmpeg_path = getFFMPEG()

    if args.dedup:
        args.audio = False
        logging.info("Dedup is enabled, audio will be disabled")

    if args.dedup_method == "ssim":
        args.dedup_sens = 1 - (args.dedup_sens / 1000)

    if args.custom_encoder:
        logging.info(
            "Custom encoder specified, use with caution since some functions can make or break the encoding process"
        )
    else:
        logging.info("No custom encoder specified, using default encoder")

    if "https://" in args.input or "http://" in args.input:
        processURL(args, mainPath)

    processingMethods = [
        args.interpolate,
        args.scenechange,
        args.upscale,
        args.segment,
        args.denoise,
        args.sharpen,
        args.resize,
        args.dedup,
        args.depth,
    ]

    result = urlparse(args.input)
    if not any(processingMethods) and not all([result.scheme, result.netloc]):
        print(
            "No processing methods specified, exiting",
        )
        logging.error(
            "No processing methods specified, exiting",
        )
        sys.exit()

    return args


def processURL(args, mainPath):
    """
    Check if the input is a URL, if it is, download the video and set the input to the downloaded video
    """
    result = urlparse(args.input)
    if all([result.scheme, result.netloc]):
        logging.info("URL is valid and will be used for processing")

        if args.output is None:
            outputFolder = os.path.join(mainPath, "output")
            os.makedirs(os.path.join(outputFolder), exist_ok=True)
            args.output = os.path.join(outputFolder, outputNameGenerator(args))

        VideoDownloader(
            args.input,
            args.output,
            args.ytdlp_quality,
            args.encode_method,
            args.custom_encoder,
            args.ffmpeg_path,
        )

        args.input = str(args.output)
        args.output = None
        logging.info(f"New input path: {args.input}")
    else:
        logging.error("URL is invalid, please check the URL and try again")
        sys.exit()
