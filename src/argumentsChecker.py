import os
import logging
import sys

from urllib.parse import urlparse
from .generateOutput import outputNameGenerator
from .checkSpecs import checkSystem
from .getFFMPEG import getFFMPEG
from src.ytdlp import VideoDownloader
from .downloadModels import downloadModels, modelsList
from .coloredPrints import green, red


def argumentChecker(args, mainPath, scriptVersion):
    banner = """
_____________            _______       _____                      ________            _____        _____             
___  __/__  /______      ___    |_________(_)______ ________      __  ___/_______________(_)_________  /_____________
__  /  __  __ \  _ \     __  /| |_  __ \_  /__  __ `__ \  _ \     _____ \_  ___/_  ___/_  /___  __ \  __/  _ \_  ___/
_  /   _  / / /  __/     _  ___ |  / / /  / _  / / / / /  __/     ____/ // /__ _  /   _  / __  /_/ / /_ /  __/  /    
/_/    /_/ /_/\___/      /_/  |_/_/ /_//_/  /_/ /_/ /_/\___/      /____/ \___/ /_/    /_/  _  .___/\__/ \___//_/     
                                                                                           /_/                       
"""

    if not args.benchmark:
        print(red(banner))

    args.sharpen_sens /= 100
    args.autoclip_sens = 100 - args.autoclip_sens

    logging.info("============== Arguments ==============")

    argsDict = vars(args)
    for arg in argsDict:
        if argsDict[arg] is None or argsDict[arg] == "":
            continue
        logging.info(f"{arg.upper()}: {argsDict[arg]}")

    checkSystem()

    logging.info("\n============== Arguments Checker ==============")
    args.ffmpeg_path = getFFMPEG()

    if args.offline:
        toPrint = "Offline mode enabled, downloading all available models, this can take some time but it will allow for the script to be used offline"
        logging.info(toPrint)
        print(green(toPrint))
        options = modelsList()
        for option in options:
            downloadModels(option)
        toPrint = "All models downloaded!"
        logging.info(toPrint)
        print(green(toPrint))

    if args.dedup:
        logging.info("Dedup is enabled, audio will be disabled")
        args.audio = False

    if args.dedup_method in ["ssim", "ssim-cuda"]:
        args.dedup_sens = 1.0 - (args.dedup_sens / 1000)
        logging.info(
            f"New dedup sensitivity for {args.dedup_method} is: {args.dedup_sens}"
        )

    if args.scenechange_sens:
        args.scenechange_sens = 0.9 - (args.scenechange_sens / 1000)
        logging.info(f"New scenechange sensitivity is: {args.scenechange_sens}")

    if args.custom_encoder:
        logging.info(
            "Custom encoder specified, use with caution since some functions can make or break the encoding process"
        )
    else:
        logging.info("No custom encoder specified, using default encoder")

    if args.upscale_skip:
        logging.info(
            "Upscale skip enabled, the script will skip frames that are upscaled to save time, this is far from perfect and can cause issues"
        )

    if args.upscale_skip and args.dedup:
        logging.error(
            "Upscale skip and dedup cannot be used together, disabling upscale skip to prevent issues"
        )
        args.upscale_skip = False

    if args.upscale_skip and not args.upscale:
        logging.error(
            "Upscale skip is enabled but upscaling is not, disabling upscale skip"
        )
        args.upscale_skip = False

    """
    # Doesn't work with AMD GPUs
    if args.half:
        try:
            import torch
        except ImportError:
            logging.info("Torch is not installed, please install it to use the script")

        if torch.cuda.is_bf16_supported():
            logging.info("Half precision enabled")
        else:
            logging.info("Half precision is not supported on your system, disabling it")
            args.half = False
    """

    if args.input is None:
        toPrint = "No input specified, please specify an input file or URL to continue"
        logging.error(toPrint)
        print(red(toPrint))
        sys.exit()
    else:
        if args.input.startswith("http") or args.input.startswith("www"):
            processURL(args, mainPath)
        else:
            try:
                args.input = os.path.abspath(args.input)
                args.input = str(args.input)
            except Exception:
                logging.error(
                    "Error processing the input, this is usually because of weird input names with spaces or characters that are not allowed"
                )
                sys.exit()

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

    if not any(processingMethods):
        toPrint = "No other processing methods specified, exiting"
        logging.error(toPrint)
        print(red(toPrint))
        sys.exit()

    return args


def processURL(args, mainPath):
    """
    Check if the input is a URL, if it is, download the video and set the input to the downloaded video
    """
    result = urlparse(args.input)
    if result.netloc.lower() in ["www.youtube.com", "youtube.com", "youtu.be"]:
        logging.info("URL is valid and will be used for processing")

        if args.output is None:
            outputFolder = os.path.join(mainPath, "output")
            os.makedirs(os.path.join(outputFolder), exist_ok=True)
            args.output = os.path.join(outputFolder, outputNameGenerator(args))

        VideoDownloader(
            args.input,
            args.output,
            args.encode_method,
            args.custom_encoder,
            args.ffmpeg_path,
            args.ae,
        )

        args.input = str(args.output)
        args.output = None
        logging.info(f"New input path: {args.input}")
    else:
        logging.error(
            "URL is invalid or not a YouTube URL, please check the URL and try again"
        )
        sys.exit()
