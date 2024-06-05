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
    if args.version:
        print(scriptVersion)
        sys.exit()
    else:
        args.version = scriptVersion

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
        toPrint = "Offline mode enabled, downloading all available models, this can take a minute but it will allow for the script to be used offline"
        logging.info(toPrint)
        print(green(toPrint))
        options = modelsList()
        for option in options:
            downloadModels(option)
        toPrint = "All models downloaded, exiting"
        logging.info(toPrint)
        print(green(toPrint))

    if args.dedup:
        args.audio = False
        logging.info("Dedup is enabled, audio will be disabled")

    if args.dedup_method == "ssim" or args.dedup_method == "ssim-cuda":
        args.dedup_sens = 1.0 - (args.dedup_sens / 1000)

    if args.custom_encoder:
        logging.info(
            "Custom encoder specified, use with caution since some functions can make or break the encoding process"
        )
    else:
        logging.info("No custom encoder specified, using default encoder")

    if args.consent:
        logging.info(
            "Consent flag detected, thank you for helping me improve the script"
        )

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

    if args.update:
        logging.info("Update flag detected, checking for updates")
        from .updateScript import updateScript

        updateScript(scriptVersion, mainPath)
        sys.exit()

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
    if result.netloc.lower() in ['www.youtube.com', 'youtube.com', 'youtu.be']:
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
        logging.error("URL is invalid or not a YouTube URL, please check the URL and try again")
        sys.exit()
