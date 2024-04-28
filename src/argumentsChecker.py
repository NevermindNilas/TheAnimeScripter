import os
import logging
import sys
import time

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

    # Simplify boolean conversion
    boolArgs = [
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
        "offline",
        "consent",
    ]
    for arg in boolArgs:
        setattr(args, arg, getattr(args, arg) == 1)

    args.sharpen_sens /= 100
    args.scenechange_sens = 100 - args.scenechange_sens

    logging.info("============== Arguments ==============")

    argsDict = vars(args)
    for arg in argsDict:
        if argsDict[arg] is None or argsDict[arg] == "":
            continue
        logging.info(f"{arg.upper()}: {argsDict[arg]}")

    checkSystem()

    logging.info("\n============== Arguments Checker ==============")
    args.ffmpeg_path = getFFMPEG()

    try:
        args.input = args.input.encode('utf-8').decode('utf-8')
    except UnicodeDecodeError:
        toPrint = "Input video contains invalid characters. Please check the input and try again."
        logging.error(toPrint)
        print(red(toPrint))
        time.sleep(3)
        sys.exit()

    if args.offline:
        toPrint = "Offline mode enabled, downloading all available models"
        logging.info(toPrint)
        print(green(toPrint))
        options = modelsList()
        for option in options:
            if options == "apisr":
                for upscaleFactor in [2, 4]:
                    downloadModels(option, upscaleFactor=upscaleFactor)
            else:
                downloadModels(option)
        toPrint = "All models downloaded, exiting"
        logging.info(toPrint)
        print(green(toPrint))
        sys.exit()

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

    if args.consent:
        logging.info(
            "Consent flag detected, thank you for helping me improve the script"
        )

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
    if all([result.scheme, result.netloc]):
        logging.info("URL is valid and will be used for processing")

        if args.output is None:
            outputFolder = os.path.join(mainPath, "output")
            os.makedirs(os.path.join(outputFolder), exist_ok=True)
            args.output = os.path.join(outputFolder, outputNameGenerator(args))

        # TO:DO: Fix this, it's not working as intended
        #else:
        #    args.output = args.output.split(".")[0] + "_temp.mp4"
            
        VideoDownloader(
            args.input,
            args.output,
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
