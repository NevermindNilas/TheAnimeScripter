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
        "offline",
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



    if args.offline:
        toPrint = "Offline mode enabled, downloading all available models, this can take a minute but it will allow for the script to be used offline"
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

    if args.dedup:
        args.audio = False
        logging.info("Dedup is enabled, audio will be disabled")

    if args.dedup_method == "ssim" or args.dedup_method == "ssim-cuda":
        args.dedup_sens = 1.0 - (args.dedup_sens / 1000) * 0.1

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
        try:
            args.input = args.input.encode("utf-8").decode("utf-8")
            args.input = args.input.replace("\\", "/")
        except UnicodeDecodeError:
            toPrint = "Input video contains invalid characters in it's name. Please check the input and try again. One suggestion would be renaming it to something simpler like test.mp4"
            logging.error(toPrint)
            print(red(toPrint))
            time.sleep(3)
            sys.exit()
    
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
        # else:
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
