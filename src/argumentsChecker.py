import os
import logging
import sys
import validators

from .generateOutput import outputNameGenerator
from .checkSpecs import checkSystem
from .getFFMPEG import getFFMPEG


def argumentChecker(args, mainPath, scriptVersion):
    if args.version:
        print(scriptVersion)
        sys.exit()
    else:
        args.version = scriptVersion

    # Whilst this is ugly, it was easier to work with the Extendscript interface this way
    # This is a temporary solution until I can find a better way to handle the arguments
    args.ytdlp_quality = True if args.ytdlp_quality == 1 else False
    args.interpolate = True if args.interpolate == 1 else False
    args.scenechange = True if args.scenechange == 1 else False
    args.ensemble = True if args.ensemble == 1 else False
    args.denoise = True if args.denoise == 1 else False
    args.sharpen = True if args.sharpen == 1 else False
    args.upscale = True if args.upscale == 1 else False
    args.segment = True if args.segment == 1 else False
    args.resize = True if args.resize == 1 else False
    args.dedup = True if args.dedup == 1 else False
    args.depth = True if args.depth == 1 else False
    args.audio = True if args.audio == 1 else False
    args.half = True if args.half == 1 else False

    args.sharpen_sens /= 100  # CAS works from 0.0 to 1.0
    args.scenechange_sens = 100 - args.scenechange_sens

    logging.info("============== Arguments ==============")

    args_dict = vars(args)
    for arg in args_dict:
        logging.info(f"{arg.upper()}: {args_dict[arg]}")

    checkSystem()

    logging.info("\n============== Arguments Checker ==============")
    args.ffmpeg_path = getFFMPEG()

    if args.dedup:
        args.audio = False
        logging.info("Dedup is enabled, audio will be disabled")

    if args.denoise:
        if args.denoise_method == "nafnet" and args.half:
            logging.info(
                "NAFNet does not support half precision, setting half to False"
            )
            args.half = False

    if args.dedup_method == "ssim":
        args.dedup_sens = (
            1 - (args.dedup_sens / 1000)
        )  # SSIM works from -1 to 1, but results prove to be efficient only inbetween ~0.9 and 0.999, lower values are not reliable and may remove important frames

    if args.dedup_method == "mse":
        args.dedup_sens = args.dedup_sens

    if args.custom_encoder:
        logging.info(
            "Custom encoder specified, use with caution since some functions can make or break the encoding process"
        )
    else:
        logging.info("No custom encoder specified, using default encoder")

    if "https://" in args.input or "http://" in args.input:
        result = validators.url(args.input)
        if result:
            logging.info("URL is valid and will be used for processing")
            from src.ytdlp import VideoDownloader

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

    processing_methods = [
        args.upscale,
        args.interpolate,
        args.denoise,
        args.sharpen,
        args.segment,
        args.resize,
        args.depth,
        args.dedup,
    ]

    if (
        not any(processing_methods)
        and "http://" not in args.input
        and "https://" not in args.input
    ):
        logging.error(
            "No processing methods specified, please enable at least one processing method"
        )
        sys.exit()

    return args
