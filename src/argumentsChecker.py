import os
import logging
import sys

from .generateOutput import outputNameGenerator
from .checkSpecs import checkSystem
from .getFFMPEG import getFFMPEG

def argumentChecker(args, mainPath):
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
        
    if args.dedup_method == "ssim":
        args.dedup_sens = 1 - (args.dedup_sens / 1000) # SSIM works from -1 to 1, but results prove to be efficient only inbetween ~0.9 and 0.999, lower values are not reliable and may remove important frames

    if not args.ytdlp == "":
        logging.info(f"Downloading {args.ytdlp} video")
        from src.ytdlp import VideoDownloader

        if args.output is None:
            outputFolder = os.path.join(mainPath, "output")
            os.makedirs(os.path.join(outputFolder), exist_ok=True)
            
        args.output = os.path.join(outputFolder, outputNameGenerator(args))

        VideoDownloader(
            args.ytdlp,
            args.output,
            args.ytdlp_quality,
            args.encode_method,
            args.custom_encoder,
            args.ffmpeg_path,
        )
        sys.exit()
    
    return args.ffmpeg_path