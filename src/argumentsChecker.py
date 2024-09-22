import os
import logging
import sys
import argparse

from .checkSpecs import checkSystem
from .getFFMPEG import getFFMPEG
from .downloadModels import downloadModels, modelsList
from .coloredPrints import green, red, blue
from rich_argparse import RichHelpFormatter
from .version import __version__ as version


def createParser(isFrozen, mainPath, outputPath):
    argParser = argparse.ArgumentParser(
        description="The Anime Scripter CLI Tool",
        usage="main.py [options]" if not isFrozen else "main.exe [options]",
        formatter_class=RichHelpFormatter,
    )

    # Basic options
    generalGroup = argParser.add_argument_group("General")
    generalGroup.add_argument("--version", action="version", version=version)
    generalGroup.add_argument("--input", type=str, help="Input video file")
    generalGroup.add_argument("--output", type=str, help="Output video file")
    generalGroup.add_argument(
        "--inpoint", type=float, default=0, help="Input start time"
    )
    generalGroup.add_argument(
        "--outpoint", type=float, default=0, help="Input end time"
    )
    generalGroup.add_argument(
        "--preview", action="store_true", help="Preview the video during processing"
    )
    generalGroup.add_argument(
        "--hide_banner", action="store_true", help="Hide the TAS banner"
    )

    # Preset Configuration options
    presetGroup = argParser.add_argument_group("Preset Configuration")
    presetGroup.add_argument(
        "--preset",
        type=str,
        help="Create and use a preset configuration file based on the current arguments",
    )
    presetGroup.add_argument(
        "--list_presets",
        action="store_true",
        help="List all available presets",
    )

    # Performance options
    performanceGroup = argParser.add_argument_group("Performance")
    performanceGroup.add_argument(
        "--half", type=bool, help="Use half precision for inference", default=True
    )
    performanceGroup.add_argument(
        "--static",
        action="store_true",
        help="Force Static Mode engine generation for TensorRT",
    )

    # Interpolation options
    interpolationGroup = argParser.add_argument_group("Interpolation")
    interpolationGroup.add_argument(
        "--interpolate", action="store_true", help="Interpolate the video"
    )
    interpolationGroup.add_argument(
        "--interpolate_factor", type=int, default=2, help="Interpolation factor"
    )

    interpolationGroup.add_argument(
        "--interpolate_method",
        type=str,
        choices=[
            "rife",
            "rife4.6",
            "rife4.15-lite",
            "rife4.16-lite",
            "rife4.17",
            "rife4.18",
            "rife4.20",
            "rife4.21",
            "rife4.22",
            "rife4.22-lite",
            "rife4.25",
            "rife-ncnn",
            "rife4.6-ncnn",
            "rife4.15-lite-ncnn",
            "rife4.16-lite-ncnn",
            "rife4.17-ncnn",
            "rife4.18-ncnn",
            "rife4.20-ncnn",
            "rife4.21-ncnn",
            "rife4.22-ncnn",
            "rife4.22-lite-ncnn",
            "rife4.6-tensorrt",
            "rife4.15-tensorrt",
            "rife4.15-lite-tensorrt",
            "rife4.17-tensorrt",
            "rife4.18-tensorrt",
            "rife4.20-tensorrt",
            "rife4.21-tensorrt",
            "rife4.22-tensorrt",
            "rife4.22-lite-tensorrt",
            "rife4.25-tensorrt",
            "rife-tensorrt",
        ],
        default="rife",
        help="Interpolation method",
    )
    interpolationGroup.add_argument(
        "--ensemble",
        action="store_true",
        help="Use the ensemble model for interpolation",
    )
    interpolationGroup.add_argument(
        "--interpolate_skip",
        action="store_true",
        help="Use SSIM to skip duplicate frames when interpolating",
    )

    # Upscaling options
    upscaleGroup = argParser.add_argument_group("Upscaling")
    upscaleGroup.add_argument(
        "--upscale", action="store_true", help="Upscale the video"
    )
    upscaleGroup.add_argument(
        "--upscale_factor", type=int, choices=[2], default=2, help="Upscaling factor"
    )
    upscaleGroup.add_argument(
        "--upscale_method",
        type=str,
        choices=[
            "shufflespan",
            "shufflecugan",
            "compact",
            "ultracompact",
            "superultracompact",
            "span",
            "compact-directml",
            "ultracompact-directml",
            "superultracompact-directml",
            "shufflespan-directml",
            "span-directml",
            "shufflecugan-ncnn",
            "span-ncnn",
            "compact-tensorrt",
            "ultracompact-tensorrt",
            "superultracompact-tensorrt",
            "span-tensorrt",
            "shufflecugan-tensorrt",
            "shufflespan-tensorrt",
            "open-proteus",
            "open-proteus-tensorrt",
            "open-proteus-directml",
            "aniscale2",
            "aniscale2-tensorrt",
            "aniscale2-directml",
        ],
        default="shufflecugan",
        help="Upscaling method",
    )
    upscaleGroup.add_argument(
        "--custom_model", type=str, default="", help="Path to custom upscaling model"
    )
    upscaleGroup.add_argument(
        "--upscale_skip",
        action="store_true",
        help="Use SSIM to skip duplicate frames when upscaling",
    )

    # Deduplication options
    dedupGroup = argParser.add_argument_group("Deduplication")
    dedupGroup.add_argument(
        "--dedup", action="store_true", help="Deduplicate the video"
    )
    dedupGroup.add_argument(
        "--dedup_method",
        type=str,
        default="ssim",
        choices=["ssim", "mse", "ssim-cuda", "mse-cuda"],
        help="Deduplication method",
    )
    dedupGroup.add_argument(
        "--dedup_sens", type=float, default=35, help="Deduplication sensitivity"
    )
    dedupGroup.add_argument(
        "--sample_size", type=int, default=224, help="Sample size for deduplication"
    )

    # Video processing options
    processingGroup = argParser.add_argument_group("Video Processing")
    processingGroup.add_argument(
        "--sharpen", action="store_true", help="Sharpen the video"
    )
    processingGroup.add_argument(
        "--sharpen_sens", type=float, default=50, help="Sharpening sensitivity"
    )
    processingGroup.add_argument(
        "--denoise", action="store_true", help="Denoise the video"
    )
    processingGroup.add_argument(
        "--denoise_method",
        type=str,
        default="scunet",
        choices=["scunet", "nafnet", "dpir", "real-plksr"],
        help="Denoising method",
    )
    processingGroup.add_argument(
        "--resize", action="store_true", help="Resize the video"
    )
    processingGroup.add_argument(
        "--resize_factor",
        type=float,
        default=2,
        help="Resize factor (can be between 0 and 1 for downscaling)",
    )
    processingGroup.add_argument(
        "--resize_method",
        type=str,
        choices=[
            "fast_bilinear",
            "bilinear",
            "bicubic",
            "experimental",
            "neighbor",
            "area",
            "bicublin",
            "gauss",
            "sinc",
            "lanczos",
            "point",
            "spline",
            "spline16",
            "spline36",
        ],
        default="bicubic",
        help="Resize method",
    )

    # Segmentation options
    segmentationGroup = argParser.add_argument_group(
        "Segmentation / Background Removal"
    )
    segmentationGroup.add_argument(
        "--segment", action="store_true", help="Segment the video"
    )
    segmentationGroup.add_argument(
        "--segment_method",
        type=str,
        default="anime",
        choices=["anime", "anime-tensorrt", "anime-directml", "cartoon"],
        help="Segmentation method",
    )

    # Scene detection options
    sceneGroup = argParser.add_argument_group("Scene Detection")
    sceneGroup.add_argument(
        "--autoclip", action="store_true", help="Detect scene changes"
    )
    sceneGroup.add_argument(
        "--autoclip_sens", type=float, default=50, help="Autoclip sensitivity"
    )
    sceneGroup.add_argument(
        "--scenechange", action="store_true", help="Detect scene changes"
    )
    sceneGroup.add_argument(
        "--scenechange_method",
        type=str,
        default="maxxvit-directml",
        choices=[
            "maxxvit-tensorrt",
            "maxxvit-directml",
            "differential",
            "differential-tensorrt",
            "shift_lpips-tensorrt",
            "shift_lpips-directml",
        ],
        help="Scene change detection method",
    )
    sceneGroup.add_argument(
        "--scenechange_sens",
        type=float,
        default=50,
        help="Scene change detection sensitivity (0-100)",
    )

    # Depth estimation options
    depthGroup = argParser.add_argument_group("Depth Estimation")
    depthGroup.add_argument(
        "--depth", action="store_true", help="Estimate the depth of the video"
    )
    depthGroup.add_argument(
        "--depth_method",
        type=str,
        choices=[
            "small_v2",
            "base_v2",
            "large_v2",
            "small_v2-tensorrt",
            "base_v2-tensorrt",
            "large_v2-tensorrt",
            "small_v2-directml",
            "base_v2-directml",
            "large_v2-directml",
        ],
        default="small_v2",
        help="Depth estimation method",
    )
    depthGroup.add_argument(
        "--depth_quality",
        type=str,
        choices=["low", "high"],
        default="high",
        help="This will determine the quality of the depth map, low is significantly faster but lower quality",
    )

    # Encoding options
    encodingGroup = argParser.add_argument_group("Encoding")
    encodingGroup.add_argument(
        "--encode_method",
        type=str,
        choices=[
            "x264",
            "x264_10bit",
            "x264_animation",
            "x264_animation_10bit",
            "x265",
            "x265_10bit",
            "nvenc_h264",
            "nvenc_h265",
            "nvenc_h265_10bit",
            "nvenc_av1",
            "qsv_h264",
            "qsv_h265",
            "qsv_h265_10bit",
            "av1",
            "h264_amf",
            "hevc_amf",
            "hevc_amf_10bit",
            "prores",
            "prores_segment",
            "gif",
            "image",
            "vp9",
            "qsv_vp9",
        ],
        default="x264",
        help="Encoding method",
    )
    encodingGroup.add_argument(
        "--custom_encoder", type=str, default="", help="Custom encoder settings"
    )

    # Stabilizer Options
    stabilizerGroup = argParser.add_argument_group("Stabilizer")
    stabilizerGroup.add_argument(
        "--stabilize", action="store_true", help="Stabilize the video using VidStab"
    )

    # Miscellaneous options
    miscGroup = argParser.add_argument_group("Miscellaneous")
    miscGroup.add_argument("--buffer_limit", type=int, default=50, help="Buffer limit")

    miscGroup.add_argument(
        "--benchmark", action="store_true", help="Benchmark the script"
    )
    miscGroup.add_argument(
        "--offline",
        type=str,
        nargs="*",
        default="none",
        help="Download a specific model or multiple models for offline use, use keyword 'all' to download all models",
    )
    miscGroup.add_argument(
        "--ae",
        action="store_true",
        help="Notify if script is run from After Effects interface",
    )
    miscGroup.add_argument(
        "--bit_depth",
        type=str,
        default="8bit",
        help="Bit Depth of the raw pipe input to FFMPEG. Useful if you want the highest quality possible - this doesn't have anything to do with --pix_fmt of the encoded ffmpeg.",
        choices=["8bit", "16bit"],
    )

    args = argParser.parse_args()
    return argumentsChecker(args, mainPath, outputPath)


def argumentsChecker(args, mainPath, outputPath):
    banner = r"""
__/\\\\\\\\\\\\\\\_____/\\\\\\\\\________/\\\\\\\\\\\___
 _\///////\\\/////____/\\\\\\\\\\\\\____/\\\/////////\\\_
  _______\/\\\________/\\\/////////\\\__\//\\\______\///__
   _______\/\\\_______\/\\\_______\/\\\___\////\\\_________
    _______\/\\\_______\/\\\\\\\\\\\\\\\______\////\\\______
     _______\/\\\_______\/\\\/////////\\\_________\////\\\___
      _______\/\\\_______\/\\\_______\/\\\__/\\\______\//\\\__
       _______\/\\\_______\/\\\_______\/\\\_\///\\\\\\\\\\\/___
        _______\///________\///________\///____\///////////_____
"""

    if args.list_presets:
        from src.presetLogic import listPresets

        listPresets(mainPath)
        sys.exit()

    if args.preset:
        from src.presetLogic import createPreset

        args = createPreset(args, mainPath)

    if not args.benchmark and not args.hide_banner:
        print(red(banner))

    logging.info("============== Version ==============")
    logging.info(f"TAS: {version}\n")

    logging.info("============== Arguments ==============")
    for arg, value in vars(args).items():
        if value not in [None, "", "none", False]:
            logging.info(f"{arg.upper()}: {value}")

    checkSystem()

    logging.info("\n============== Arguments Checker ==============")
    args.ffmpeg_path = getFFMPEG()

    def adjustFeature(
        feature,
        dependsOn,
        imcompatibleWith,
        enabledMessage,
        imcompatibleMessage,
        missingDependencyMessage,
    ):
        if getattr(args, feature):
            logging.info(
                enabledMessage,
            )
            if getattr(
                args,
                imcompatibleWith,
            ):
                logging.error(
                    imcompatibleMessage,
                )
                setattr(args, feature, False)
            elif not getattr(args, dependsOn):
                logging.error(
                    missingDependencyMessage,
                )
                setattr(args, feature, False)

    adjustFeature(
        "upscale_skip",
        "upscale",
        "dedup",
        "Upscale skip enabled...",
        "Upscale skip and dedup cannot be used together...",
        "Upscale skip is enabled but upscaling is not...",
    )

    adjustFeature(
        "interpolate_skip",
        "interpolate",
        "dedup",
        "Interpolate skip enabled...",
        "Interpolate skip and dedup cannot be used together...",
        "Interpolate skip is enabled but interpolation is not...",
    )

    if args.offline != "none":
        logging.info(f"Offline mode enabled, downloading {args.offline} model(s)...")
        print(green(f"Offline mode enabled, downloading {args.offline} model(s)..."))
        options = modelsList() if args.offline == ["all"] else args.offline
        for option in options:
            for precision in [True, False]:
                try:
                    downloadModels(option.lower(), half=precision)
                except Exception:
                    logging.error(
                        f"Failed to download model: {option} with precision: {'fp16' if precision else 'fp32'}"
                    )
        logging.info("All model(s) downloaded!")
        print(green("All model(s) downloaded!"))

    if args.dedup:
        if args.dedup_method in ["ssim", "ssim-cuda"]:
            args.dedup_sens = 1.0 - (args.dedup_sens / 1000)
            logging.info(
                f"New dedup sensitivity for {args.dedup_method} is: {args.dedup_sens}"
            )

    sensMap = {"differential": 0.65, "shift_lpips": 0.50, "maxxvit": 0.9}
    if args.scenechange_method in sensMap:
        args.scenechange_sens = sensMap[args.scenechange_method] - (
            args.scenechange_sens / 1000
        )
        logging.info(
            f"New scenechange sensitivity for {args.scenechange_method} is: {args.scenechange_sens}"
        )

    if args.custom_encoder:
        logging.info("Custom encoder specified, use with caution")
    else:
        logging.info("No custom encoder specified, using default encoder")

    if args.bit_depth == "16bit" and args.segment:
        logging.error(
            "16bit input is not supported with segmentation, defaulting to 8bit"
        )
        args.bit_depth = "8bit"

    if args.output and not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

    if not args.input:
        logging.error("No input specified")
        sys.exit()
    elif args.input.startswith(("http", "www")):
        processURL(args, outputPath)
    elif args.input.lower().endswith((".png", ".jpg", ".jpeg")):
        if args.encode_method not in ["gif", "image"]:
            logging.error(
                "Image input detected but encoding method not set to GIF or Image, defaulting to Image encoding"
            )
            args.encode_method = "image"
            args.isImage = True
    else:
        try:
            args.input = os.path.abspath(args.input)
        except Exception:
            logging.error("Error processing input")
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
        args.autoclip,
        args.stabilize,
    ]
    if not any(processingMethods):
        logging.error("No processing methods specified, exiting")
        sys.exit()

    return args


def processURL(args, outputPath):
    """
    Check if the input is a URL, if it is, download the video and set the input to the downloaded video
    """
    from urllib.parse import urlparse
    from src.ytdlp import VideoDownloader

    result = urlparse(args.input)

    if result.netloc.lower() in ["www.youtube.com", "youtube.com", "youtu.be"]:
        logging.info("URL is valid and will be used for processing")

        if args.output is None:
            from .generateOutput import outputNameGenerator

            outputFolder = os.path.join(outputPath, "output")
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


def inputChecker(input, encodeMethod, customEncoder):
    """
    In order to prevent issues with webm inputs and certain encoders, we need to check the input and the encoder
    """
    if not customEncoder:
        if input.endswith(".webm"):
            if encodeMethod not in ["vp9", "qsv_vp9", "av1"]:
                toPrint = "WebM input detected, defaulting to VP9 encoding"
                logging.error(toPrint)
                print(blue(toPrint))
        elif input.endswith((".png", ".jpg", ".jpeg")):
            if encodeMethod not in [".gif", "image"]:
                logging.error(
                    "Image input detected but encoding method is not set to GIF or Image, defaulting to Image encoding"
                )
                encodeMethod = "image"
