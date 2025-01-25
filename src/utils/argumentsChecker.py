import os
import logging
import sys
import argparse
import shutil

from .checkSpecs import checkSystem
from .downloadModels import downloadModels, modelsList
from .coloredPrints import green, yellow
from rich_argparse import RichHelpFormatter
from src.version import __version__ as version
from .generateOutput import outputNameGenerator
from src.utils.logAndPrint import logAndPrint


def isAnyOtherProcessingMethodEnabled(args):
    proccessingMethods = [
        args.interpolate,
        args.scenechange,
        args.upscale,
        args.segment,
        args.restore,
        args.sharpen,
        args.resize,
        args.dedup,
        args.depth,
        args.autoclip,
    ]

    return any(proccessingMethods)


def str2bool(arg):
    """
    No clue if this is the right approach. But it works.
    """
    if isinstance(arg, bool):
        return arg
    if arg.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif arg.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def createParser(isFrozen, mainPath, outputPath, sysUsed):
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
        "--realtime",
        action="store_true",
        help="Realtime Preview the video during processing, this downloads FFPLAY if not found",
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
        "--precision",
        type=str,
        choices=["fp32", "fp16"],
        default="fp16",
        help="NOT IMPLEMENTED YET! Precision for inference, default is fp16",
    )
    performanceGroup.add_argument(
        "--half",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Use half precision for inference (default: True)",
    )
    performanceGroup.add_argument(
        "--static",
        action="store_true",
        help="Force Static Mode engine generation for TensorRT",
    )
    performanceGroup.add_argument(
        "--decode_threads",
        type=int,
        default=4,
        help="Number of threads to use for decoding",
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
            "rife4.25-lite",
            "rife4.25-heavy",
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
            "rife4.17-tensorrt",
            "rife4.18-tensorrt",
            "rife4.20-tensorrt",
            "rife4.21-tensorrt",
            "rife4.22-tensorrt",
            "rife4.22-lite-tensorrt",
            "rife4.25-tensorrt",
            "rife4.25-lite-tensorrt",
            "rife4.25-heavy-tensorrt",
            "rife-tensorrt",
            "gmfss",
            "gmfss-tensorrt",
            "rife_elexor",
            "rife_elexor-tensorrt",
            "rife4.6-directml",
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
        "--dynamic_scale",
        action="store_true",
        help="Use dynamic scaling for interpolation, this can improve the quality of the interpolation at the cost of performance, this is experimental and only works with Rife CUDA",
    )

    # Upscaling options
    upscaleGroup = argParser.add_argument_group("Upscaling")
    upscaleGroup.add_argument(
        "--upscale", action="store_true", help="Upscale the video"
    )
    upscaleGroup.add_argument(
        "--upscale_factor",
        type=int,
        choices=[1, 2, 3, 4],
        default=2,
        help="Upscaling factor",
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
            "rtmosr-tensorrt",
            "rtmosr-directml",
        ],
        default="shufflecugan",
        help="Upscaling method",
    )
    upscaleGroup.add_argument(
        "--custom_model", type=str, default="", help="Path to custom upscaling model"
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
        choices=["ssim", "mse", "ssim-cuda", "mse-cuda", "flownets"],
        help="Deduplication method",
    )
    dedupGroup.add_argument(
        "--dedup_sens", type=float, default=35, help="Deduplication sensitivity"
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
        "--restore", action="store_true", help="Restore the video"
    )
    processingGroup.add_argument(
        "--restore_method",
        type=str,
        default="scunet",
        choices=[
            "scunet",
            "nafnet",
            "dpir",
            "real-plksr",
            "anime1080fixer",
            "anime1080fixer-tensorrt",
            "anime1080fixer-directml",
            "fastlinedarken",
            "fastlinedarken-tensorrt",
        ],
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
        default="low",
        help="[DEPRECATED]This will determine the quality of the depth map, low is significantly faster but lower quality",
    )

    # Encoding options
    encodingGroup = argParser.add_argument_group("Encoding")
    encodingGroup.add_argument(
        "--encode_method",
        type=str,
        choices=[
            "x264",
            "slow_x264",
            "x264_10bit",
            "x264_animation",
            "x264_animation_10bit",
            "x265",
            "slow_x265",
            "x265_10bit",
            "nvenc_h264",
            "slow_nvenc_h264",
            "nvenc_h265",
            "slow_nvenc_h265",
            "nvenc_h265_10bit",
            "nvenc_av1",
            "slow_nvenc_av1",
            "qsv_h264",
            "qsv_h265",
            "qsv_h265_10bit",
            "av1",
            "slow_av1",
            "h264_amf",
            "hevc_amf",
            "hevc_amf_10bit",
            "prores",
            "prores_segment",
            "gif",
            "vp9",
            "qsv_vp9",
            "h266",
        ],
        default="x264",
        help="Encoding method",
    )
    encodingGroup.add_argument(
        "--custom_encoder", type=str, default="", help="Custom encoder settings"
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
    return argumentsChecker(args, mainPath, outputPath, sysUsed)


def argumentsChecker(args, mainPath, outputPath, sysUsed):
    if args.list_presets:
        from src.utils.presetLogic import listPresets

        listPresets(mainPath)
        sys.exit()

    if args.preset:
        from src.utils.presetLogic import createPreset

        args = createPreset(args, mainPath)

    logging.info("============== Version ==============")
    logging.info(f"TAS: {version}\n")

    logging.info("============== Arguments ==============")
    for arg, value in vars(args).items():
        if value not in [None, "", "none", False]:
            logging.info(f"{arg.upper()}: {value}")

    if not args.benchmark:
        checkSystem(sysUsed)

    logging.info("\n============== Arguments Checker ==============")
    args.ffmpeg_path = os.path.join(
        mainPath,
        "ffmpeg",
        "ffmpeg.exe" if sysUsed == "Windows" else "ffmpeg",
    )

    args.ffprobe_path = os.path.join(
        mainPath,
        "ffmpeg",
        "ffprobe.exe" if sysUsed == "Windows" else "ffprobe",
    )

    args.mpv_path = os.path.join(
        mainPath,
        "ffmpeg",
        "mpv.exe" if sysUsed == "Windows" else "mpv",
    )

    if not os.path.exists(args.ffmpeg_path) or (
        args.realtime
        and not os.path.exists(args.mpv_path)
        or not os.path.exists(args.ffprobe_path)
    ):
        from src.utils.getFFMPEG import getFFMPEG

        args.ffmpeg_path, args.mpv_path, args.ffprobe_path = getFFMPEG(
            sysUsed, args.ffmpeg_path, args.realtime
        )

    if args.realtime:
        print(yellow("Realtime preview enabled, this is experimental!"))

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

    # ["tensorrt", "directml"] in args.depth_method:
    if args.depth_quality != "low" and args.depth_method.split("-")[-1] in [
        "tensorrt",
        "directml",
    ]:
        logging.error(
            "High quality depth estimation is deprecated for tensorrt and directml, defaulting to low quality"
        )
        print(
            yellow(
                "High quality depth estimation is deprecated for tensorrt and directml, defaulting to low quality"
            )
        )
        args.depth_quality = "low"

    if args.benchmark and args.realtime:
        logging.error("Realtime preview is not supported in benchmark mode")
        args.realtime = False

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

        elif args.dedup_method in ["flownets"]:
            args.dedup_sens = args.dedup_sens / 100

        logging.info(
            f"New dedup sensitivity for {args.dedup_method} is: {args.dedup_sens}"
        )

    sensMap = {
        "differential": 0.75,
        "differential-tensorrt": 0.75,
        "shift_lpips": 0.50,
        "maxxvit": 0.9,
        "maxxvit-tensorrt": 0.9,
        "maxxvit-directml": 0.9,
    }
    if args.scenechange_method in sensMap and args.scenechange is True:
        args.scenechange_sens = sensMap[args.scenechange_method] - (
            args.scenechange_sens / 1000
        )
        logging.info(
            f"New scenechange sensitivity for {args.scenechange_method} is: {args.scenechange_sens}"
        )

    if args.sharpen:
        args.sharpen_sens = args.sharpen_sens / 100
        logging.info(f"New sharpen sensitivity is: {args.sharpen_sens}")

    if args.custom_encoder:
        logging.info("Custom encoder specified, use with caution")

    if args.bit_depth == "16bit" and args.segment:
        logging.error(
            "16bit input is not supported with segmentation, defaulting to 8bit"
        )
        args.bit_depth = "8bit"

    if args.decode_threads < 1:
        logging.error("Invalid decode threads, defaulting to 2")
        args.decode_threads = 2
    elif args.decode_threads > os.cpu_count():
        logging.error(
            f"Decode threads higher than available CPU threads, defaulting to {os.cpu_count()}"
        )
        args.decode_threads = os.cpu_count()

    if args.output and not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

    if not args.input:
        logging.error("No input specified")
        sys.exit()
    elif args.input.startswith(("http", "www")):
        processURL(args, outputPath)
    elif args.input.lower() == "anime":
        processAniPy(args, outputPath)

    elif args.input.lower().endswith((".png", ".jpg", ".jpeg")):
        raise Exception(
            "Image input is not supported, use Chainner for image processing"
        )
    elif args.input.lower().endswith((".gif")):
        if args.encode_method != "gif":
            logging.error(
                "GIF input detected but encoding method is not set to GIF, defaulting to GIF encoding"
            )
            args.encode_method = "gif"
    else:
        try:
            args.input = os.path.abspath(args.input)
        except Exception:
            logging.error("Error processing input")
            sys.exit()

    if not isAnyOtherProcessingMethodEnabled(args):
        logAndPrint(
            "No processing methods specified, make sure to use enabler arguments like --upscale, --interpolate, etc.",
            "red",
        )
        # logging.error("No processing methods specified, exiting")
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
            outputFolder = os.path.join(outputPath, "output")
            os.makedirs(os.path.join(outputFolder), exist_ok=True)
            args.output = os.path.join(
                outputFolder, outputNameGenerator(args, args.input)
            )
        elif os.path.isdir(args.output):
            outputFolder = args.output
            os.makedirs(os.path.join(outputFolder), exist_ok=True)
        else:
            outputFolder = os.path.dirname(args.output)
            os.makedirs(os.path.join(outputFolder), exist_ok=True)

        tempOutput = os.path.join(outputFolder, outputNameGenerator(args, args.input))

        VideoDownloader(
            args.input,
            tempOutput,
            args.encode_method,
            args.custom_encoder,
            args.ffmpeg_path,
            args.ae,
        )
        print(green(f"Video downloaded to: {tempOutput}"))

        if not isAnyOtherProcessingMethodEnabled(args):
            shutil.move(tempOutput, args.output)
            sys.exit()

        args.input = str(tempOutput)
        logging.info(f"New input path: {args.input}")

    else:
        logging.error(
            "URL is invalid or not a YouTube URL, please check the URL and try again"
        )
        sys.exit()


def processAniPy(args, outputPath: str):
    from .generateOutput import outputNameGenerator

    if args.output is None:
        outputFolder = os.path.join(outputPath, "output")
        os.makedirs(os.path.join(outputFolder), exist_ok=True)
        fullOutput = os.path.join(outputFolder, outputNameGenerator(args, args.input))
    elif os.path.isdir(args.output):
        fullOutput = os.path.join(args.output, outputNameGenerator(args, args.input))
    else:
        fullOutput = args.output
    from src.utils.anipyLogic import aniPyHandler

    args.input = aniPyHandler(fullOutput, args.ffmpeg_path)
    logging.info(f"New input path: {args.input}")
