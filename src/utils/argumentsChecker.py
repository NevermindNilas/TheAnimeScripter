import os
import logging
import sys
import argparse
import shutil
import src.constants as cs

from .coloredPrints import green, yellow
from rich_argparse import RichHelpFormatter
from src.version import __version__
from .inputOutputHandler import outputNameGenerator
from src.utils.logAndPrint import logAndPrint


def isAnyOtherProcessingMethodEnabled(args):
    return any(
        [
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
            args.obj_detect,
        ]
    )


def str2bool(arg):
    if isinstance(arg, bool):
        return arg
    if arg.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif arg.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def createParser(outputPath):
    argParser = argparse.ArgumentParser(
        description="The Anime Scripter CLI Tool",
        usage="main.py [options]",
        formatter_class=RichHelpFormatter,
    )

    # Basic options
    generalGroup = argParser.add_argument_group("General")
    generalGroup.add_argument("--version", action="version", version=__version__)
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
        "--list_presets", action="store_true", help="List all available presets"
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

    # Interpolation options
    _addInterpolationOptions(argParser)

    # Upscaling options
    _addUpscalingOptions(argParser)

    # Deduplication options
    _addDedupOptions(argParser)

    # Video processing options
    _addVideoProcessingOptions(argParser)

    # Segmentation options
    _addSegmentationOptions(argParser)

    # Scene detection options
    _addSceneDetectionOptions(argParser)

    # Depth estimation options
    _addDepthOptions(argParser)

    # Encoding options
    _addEncodingOptions(argParser)

    # Object Detection options
    objectGroup = argParser.add_argument_group("Object Detection")
    objectGroup.add_argument(
        "--obj_detect", action="store_true", help="Detect objects in the video"
    )

    # Miscellaneous options
    _addMiscOptions(argParser)

    args = argParser.parse_args()
    return argumentsChecker(args, outputPath)


def _addInterpolationOptions(argParser):
    interpolationGroup = argParser.add_argument_group("Interpolation")
    interpolationGroup.add_argument(
        "--interpolate", action="store_true", help="Interpolate the video"
    )
    interpolationGroup.add_argument(
        "--interpolate_factor", type=float, default=2, help="Interpolation factor"
    )

    interpolationMethods = [
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
    ]

    interpolationGroup.add_argument(
        "--interpolate_method",
        type=str,
        choices=interpolationMethods,
        default="rife",
        help="Interpolation method",
    )
    interpolationGroup.add_argument(
        "--slowmo",
        action="store_true",
        help="Enable slow motion interpolation, this will slow down the video instead of increasing the frame rate",
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
    interpolationGroup.add_argument(
        "--static_step",
        action="store_true",
        help="Force Static Timestep generation for Rife CUDA",
    )
    interpolationGroup.add_argument(
        "--interpolate_first",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        choices=[True, False],
        help="Switch back to the old approach where interpolated frames would instantly be written to the write queue",
    )


def _addUpscalingOptions(argParser):
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

    upscaleMethods = [
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
    ]

    upscaleGroup.add_argument(
        "--upscale_method",
        type=str,
        choices=upscaleMethods,
        default="shufflecugan",
        help="Upscaling method",
    )
    upscaleGroup.add_argument(
        "--custom_model", type=str, default="", help="Path to custom upscaling model"
    )


def _addDedupOptions(argParser):
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


def _addVideoProcessingOptions(argParser):
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

    restoreMethods = [
        "scunet",
        "nafnet",
        "dpir",
        "real-plksr",
        "anime1080fixer",
        "anime1080fixer-tensorrt",
        "anime1080fixer-directml",
        "fastlinedarken",
        "fastlinedarken-tensorrt",
    ]

    processingGroup.add_argument(
        "--restore_method",
        type=str,
        default="scunet",
        choices=restoreMethods,
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


def _addSegmentationOptions(argParser):
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


def _addSceneDetectionOptions(argParser):
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

    scenechangeMethods = [
        "maxxvit-tensorrt",
        "maxxvit-directml",
        "differential",
        "differential-tensorrt",
        "shift_lpips-tensorrt",
        "shift_lpips-directml",
    ]

    sceneGroup.add_argument(
        "--scenechange_method",
        type=str,
        default="maxxvit-directml",
        choices=scenechangeMethods,
        help="Scene change detection method",
    )
    sceneGroup.add_argument(
        "--scenechange_sens",
        type=float,
        default=50,
        help="Scene change detection sensitivity (0-100)",
    )


def _addDepthOptions(argParser):
    depthGroup = argParser.add_argument_group("Depth Estimation")
    depthGroup.add_argument(
        "--depth", action="store_true", help="Estimate the depth of the video"
    )

    depthMethods = [
        "small_v2",
        "base_v2",
        "large_v2",
        "small_v2-tensorrt",
        "base_v2-tensorrt",
        "large_v2-tensorrt",
        "small_v2-directml",
        "base_v2-directml",
        "large_v2-directml",
        "distill_small_v2",
        "distill_base_v2",
        "og_small_v2",
        "og_base_v2",
        "og_large_v2",
        "og_distill_small_v2",
        "og_distill_base_v2",
    ]

    depthGroup.add_argument(
        "--depth_method",
        type=str,
        choices=depthMethods,
        default="small_v2",
        help="Depth estimation method",
    )
    depthGroup.add_argument(
        "--depth_quality",
        type=str,
        choices=["low", "medium", "high"],
        default="low",
        help="This will determine the quality of the depth map, low is significantly faster but lower quality, only works with CUDA Depth Maps",
    )


def _addEncodingOptions(argParser):
    encodingGroup = argParser.add_argument_group("Encoding")

    encodeMethods = [
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
        "lossless",
        "lossless_nvenc",
    ]

    encodingGroup.add_argument(
        "--encode_method",
        type=str,
        choices=encodeMethods,
        default="x264",
        help="Encoding method",
    )
    encodingGroup.add_argument(
        "--custom_encoder", type=str, default="", help="Custom encoder settings"
    )


def _addMiscOptions(argParser):
    miscGroup = argParser.add_argument_group("Miscellaneous")
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
        choices=["8bit", "16bit"],
        help="Bit Depth of the raw pipe input to FFMPEG",
    )


def argumentsChecker(args, outputPath):
    if args.list_presets:
        from src.utils.presetLogic import listPresets

        listPresets()
        sys.exit()

    if args.preset:
        from src.utils.presetLogic import createPreset

        args = createPreset(args)

    logging.info("============== Version ==============")
    logging.info(f"TAS: {__version__}\n")

    logging.info("============== Arguments ==============")
    for arg, value in vars(args).items():
        if value not in [None, "", "none", False]:
            logging.info(f"{arg.upper()}: {value}")

    if not args.benchmark:
        from .checkSpecs import checkSystem

        checkSystem()

    if args.ae:
        logging.info("After Effects interface detected")
        cs.ADOBE = True

    logging.info("\n============== Arguments Checker ==============")
    _handleDependencies()

    if not os.path.exists(cs.FFMPEGPATH) or (
        args.realtime
        and not os.path.exists(cs.MPVPATH)
        or not os.path.exists(cs.FFPROBEPATH)
    ):
        from src.utils.getFFMPEG import getFFMPEG

        getFFMPEG(args.realtime)

    if args.realtime:
        print(yellow("Realtime preview enabled, this is experimental!"))

    if args.slowmo and not args.interpolate:
        logAndPrint(
            "Slow motion is enabled but interpolation is not, disabling slowmo",
            "yellow",
        )
        args.slowmo = False

    _handleDepthSettings(args)

    if args.benchmark and args.realtime:
        logging.error("Realtime preview is not supported in benchmark mode")
        args.realtime = False

    if args.offline != "none":
        _downloadOfflineModels(args)

    _configureProcessingSettings(args)

    # Check CUDA availability and adjust methods if needed
    _adjustMethodsBasedOnCuda(args)

    if args.custom_encoder:
        logging.info("Custom encoder specified, use with caution")

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
        sys.exit()

    return args


def _handleDependencies():
    cs.FFMPEGPATH = os.path.join(
        cs.MAINPATH,
        "ffmpeg",
        "ffmpeg.exe" if cs.SYSTEM == "Windows" else "ffmpeg",
    )

    cs.FFPROBEPATH = os.path.join(
        cs.MAINPATH,
        "ffmpeg",
        "ffprobe.exe" if cs.SYSTEM == "Windows" else "ffprobe",
    )

    cs.MPVPATH = os.path.join(
        cs.MAINPATH,
        "ffmpeg",
        "mpv.exe" if cs.SYSTEM == "Windows" else "mpv",
    )

    # Handling isNvidia present and what dependencies to install
    # First try import torch just to avoid doing this over and over again
    try:
        import torch
    except ImportError:
        logging.info("Torch not found, handling dependency installation")
        from src.utils.isCudaInit import detectNVidiaGPU
        from src.utils.dependencyHandler import installDependencies

        isNvidia, gpuName = detectNVidiaGPU()
        if isNvidia:
            logAndPrint(
                f"NVIDIA GPU detected: {gpuName}, installing dependencies for NVIDIA",
                "green",
            )
        else:
            logAndPrint(
                "No NVIDIA GPU detected, installing dependencies for CPU",
                "yellow",
            )

        success, message = installDependencies(isNvidia)

        if not success:
            logAndPrint(f"Failed to install dependencies: {message}", "red")

            sys.exit()
        else:
            logAndPrint(
                "Dependencies installed successfully",
                "green",
            )


def _handleDepthSettings(args):
    if args.depth:
        logging.info("Depth enabled, audio processing will be disabled")
        cs.AUDIO = False

    if args.depth_quality not in ["low"] and args.depth_method.split("-")[-1] in [
        "tensorrt",
        "directml",
    ]:
        logAndPrint(
            f"{args.depth_quality.upper()} depth estimation quality is incomaptible with tensorrt and directml, defaulting to low quality",
            "yellow",
        )
        args.depth_quality = "low"


def _downloadOfflineModels(args):
    from .downloadModels import downloadModels, modelsList

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


def _configureProcessingSettings(args):
    if args.slowmo:
        cs.AUDIO = False
        logging.info("Slow motion enabled, audio processing disabled")

    if args.static_step and isinstance(args.interpolate_factor, float):
        logging.info("Interpolate Factor is a float, static step will be disabled")
        args.static_step = False

    if args.dedup:
        cs.AUDIO = False
        logging.info("Deduplication enabled, audio processing disabled")

        if args.dedup_method in ["ssim", "ssim-cuda"]:
            args.dedup_sens = 1.0 - (args.dedup_sens / 1000)
        elif args.dedup_method in ["flownets"]:
            args.dedup_sens = args.dedup_sens / 100

        logging.info(
            f"New dedup sensitivity for {args.dedup_method} is: {args.dedup_sens}"
        )

    # Map of scene change methods to their sensitivity adjustment values
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


def _adjustMethodsBasedOnCuda(args):
    def adjustMethod(method, modelsList):
        base = method.lower().split("-")[0]
        directML = f"{base}-directml"
        if directML in modelsList:
            return directML
        newMethod = f"{base}-ncnn"
        if newMethod in modelsList:
            return newMethod
        return method

    from src.utils.isCudaInit import CudaChecker

    isCuda = CudaChecker()
    if not isCuda.cudaAvailable:
        from .downloadModels import modelsList

        availableModels = modelsList()
        methodAttributes = [
            "interpolate_method",
            "upscale_method",
            "segment_method",
            "scenechange_method",
            "depth_method",
            "restore_method",
        ]

        for attr in methodAttributes:
            currentMethod = getattr(args, attr)
            newMethod = adjustMethod(currentMethod, availableModels)
            if newMethod != currentMethod:
                logging.info(f"Adjusted {attr} from {currentMethod} to {newMethod}")
                setattr(args, attr, newMethod)
            else:
                logging.info(
                    f"No adjustment for {attr} ({currentMethod} remains unchanged)"
                )


def processURL(args, outputPath):
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

        VideoDownloader(args.input, tempOutput, args.encode_method, args.custom_encoder)
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
