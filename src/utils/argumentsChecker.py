"""
Command Line Argument Parser

Handles parsing and validation of command line arguments for The Anime Scripter.
Provides comprehensive argument definitions and validation logic.
"""

import os
import logging
import sys
import argparse
import shutil
import src.constants as cs

from rich_argparse import RichHelpFormatter
from src.version import __version__
from .inputOutputHandler import generateOutputName
from src.utils.logAndPrint import logAndPrint
from src.utils.dependencyHandler import installDependencies
from src.utils.getFFMPEG import remove_readonly


def isAnyOtherProcessingMethodEnabled(args):
    """
    Check if any video processing operations are enabled.

    Args:
        args: Parsed command line arguments

    Returns:
        bool: True if any processing method is enabled
    """
    return any(
        [
            args.interpolate,
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
    """
    Convert string argument to boolean value.

    Args:
        arg: String or boolean argument

    Returns:
        bool: Converted boolean value

    Raises:
        argparse.ArgumentTypeError: If argument cannot be converted to boolean
    """
    if isinstance(arg, bool):
        return arg
    if arg.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif arg.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def _loadJsonConfig(args, parser):
    """
    Load configuration from JSON file and merge with command line arguments.
    CLI arguments take precedence over JSON values.

    Args:
        args: Parsed command line arguments
        parser: The ArgumentParser instance

    Returns:
        argparse.Namespace: Updated arguments with JSON values
    """
    import json

    cliProvidedArgs = set()
    for arg in sys.argv[1:]:
        if arg.startswith("--") and arg != "--json":
            argName = arg[2:].replace("-", "_")
            cliProvidedArgs.add(argName)

    if len(cliProvidedArgs) > 0:
        logAndPrint(
            "Cannot use --json with other command line arguments. Use --json alone.",
            "red",
        )
        sys.exit()

    jsonPath = os.path.abspath(args.json)

    if not os.path.exists(jsonPath):
        logAndPrint(f"JSON config file not found: {jsonPath}", "red")
        sys.exit()

    try:
        with open(jsonPath, "r", encoding="utf-8") as f:
            jsonConfig = json.load(f)
    except json.JSONDecodeError as e:
        logAndPrint(f"Invalid JSON format in config file: {e}", "red")
        sys.exit()
    except Exception as e:
        logAndPrint(f"Error reading JSON config: {e}", "red")
        sys.exit()

    defaults = {}
    for action in parser._actions:
        if action.dest not in ["help", "version", "json"]:
            defaults[action.dest] = action.default

    loadedKeys = set()
    for key, value in jsonConfig.items():
        if key == "json":
            continue

        if hasattr(args, key):
            currentValue = getattr(args, key)
            defaultValue = defaults.get(key)

            if currentValue == defaultValue:
                setattr(args, key, value)
                logging.info(f"Loaded from JSON: {key} = {value}")
            loadedKeys.add(key)
        else:
            logging.warning(f"Unknown option in JSON config: {key}")

    return args, loadedKeys


def createParser(outputPath):
    """
    Create and configure the command line argument parser.

    Args:
        outputPath (str): Default output directory path

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    argParser = argparse.ArgumentParser(
        description="The Anime Scripter CLI Tool",
        usage="main.py [options]",
        formatter_class=RichHelpFormatter,
    )

    # Basic options
    generalGroup = argParser.add_argument_group("General")
    generalGroup.add_argument("-v", "--version", action="version", version=__version__)
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
        "--json",
        type=str,
        help="Path to JSON configuration file with processing options",
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
        "--decode_method",
        type=str,
        choices=["cpu", "nvdec"],
        default="cpu",
        help="Decoding backend to use, default is cpu. 'nvdec' requires an NVIDIA GPU with NVDEC support.",
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
        "--compile_mode",
        type=str,
        choices=["default", "max", "max-graphs"],
        default="default",
        help="[EXPERIMENTAL] Enable PyTorch compilation for CUDA models to improve performance. "
        "Note: Only compatible with CUDA workflows and may cause compatibility issues with some models. "
        "Increases startup time and memory usage. "
        "'default' uses standard CudaGraph workflow without compilation, "
        "'max' uses 'max-autotune-no-cudagraphs' mode, "
        "'max-graphs' uses 'max-autotune-no-cudagraphs' with fullGraph=True. "
        "Both 'max' options disable CudaGraphs, which may reduce performance at lower resolutions.",
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

    objectGroup.add_argument(
        "--obj_detect_method",
        type=str,
        default="yolov9_small-directml",
        choices=[
            "yolov9_small-directml",
            "yolov9_medium-directml",
            "yolov9_large-directml",
            "yolov9_small-openvino",
            "yolov9_medium-openvino",
            "yolov9_large-openvino",
            # TO:DO: ADD TENSORRT Backend
        ],
    )

    objectGroup.add_argument(
        "--obj_detect_disable_annotations",
        type=str2bool,
        default=False,
        help="Disable class labels and confidence percentages on detection boxes (default: False)",
    )

    # Miscellaneous options
    _addMiscOptions(argParser)

    args = argParser.parse_args()
    return argumentsChecker(args, outputPath, argParser)


def _addInterpolationOptions(argParser):
    interpolationGroup = argParser.add_argument_group("Interpolation")
    interpolationGroup.add_argument(
        "--interpolate", action="store_true", help="Interpolate the video"
    )
    interpolationGroup.add_argument(
        "--interpolate_factor", type=float, default=2, help="Interpolation factor"
    )

    interpolationMethods = [
        "distildrba",
        "distildrba-lite",
        "distildrba-tensorrt",
        "distildrba-lite-tensorrt",
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
        "rife4.25-depth",
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
        "rife4.6-tensorrt",
        "rife4.6-directml",
        "rife4.6-openvino",
    ]

    interpolationGroup.add_argument(
        "--interpolate_method",
        type=str,
        choices=interpolationMethods,
        default="rife4.6",
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
        choices=[2, 3, 4],
        default=2,
        help="Upscaling factor (minimum 2)",
    )

    upscaleMethods = [
        "shufflecugan",
        "fallin_soft",
        "fallin_soft-tensorrt",
        "fallin_soft-directml",
        "fallin_strong",
        "fallin_strong-tensorrt",
        "fallin_strong-directml",
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
        "rtmosr",
        "rtmosr-tensorrt",
        "rtmosr-directml",
        "saryn",
        "saryn-tensorrt",
        "saryn-directml",
        "animesr",
        "animesr-tensorrt",
        "animesr-directml",
        "animesr-openvino",
        "compact-openvino",
        "ultracompact-openvino",
        "superultracompact-openvino",
        "span-openvino",
        "open-proteus-openvino",
        "aniscale2-openvino",
        "shufflespan-openvino",
        "rtmosr-openvino",
        "saryn-openvino",
        "fallin_soft-openvino",
        "fallin_strong-openvino",
        "gauss",
        "gauss-tensorrt",
        "gauss-directml",
        "gauss-openvino",
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
        choices=[
            "ssim",
            "mse",
            "ssim-cuda",
            "mse-cuda",
            "flownets",
            "vmaf",
            "vmaf-cuda",
        ],
        help="Deduplication method",
    )
    dedupGroup.add_argument(
        "--dedup_sens", type=float, default=35, help="Deduplication sensitivity"
    )
    dedupGroup.add_argument(
        "--smooth_dedup",
        action="store_true",
        help="Smooth deduplication, this will remove duplicates while also generating new frames to make the video smoother, this is experimental and may not work well with all videos, use --interpolate_method to set the interpolation method",
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
        "scunet-tensorrt",
        "scunet-directml",
        "scunet-openvino",
        "nafnet",
        "dpir",
        "real-plksr",
        "anime1080fixer",
        "anime1080fixer-tensorrt",
        "anime1080fixer-directml",
        "anime1080fixer-openvino",
        "fastlinedarken",
        "fastlinedarken-tensorrt",
        "gater3",
        "gater3-directml",
        "gater3-openvino",
        "deh264_real",
        "deh264_real-tensorrt",
        "deh264_real-directml",
        "deh264_real-openvino",
        "deh264_span",
        "deh264_span-tensorrt",
        "deh264_span-directml",
        "deh264_span-openvino",
        "hurrdeblur",
        "hurrdeblur-tensorrt",
        "hurrdeblur-directml",
        "hurrdeblur-openvino",
        "linethinner-lite",
        "linethinner-medium",
        "linethinner-heavy",
        "linethinner-lite-cuda",
        "linethinner-medium-cuda",
        "linethinner-heavy-cuda",
    ]

    processingGroup.add_argument(
        "--restore_method",
        type=str,
        nargs="+",
        default=["anime1080fixer"],
        choices=restoreMethods,
        help="Denoising method(s), can specify multiple for chaining",
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
        "--output_scale",
        type=str,
        default="",
        help="Output resolution in WIDTHxHEIGHT format (e.g., 2560x1440)",
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


def _addDepthOptions(argParser):
    depthGroup = argParser.add_argument_group("Depth Estimation")
    depthGroup.add_argument(
        "--depth", action="store_true", help="Estimate the depth of the video"
    )

    depthMethods = [
        "small_v2",
        "base_v2",
        "large_v2",
        "giant_v2",
        "distill_small_v2",
        "distill_base_v2",
        "distill_large_v2",
        "og_small_v2",
        "og_base_v2",
        "og_large_v2",
        "og_giant_v2",
        "og_distill_small_v2",
        "og_distill_base_v2",
        "og_distill_large_v2",
        "og_video_small_v2",
        "small_v2-tensorrt",
        "base_v2-tensorrt",
        "large_v2-tensorrt",
        "distill_small_v2-tensorrt",
        "distill_base_v2-tensorrt",
        "distill_large_v2-tensorrt",
        "small_v2-directml",
        "base_v2-directml",
        "large_v2-directml",
        "distill_small_v2-directml",
        "distill_base_v2-directml",
        "distill_large_v2-directml",
        "og_small_v2-tensorrt",
        "og_base_v2-tensorrt",
        "og_large_v2-tensorrt",
        "og_distill_small_v2-tensorrt",
        "og_distill_base_v2-tensorrt",
        "og_distill_large_v2-tensorrt",
        "small_v3",
        "base_v3",
        "large_v3",
        "giant_v3",
        "small_v3-directml",
        "base_v3-directml",
        "large_v3-directml",
        "giant_v3-directml",
        "small_v3-tensorrt",
        "base_v3-tensorrt",
        "large_v3-tensorrt",
        "giant_v3-tensorrt",
        "small_v2-openvino",
        "base_v2-openvino",
        "large_v2-openvino",
        "distill_small_v2-openvino",
        "distill_base_v2-openvino",
        "distill_large_v2-openvino",
        "og_small_v2-openvino",
        "og_base_v2-openvino",
        "og_large_v2-openvino",
        "small_v3-openvino",
        "base_v3-openvino",
        "large_v3-openvino",
        "giant_v3-openvino",
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
        "png",
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
        "--profile",
        action="store_true",
        help="Enable torch.profiler to analyze GPU/CPU performance bottlenecks",
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
        type=str,
        default="",
        help="Notify if script is run from After Effects interface",
    )
    miscGroup.add_argument(
        "--bit_depth",
        type=str,
        default="8bit",
        choices=["8bit", "16bit"],
        help="Bit Depth of the raw pipe input to FFMPEG",
    )
    miscGroup.add_argument(
        "--download_requirements",
        action="store_true",
        help="Download all required libraries for the script, only used for Adobe Edition",
    )
    miscGroup.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove unused libraries from the script, in case if there were migrations or changes in the script",
    )


def _autoEnableParentFlags(args):
    """
    Automatically enable parent flags when *_method arguments are provided.
    For example: --upscale_method shufflecugan will auto-enable --upscale

    Args:
        args: Parsed command line arguments
    """
    logging.info(f"[DEBUG] hasattr(args, '_json_keys'): {hasattr(args, '_json_keys')}")
    if hasattr(args, "_json_keys"):
        logging.info(f"[DEBUG] args._json_keys: {args._json_keys}")

    methodToFlagMapping = {
        "interpolate_method": ("interpolate", "rife4.6"),
        "interpolate_factor": ("interpolate", 2.0),
        "upscale_method": ("upscale", "shufflecugan"),
        "upscale_factor": ("upscale", 2),
        "dedup_method": ("dedup", "ssim"),
        "dedup_sens": ("dedup", 35.0),
        "restore_method": ("restore", ["anime1080fixer"]),
        "segment_method": ("segment", "anime"),
        "depth_method": ("depth", "small_v2"),
        "obj_detect_method": ("obj_detect", "yolov9_small-directml"),
        "resize_factor": ("resize", 2),
        "output_scale": ("resize", ""),
    }

    cliProvided = set()
    for a in sys.argv[1:]:
        if a.startswith("--"):
            name = a[2:].split("=")[0]
            cliProvided.add(name)

    for methodArg, (parentFlag, defaultValue) in methodToFlagMapping.items():
        if hasattr(args, methodArg):
            currentValue = getattr(args, methodArg)

            providedOnCLI = (
                methodArg in cliProvided or methodArg.replace("_", "-") in cliProvided
            )

            isExplicitlyProvided = providedOnCLI or (
                hasattr(args, "_json_keys") and methodArg in args._json_keys
            )

            if methodArg == "interpolate_method":
                logging.info(
                    f"[DEBUG] interpolate_method - providedOnCLI: {providedOnCLI}, isExplicitlyProvided: {isExplicitlyProvided}"
                )

            if isExplicitlyProvided:
                if not getattr(args, parentFlag):
                    setattr(args, parentFlag, True)
                    logging.info(
                        f"Auto-enabling --{parentFlag} because --{methodArg} was provided"
                    )
            else:
                if currentValue != defaultValue:
                    if not getattr(args, parentFlag):
                        setattr(args, parentFlag, True)
                        logging.info(
                            f"Auto-enabling --{parentFlag} because {methodArg} differs from default"
                        )


def argumentsChecker(args, outputPath, parser):
    if args.list_presets:
        from src.utils.presetLogic import listPresets

        listPresets()
        sys.exit()

    if args.preset:
        from src.utils.presetLogic import createPreset

        args = createPreset(args)

    if args.json:
        args, loadedKeys = _loadJsonConfig(args, parser)
        args._json_keys = loadedKeys

    _autoEnableParentFlags(args)

    if args.download_requirements:
        _handleDependencies(args)

        logAndPrint(
            "All required libraries have been downloaded, you can now run the script freely.",
            "green",
        )
        sys.exit()

    if args.cleanup:
        from src.utils.dependencyHandler import uninstallDependencies
        from src.utils.isCudaInit import detectNVidiaGPU, detectGPUArchitecture

        isNvidia = detectNVidiaGPU()
        supportsCuda = False
        if isNvidia:
            supportsCuda, _, _ = detectGPUArchitecture()
        if cs.SYSTEM == "Windows":
            extension = (
                "extra-requirements-windows.txt"
                if supportsCuda
                else "extra-requirements-windows-lite.txt"
            )
        else:  # Linux and other systems
            extension = (
                "extra-requirements-linux.txt"
                if supportsCuda
                else "extra-requirements-linux-lite.txt"
            )
        success, message = uninstallDependencies(extension=extension)

        logging.info(message)

        if success:
            logAndPrint(
                "Unused libraries have been removed, you can now run the script without the --cleanup argument",
                "green",
            )
            sys.exit()
        else:
            logAndPrint(
                "Failed to remove unused libraries, please check the logs for more details",
                "red",
            )
            logging.error("Failed to remove unused libraries")
            print(message)

    logging.info("============== Version ==============")
    logging.info(f"TAS: {__version__}\n")

    cliArgs = sys.argv[1:]
    providedArgs = set()
    for i, arg in enumerate(cliArgs):
        if arg.startswith("--"):
            argName = arg[2:]
            providedArgs.add(argName)

    logging.info("============== Arguments ==============")
    for arg, value in vars(args).items():
        if arg in providedArgs and value not in [None, "", "none"]:
            logging.info(f"{arg.upper()}: {value}")

    if not args.benchmark:
        from .checkSpecs import checkSystem

        checkSystem()

    if args.preview and args.benchmark:
        logAndPrint(
            "Preview cannot be enabled in benchmark mode, disabling preview",
            "yellow",
        )
        args.preview = False

    if args.ae:
        logging.info("After Effects interface detected")
        cs.ADOBE = True
        from src.utils.aeComms import startServerInThread

        try:
            startServerInThread(
                host=args.ae,
            )
        except Exception as e:
            logging.error(f"Failed to start AE comms server: {e}")
            logAndPrint(
                "Failed to start AE comms server, please check the logs for more details",
                "red",
            )

    logging.info("\n============== Arguments Checker ==============")
    _handleDependencies(args)

    if args.slowmo and not args.interpolate:
        logAndPrint(
            "Slow motion is enabled but interpolation is not, disabling slowmo",
            "yellow",
        )
        args.slowmo = False

    _handleDepthSettings(args)

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

    if args.encode_method in ["gif", "png"]:
        logging.info(
            f"Encoding method is set to {args.encode_method}, disabling audio processing"
        )
        cs.AUDIO = False

    if not args.input:
        logging.error("No input specified")
        sys.exit()
    elif args.input.startswith(("http", "www")):
        processURL(args, outputPath)
    elif args.input.lower().endswith((".png", ".jpg", ".jpeg")):
        if "%" in args.input:
            logging.info(f"Image sequence pattern detected: {args.input}")
            args.input = os.path.abspath(args.input)
            cs.AUDIO = False
        else:
            raise Exception(
                "Single image input is not supported. For image sequences, use a pattern like 'frames_%05d.png' or provide a folder containing PNG files."
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

    if args.output_scale:
        try:
            width, height = args.output_scale.split("x")
            args.output_scale_width = int(width)
            args.output_scale_height = int(height)
            if args.output_scale_width <= 0 or args.output_scale_height <= 0:
                raise ValueError("Width and height must be positive integers")
            logging.info(
                f"Output scale set to {args.output_scale_width}x{args.output_scale_height}"
            )
        except (ValueError, AttributeError):
            logAndPrint(
                f"Invalid output_scale format: {args.output_scale}. Expected format: WIDTHxHEIGHT (e.g., 2560x1440)"
            )
            sys.exit()
    else:
        args.output_scale_width = None
        args.output_scale_height = None

    if args.upscale and hasattr(args, "upscale_factor"):
        try:
            if int(args.upscale_factor) < 2:
                logAndPrint(
                    "Upscale factor must be at least 2 when --upscale is enabled; defaulting to 2",
                    "yellow",
                )
                logging.info(
                    f"Adjusted upscale_factor from {args.upscale_factor} to 2 to satisfy minimum requirement"
                )
                args.upscale_factor = 2
        except Exception:
            logAndPrint(
                "Invalid upscale_factor provided; defaulting to 2",
                "yellow",
            )
            logging.info("Invalid upscale_factor value encountered; set to 2")
            args.upscale_factor = 2

    logging.info(
        f"[DEBUG] Before processing check - args.interpolate: {args.interpolate}"
    )

    if not isAnyOtherProcessingMethodEnabled(args):
        logAndPrint(
            "No processing methods specified, make sure to use enabler arguments like --upscale, --interpolate, etc.",
            "red",
        )
        sys.exit()

    return args


def _handleDependencies(args):
    legacyFFMPEG = os.path.join(cs.MAINPATH, "ffmpeg")
    if os.path.isdir(legacyFFMPEG):
        try:
            shutil.rmtree(legacyFFMPEG, onerror=remove_readonly)
            logging.info(f"Removed legacy FFmpeg folder: {legacyFFMPEG}")
        except Exception as e:
            logging.warning(f"Failed to remove legacy FFmpeg folder: {e}")

    ffmpegSharedDir = os.path.join(cs.MAINPATH, "ffmpeg_shared")

    cs.FFMPEGPATH = os.path.join(
        ffmpegSharedDir,
        "ffmpeg.exe" if cs.SYSTEM == "Windows" else "ffmpeg",
    )

    cs.FFPROBEPATH = os.path.join(
        ffmpegSharedDir,
        "ffprobe.exe" if cs.SYSTEM == "Windows" else "ffprobe",
    )

    if cs.SYSTEM == "Windows":
        if "ffprobe.exe" not in os.environ["PATH"]:
            os.environ["PATH"] += os.pathsep + os.path.dirname(cs.FFPROBEPATH)
    else:
        if "ffprobe" not in os.environ["PATH"]:
            os.environ["PATH"] += os.pathsep + os.path.dirname(cs.FFPROBEPATH)

    if not os.path.exists(cs.FFMPEGPATH) or not os.path.exists(cs.FFPROBEPATH):
        from src.utils.getFFMPEG import getFFMPEG

        getFFMPEG()
    else:
        from src.utils.getFFMPEG import logFfmpegVersion

        logFfmpegVersion(cs.FFMPEGPATH)

    if cs.SYSTEM == "Windows":
        ffmpeg_dir = os.path.dirname(cs.FFMPEGPATH)
        if os.path.exists(ffmpeg_dir):
            try:
                os.add_dll_directory(ffmpeg_dir)
                logging.info(f"Added FFmpeg directory to DLL search path: {ffmpeg_dir}")
            except Exception as e:
                logging.warning(f"Failed to add FFmpeg to DLL search path: {e}")

    try:
        from src.utils.isCudaInit import detectNVidiaGPU, detectGPUArchitecture

        isNvidia = detectNVidiaGPU()
        supportsCuda = False
        if isNvidia:
            supportsCuda, _, _ = detectGPUArchitecture()
    except ImportError:
        isNvidia = False
        supportsCuda = False

    if cs.SYSTEM == "Windows":
        extension = (
            "extra-requirements-windows.txt"
            if supportsCuda
            else "extra-requirements-windows-lite.txt"
        )
    else:
        extension = (
            "extra-requirements-linux.txt"
            if supportsCuda
            else "extra-requirements-linux-lite.txt"
        )

    requirementsPath = os.path.join(cs.WHEREAMIRUNFROM, extension)

    try:
        from src.utils.dependencyHandler import DependencyChecker

        checker = DependencyChecker()

        if checker.needsUpdate(requirementsPath):
            logAndPrint("Dependencies need updating...", "yellow")
            success, message = installDependencies(extension, isNvidia=isNvidia)
            if not success:
                logAndPrint(message, "red")
                sys.exit()
            else:
                logAndPrint(message, "green")
                checker.updateCache(requirementsPath)
        else:
            logging.info("Dependencies are up to date")

    except ImportError:
        success, message = installDependencies(extension, isNvidia=isNvidia)
        if not success:
            logAndPrint(message, "red")
            sys.exit()
        else:
            logAndPrint(message, "green")
            from src.utils.dependencyHandler import DependencyChecker

            checker = DependencyChecker()
            checker.updateCache(requirementsPath)


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

    if args.depth_method in ["giant_v2", "og_giant_v2"]:
        logAndPrint(
            f"{args.depth_method} is a very large model and may cause out of memory errors on GPUs with less than 16GB of VRAM",
            "yellow",
        )
        if args.half:
            logAndPrint(
                "Half precision does not work with giant models, switching to full precision",
            )
            args.half = False


def _downloadOfflineModels(args):
    from .downloadModels import downloadModels, modelsList

    logAndPrint(
        f"Offline mode enabled, downloading {args.offline} model(s)...", "green"
    )

    options = modelsList() if args.offline == ["all"] else args.offline
    for option in options:
        for precision in [True, False]:
            try:
                downloadModels(option.lower(), half=precision)
            except Exception:
                logging.error(
                    f"Failed to download model: {option} with precision: {'fp16' if precision else 'fp32'}"
                )
    logAndPrint(
        f"Offline model(s) {', '.join(options)} downloaded successfully!", "green"
    )


def _configureProcessingSettings(args):
    if args.slowmo:
        cs.AUDIO = False
        logging.info("Slow motion enabled, audio processing disabled")

    if args.static_step and isinstance(args.interpolate_factor, float):
        logging.info("Interpolate Factor is a float, static step will be disabled")
        args.static_step = False

    if args.dedup:
        if not args.smooth_dedup:
            cs.AUDIO = False
            logging.info(
                "Deduplication enabled and smooth dedup disabled, audio processing disabled"
            )

        if args.dedup_method in ["ssim", "ssim-cuda"]:
            args.dedup_sens = 1.0 - (args.dedup_sens / 1000)
        elif args.dedup_method in ["flownets"]:
            args.dedup_sens = args.dedup_sens / 100

        logging.info(
            f"New dedup sensitivity for {args.dedup_method} is: {args.dedup_sens}"
        )

    if args.sharpen:
        args.sharpen_sens = args.sharpen_sens / 100
        logging.info(f"New sharpen sensitivity is: {args.sharpen_sens}")

    if args.autoclip:
        # For some reason, the sensitivity is inverted in the autoclip method, could be some hard math that I don't understand
        # but for now, we will just invert it to make it work as expected
        args.autoclip_sens = float(100 - args.autoclip_sens)
        logging.info(f"New autoclip sensitivity is: {args.autoclip_sens}")

    if args.compile_mode != "default":
        logging.info(
            f"Pytorch Compile mode is set to {args.compile_mode}, this will increase startup time and memory usage and may lead to instability with some models"
        )


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

    from src.utils.isCudaInit import CudaChecker, detectGPUArchitecture

    isCuda = CudaChecker()

    # Check if GPU architecture supports modern CUDA features
    needsFallback = False
    if isCuda.cudaAvailable:
        isModernGPU, gpuName, computeCap = detectGPUArchitecture()
        if not isModernGPU:
            logAndPrint(
                f"Detected {gpuName} (compute capability: {computeCap}). "
                f"This GPU may not support modern CUDA kernels. "
                f"Automatically switching to DirectML/NCNN backends for compatibility.",
                "yellow",
            )
            needsFallback = True
    else:
        needsFallback = True

    if needsFallback:
        from .downloadModels import modelsList

        availableModels = modelsList()
        methodAttributes = [
            "interpolate_method",
            "upscale_method",
            "segment_method",
            "depth_method",
            "restore_method",
            "dedup_method",
            "obj_detect_method",
        ]

        for attr in methodAttributes:
            currentMethod = getattr(args, attr)

            if attr == "restore_method" and isinstance(currentMethod, list):
                adjusted = []
                for method in currentMethod:
                    if any(
                        backend in method.lower()
                        for backend in ["-directml", "-ncnn", "-tensorrt"]
                    ):
                        logging.info(
                            f"{attr} method {method} already using non-default backend"
                        )
                        adjusted.append(method)
                        continue

                    newMethod = adjustMethod(method, availableModels)
                    if newMethod != method:
                        logging.info(
                            f"Adjusted {attr} method from {method} to {newMethod}"
                        )
                    adjusted.append(newMethod)
                setattr(args, attr, adjusted)
            else:
                if any(
                    backend in currentMethod.lower()
                    for backend in ["-directml", "-ncnn", "-tensorrt"]
                ):
                    logging.info(
                        f"{attr} already using non-default backend: {currentMethod}"
                    )
                    continue

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
                outputFolder, generateOutputName(args, args.input)
            )
        elif os.path.isdir(args.output):
            outputFolder = args.output
            os.makedirs(os.path.join(outputFolder), exist_ok=True)
        else:
            outputFolder = os.path.dirname(args.output)
            os.makedirs(os.path.join(outputFolder), exist_ok=True)

        tempOutput = os.path.join(outputFolder, generateOutputName(args, args.input))

        VideoDownloader(args.input, tempOutput, args.encode_method, args.custom_encoder)
        logAndPrint(
            f"Video downloaded successfully to {tempOutput}",
            "green",
        )

        if not isAnyOtherProcessingMethodEnabled(args):
            if tempOutput != args.output:
                os.rename(tempOutput, args.output)
                logging.info(f"Renamed output to: {args.output}")
                sys.exit()

        args.input = str(tempOutput)
        logging.info(f"New input path: {args.input}")
    else:
        logging.error(
            "URL is invalid or not a YouTube URL, please check the URL and try again"
        )
        sys.exit()
