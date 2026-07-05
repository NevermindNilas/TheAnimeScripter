import logging
import os
import sys

import src.constants as cs
from src.cli.config import CliConfig
from src.cli.startup import _handleDependencies, _promptDownloadRequirementsSelection
from src.cli.validation import CliValidationError, applyRuntimeValidation
from src.infra.logAndPrint import logAndPrint
from src.io.inputNormalization import InputNormalizationError, normalizeInputArgs


def isAnyOtherProcessingMethodEnabled(args):
    return any(
        [
            args.interpolate,
            args.upscale,
            args.segment,
            args.restore,
            args.stabilize,
            args.resize,
            args.dedup,
            args.depth,
            args.autoclip,
            args.obj_detect,
            args.moblur,
        ]
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

    if args.depth_norm and "video" in args.depth_method:
        logAndPrint(
            "Depth normalization is not compatible with video depth methods, disabling depth_norm",
            "yellow",
        )
        args.depth_norm = False

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

    if args.autoclip:
        if args.autoclip_method == "pyscenedetect":
            args.autoclip_sens = float(100 - args.autoclip_sens)
        else:
            args.autoclip_sens = float(1.0 - (args.autoclip_sens / 100.0))
        logging.info(
            f"New autoclip sensitivity for {args.autoclip_method} is: {args.autoclip_sens}"
        )

    if args.compile_mode != "default":
        logging.info(
            f"Pytorch Compile mode is set to {args.compile_mode}, this will increase startup time and memory usage and may lead to instability with some models"
        )


def _adjustMethodsBasedOnCuda(args, availableModels=None):
    supportsCuda = getattr(args, "supportsCuda", None)

    if supportsCuda is None:
        from src.infra.isCudaInit import CudaChecker, detectGPUArchitecture

        isCuda = CudaChecker()

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
    else:
        needsFallback = not supportsCuda

    if needsFallback:
        from src.infra.backendFallback import applyBackendFallbacks

        if availableModels is None:
            from src.model.registry import modelsList

            availableModels = modelsList()

        applyBackendFallbacks(
            args,
            availableModels,
            preferMps=cs.SYSTEM == "Darwin",
        )


def _downloadOfflineModels(args):
    from src.model.download import downloadModels
    from src.model.registry import modelsList

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


def prepareRuntimeArgs(args, outputPath, parser):
    from src.version import __version__

    args.png_passthrough = False
    args.single_image_input = False

    if args.list_presets:
        from src.server.presetLogic import listPresets

        listPresets()
        sys.exit()

    if args.list_methods is not None:
        from src.cli.parser import _listMethods

        sys.exit(_listMethods(parser, args.list_methods))

    if args.preset:
        from src.server.presetLogic import createPreset

        args = createPreset(args)

    cliConfig = CliConfig.fromArgs(args, parser, sys.argv[1:])
    args = cliConfig.args

    if args.download_requirements is not None:
        from src.infra.dependencyHandler import DependencyChecker

        _handleDependencies(args)
        selectedProfile = args.download_requirements.strip().lower()
        if not selectedProfile:
            selectedProfile = _promptDownloadRequirementsSelection()

        checker = DependencyChecker()
        if not checker.installProfile(selectedProfile):
            sys.exit(1)

        logAndPrint(
            "All required libraries have been downloaded, you can now run the script freely.",
            "green",
        )
        sys.exit()

    if args.cleanup:
        from src.infra.dependencyHandler import (
            DependencyChecker,
            getDependencyProfile,
            getRequirementsFileForProfile,
            uninstallDependencies,
        )

        checker = DependencyChecker()
        storedProfile = checker.loadStoredProfile()

        if storedProfile:
            try:
                extension = getRequirementsFileForProfile(storedProfile)
            except ValueError:
                storedProfile = None

        if not storedProfile:
            from src.infra.isCudaInit import detectGPUArchitecture, detectNVidiaGPU

            isNvidia = detectNVidiaGPU()
            supportsCuda = False
            if isNvidia:
                supportsCuda, _, _ = detectGPUArchitecture()
            extension = getRequirementsFileForProfile(
                getDependencyProfile(cs.SYSTEM, supportsCuda)
            )

        success, message = uninstallDependencies(extension=extension)
        checker.clearCache()

        logging.info(message)

        if success:
            logAndPrint(
                "Dependencies from the selected runtime profile were uninstalled.",
                "green",
            )
            sys.exit()
        else:
            logAndPrint(
                "Failed to uninstall dependencies, please check the logs for more details",
                "red",
            )
            logging.error("Failed to uninstall dependencies")
            print(message)

    logging.info("============== Version ==============")
    logging.info(f"TAS: {__version__}\n")

    logging.info("============== Arguments ==============")
    for arg, value in vars(args).items():
        if arg in cliConfig.providedOptions and value not in [None, "", "none"]:
            logging.info(f"{arg.upper()}: {value}")

    if not args.benchmark:
        from src.infra.checkSpecs import checkSystem

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
        from src.server.aeComms import startServerInThread

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

    _adjustMethodsBasedOnCuda(args)

    if args.custom_encoder:
        logging.info("Custom encoder specified, use with caution")

    if args.bit_depth == "16bit" and args.segment:
        logging.error(
            "16bit input is not supported with segmentation, defaulting to 8bit"
        )
        args.bit_depth = "8bit"

    if args.output:
        outDir = os.path.dirname(os.path.abspath(args.output))
        if outDir and not os.path.exists(outDir):
            os.makedirs(outDir, exist_ok=True)

    if args.encode_method in ["gif", "png"]:
        logging.info(
            f"Encoding method is set to {args.encode_method}, disabling audio processing"
        )
        cs.AUDIO = False

    try:
        shouldContinue = normalizeInputArgs(
            args,
            outputPath,
            isAnyOtherProcessingMethodEnabled(args),
        )
    except InputNormalizationError as e:
        logging.error(str(e))
        logAndPrint(str(e), "red")
        sys.exit()

    if not shouldContinue:
        sys.exit()

    try:
        warning = applyRuntimeValidation(args)
    except CliValidationError as e:
        logAndPrint(str(e), "red")
        sys.exit()

    if args.output_scale_width and args.output_scale_height:
        logging.info(
            f"Output scale set to {args.output_scale_width}x{args.output_scale_height}"
        )

    if warning:
        logAndPrint(warning, "yellow")
        logging.info(warning)

    logging.info(
        f"[DEBUG] Before processing check - args.interpolate: {args.interpolate}"
    )

    if not isAnyOtherProcessingMethodEnabled(args) and not args.png_passthrough:
        logAndPrint(
            "No processing methods specified, make sure to use enabler arguments like --upscale, --interpolate, etc.",
            "red",
        )
        sys.exit()

    return args
