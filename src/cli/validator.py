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


# The bare rife aliases are just names for 4.22 -- that is what modelsMap()
# returns for "rife" and what the arch tables load. Left un-normalized they also
# resolve their own weights/rife*/ download folder and, on TensorRT, a second
# engine cache entry keyed off that path, so picking the alias cost a duplicate
# weight download and a duplicate multi-minute engine build for a byte-identical
# engine. Canonicalize once, here, rather than in each backend.
_BARE_RIFE_ALIASES = {
    "rife": "rife4.22",
    "rife-ncnn": "rife4.22-ncnn",
    "rife-tensorrt": "rife4.22-tensorrt",
    "rife-mps": "rife4.22-mps",
}


def _normalizeMethodAliases(args):
    for attribute in ("interpolate_method", "moblur_method"):
        current = getattr(args, attribute, None)
        canonical = _BARE_RIFE_ALIASES.get(current)
        if canonical is not None:
            logging.info(f"{attribute} '{current}' is an alias for '{canonical}'")
            setattr(args, attribute, canonical)


def _handleDepthSettings(args):
    if args.depth:
        logging.info("Depth enabled, audio processing will be disabled")
        cs.AUDIO = False

    # "openvino" belongs here too: it is not a separate backend, it is a
    # provider branch inside the same DepthDirectMLV2 class the "-directml"
    # methods use (src/factories/standalone.py), so it has the same constraint.
    # The --depth_batch clamp below already listed it.
    backend = args.depth_method.split("-")[-1]
    if args.depth_quality not in ["low"] and backend in [
        "tensorrt",
        "directml",
        "openvino",
    ]:
        logAndPrint(
            f"{args.depth_quality.upper()} depth estimation quality is incompatible "
            f"with the {backend} backend, defaulting to low quality",
            "yellow",
        )
        args.depth_quality = "low"

    # Normalize once so every later read is safe, including for args objects
    # built by other entrypoints that never define depth_batch.
    args.depth_batch = max(1, int(getattr(args, "depth_batch", 1)))

    if args.depth_batch > 1:
        backend = args.depth_method.split("-")[-1]
        if "video" in args.depth_method:
            logAndPrint(
                "--depth_batch is not supported for video depth methods, using 1",
                "yellow",
            )
            args.depth_batch = 1
        elif backend in ["directml", "openvino"]:
            logAndPrint(
                f"--depth_batch is not implemented for the {backend} depth backend, "
                f"using 1",
                "yellow",
            )
            args.depth_batch = 1

    # Strip "-mps": giant_v2-mps and og_giant_v2-mps are the same models and
    # equally large, but the bare list let them keep --half.
    if args.depth_method.removesuffix("-mps") in ["giant_v2", "og_giant_v2"]:
        logAndPrint(
            f"{args.depth_method} is a very large model and may cause out of memory errors on GPUs with less than 16GB of VRAM",
            "yellow",
        )
        if args.half:
            logAndPrint(
                "Half precision does not work with giant models, switching to full precision",
            )
            args.half = False


def _handleSegmentSettings(args):
    # Normalize once so every later read is safe, including for args objects
    # built by other entrypoints that never define segment_batch.
    args.segment_batch = max(1, int(getattr(args, "segment_batch", 1)))

    if args.segment_batch > 1:
        backend = args.segment_method.split("-")[-1]
        if backend in ["directml", "openvino"]:
            logAndPrint(
                f"--segment_batch is not implemented for the {backend} segmentation "
                f"backend, using 1",
                "yellow",
            )
            args.segment_batch = 1


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
        elif args.dedup_method in ["vmaf", "vmaf-cuda"]:
            # VMAF is SSIM's "high == similar" scale times 100, so it needs the
            # same mapping. Passed through raw, --dedup_sens 35 became "any pair
            # scoring VMAF >= 35 is a duplicate" -- true of nearly every
            # consecutive pair -- and raising the flag made dedup *less*
            # aggressive, the opposite of what it does for every other method.
            args.dedup_sens = 100.0 - (args.dedup_sens / 10.0)
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

    # Streaming scene-cut skip for the interpolation path. Map the 0-100
    # sensitivity to a per-method threshold (see src/sceneChange/detector.py for
    # the compare direction of each metric). Only meaningful with --interpolate.
    args.scenechange_threshold = None
    if getattr(args, "scenechange", False):
        if not args.interpolate:
            logging.warning(
                "--scenechange has no effect without --interpolate; disabling it"
            )
            args.scenechange = False
        else:
            method = args.scenechange_method
            sens = args.scenechange_sens
            # Device guard: the default (ssim-cuda) and the other CUDA/TensorRT
            # detectors initialize on torch.device("cuda"); downgrade to a
            # CPU/DML equivalent when CUDA is unavailable (CPU/MPS boxes) so
            # enabling --scenechange there does not crash at detector init. The
            # threshold formula is grouped by metric, so remapping here first is
            # threshold-neutral.
            from src.infra.isCudaInit import CudaChecker

            if not CudaChecker().cudaAvailable:
                downgrade = {
                    "ssim-cuda": "ssim",
                    "mse-cuda": "mse",
                    "maxxvit-tensorrt": "maxxvit-directml",
                }
                if method in downgrade:
                    logging.warning(
                        f"CUDA unavailable; scenechange_method {method} -> "
                        f"{downgrade[method]}"
                    )
                    method = downgrade[method]
                    args.scenechange_method = method
            if method in ("ssim", "ssim-cuda"):
                # cut when ssim < threshold; higher sensitivity -> higher threshold
                args.scenechange_threshold = sens / 100.0
            elif method in ("mse", "mse-cuda"):
                # cut when mse > threshold; higher sensitivity -> lower threshold
                args.scenechange_threshold = (100.0 - sens) * 50.0
            elif method in ("maxxvit-tensorrt", "maxxvit-directml"):
                # cut when cut-prob > threshold; mirrors autoclip maxxvit mapping
                args.scenechange_threshold = 1.0 - (sens / 100.0)
            else:
                raise ValueError(
                    f"Unsupported scenechange_method: {method}. "
                    f"transnetv2/pyscenedetect are prepass-only; use --autoclip."
                )
            logging.info(
                f"scenechange threshold for {method} is: {args.scenechange_threshold}"
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

        # Compute provided options up front (before CliConfig exists) so a
        # loaded preset cannot overwrite flags the user explicitly typed.
        providedOptions = CliConfig.collectProvidedOptions(sys.argv[1:])
        args = createPreset(args, providedOptions)

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
            sys.exit(1)

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

    _normalizeMethodAliases(args)

    _configureProcessingSettings(args)

    _adjustMethodsBasedOnCuda(args)

    # After the CUDA fallback, so the quality/batch clamps see the backend that
    # will actually run rather than the one the user typed. Before this move, a
    # CUDA-less box running `--depth --depth_quality high` passed the clamp (the
    # method was still "small_v2"), and the fallback then rewrote it to
    # "small_v2-directml" with depth_quality still "high".
    _handleDepthSettings(args)

    _handleSegmentSettings(args)

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
        sys.exit(1)

    # shouldContinue is False only for the URL-download-without-processing
    # short-circuit (see processUrlInput): the download already succeeded and
    # was renamed to the output, so there is nothing left to do — exit 0.
    if not shouldContinue:
        sys.exit()

    try:
        warning = applyRuntimeValidation(args)
    except CliValidationError as e:
        logAndPrint(str(e), "red")
        sys.exit(1)

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

    # Every "this mode disables audio" rule has run by now, so freeze the intent.
    # getVideoMetadata re-derives cs.AUDIO from it per video.
    cs.AUDIO_REQUESTED = cs.AUDIO

    if not isAnyOtherProcessingMethodEnabled(args) and not args.png_passthrough:
        logAndPrint(
            "No processing methods specified, make sure to use enabler arguments like --upscale, --interpolate, etc.",
            "red",
        )
        sys.exit(1)

    return args
