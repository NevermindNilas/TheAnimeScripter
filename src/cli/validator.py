import logging
import os
import sys

import src.constants as cs
from src.cli.config import CliConfig
from src.cli.startup import _handleDependencies, _promptDownloadRequirementsSelection
from src.cli.validation import CliValidationError, applyRuntimeValidation
from src.infra.logAndPrint import logAndPrint, logWarning
from src.io.inputNormalization import InputNormalizationError, normalizeInputArgs
from src.io.inputOutputHandler import SEQUENCE_EXTENSIONS


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
            # --output_scale is a real request even though it is applied by the
            # encoder rather than a model. It used to count by auto-enabling
            # --resize, which also dragged in --resize_factor's default of 2 and
            # doubled the decode resolution (see src/cli/config.py).
            bool(getattr(args, "output_scale", "")),
        ]
    )


def _downgradeCudaDetector(method: str, flagName: str) -> str:
    """Swap a CUDA-only frame comparator for its CPU twin on a CUDA-less box.

    `--dedup`, `--smooth_dedup` and `--scenechange` drive the same comparators
    through `src/factories/dedup.py`, and the CUDA ones initialize on
    `torch.device("cuda")`. Only `--smooth_dedup` used to guard them, so
    `--dedup --dedup_method flownets` on a Mac or a CUDA-less Linux box ran all
    the way through startup, downloaded the FlowNetS weights, and then died at
    model init with a raw "Torch not compiled with CUDA enabled" that never
    named the flag.
    """
    try:
        from src.infra.isCudaInit import CudaChecker

        cudaAvailable = CudaChecker().cudaAvailable
    except Exception:
        # No torch at all (a bare CI venv): leave the pick alone rather than
        # rewriting it on the strength of a failed probe.
        return method

    if cudaAvailable:
        return method

    downgrade = {"ssim-cuda": "ssim", "mse-cuda": "mse", "vmaf-cuda": "vmaf"}
    if method in downgrade:
        logging.warning(f"CUDA unavailable; {flagName} {method} -> {downgrade[method]}")
        return downgrade[method]
    if method == "flownets":
        logAndPrint(
            f"{flagName} flownets requires CUDA, falling back to ssim",
            "yellow",
        )
        return "ssim"
    return method


def _neluxSupportsEncoderResize() -> bool:
    """True when the installed nelux can honor `--output_scale` itself.

    Encoder-side resize (`VideoEncoder(resize=True)`) exists since nelux
    0.18.0. Read the version from package metadata rather than importing
    nelux, which is heavy and demands torch be imported first. When nelux is
    not installed (a bare CI venv) the run dies at `import nelux` long before
    the writer matters, so report support rather than downgrading the method.
    An unparseable version reports no support: the loud FFmpeg-twin downgrade
    beats a raw TypeError from `VideoEncoder(resize=...)` mid-run.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        installed = version("nelux")
    except PackageNotFoundError:
        return True
    try:
        major, minor = installed.split(".")[:2]
        return (int(major), int(minor)) >= (0, 18)
    except ValueError:
        return False


def _resolveNeluxEncoder(args):
    """Swap a `*_nelux` encoder for its FFmpeg twin when it cannot be used.

    `NeluxWriteBuffer` takes most of `WriteBuffer`'s keyword arguments through
    `**kwargs` and then never reads them, and `--depth` does not go through the
    Nelux writer at all. Both used to fail silently: `--output_scale 320x180`
    wrote a full-resolution file, `--bit_depth 16bit` wrote 8-bit, and
    `--custom_encoder` was dropped. Resolve it once here, like every other
    incompatible-option downgrade, rather than per input file in the writer.

    `--output_scale` is honored natively since nelux 0.18.0 (encoder-side
    resize) and only downgrades when the installed nelux predates it.
    """
    method = getattr(args, "encode_method", "") or ""
    if not method.endswith("_nelux"):
        return

    dropped = []
    if args.custom_encoder:
        dropped.append("--custom_encoder")
    if getattr(args, "bit_depth", "8bit") != "8bit":
        dropped.append("--bit_depth")
    if (
        args.output_scale_width or args.output_scale_height
    ) and not _neluxSupportsEncoderResize():
        dropped.append("--output_scale (nelux < 0.18.0)")

    if dropped:
        reason = f"cannot apply {', '.join(dropped)}"
    elif args.depth:
        reason = "is not available for --depth"
    else:
        return

    args.encode_method = method[: -len("_nelux")]
    # With --custom_encoder the FFmpeg writer runs that encoder verbatim and
    # never consults the method, so naming a substitute encoder would be a lie.
    substitute = (
        "your --custom_encoder"
        if args.custom_encoder
        else f"the equivalent {args.encode_method}"
    )
    logWarning(
        f"{method} {reason}. Encoding with {substitute} instead, so the output "
        "matches what you asked for."
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


# Trailing tokens that name a non-torch interpolation backend. "-lite" /
# "-heavy" are model variants, not backends, so they stay off this list.
_NON_TORCH_INTERP_BACKENDS = frozenset({"tensorrt", "directml", "openvino", "ncnn"})


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


def _applyImpliedFlags(args):
    """Resolve flags that turn other capabilities on or off.

    This runs before every guard that reads those capabilities. `--smooth_dedup`
    used to enable `--interpolate` from inside `_configureProcessingSettings`,
    which is late: the `--slowmo` and `--dynamic_scale` guards had already seen
    `interpolate=False` and silently switched themselves off.
    """
    if not getattr(args, "smooth_dedup", False):
        return

    # --smooth_dedup runs the duplicate detector itself and turns each dropped
    # frame into extra interpolation slots instead of shortening the video, so
    # the frame-dropping --dedup path must not also run.
    if args.dedup:
        logAndPrint(
            "--smooth_dedup handles duplicate frames itself, disabling --dedup",
            "yellow",
        )
        args.dedup = False

    if not args.interpolate:
        logging.info("Auto-enabling --interpolate because --smooth_dedup was provided")
        args.interpolate = True


def _mapDedupSensitivity(method, sensitivity):
    """Map a 0-100 CLI sensitivity onto the threshold a detector backend wants.

    Shared by --dedup and --smooth_dedup: both drive the same detectors from
    src/factories/dedup.py, so they must agree on the scale.
    """
    if method in ["ssim", "ssim-cuda"]:
        return 1.0 - (sensitivity / 1000)
    if method in ["vmaf", "vmaf-cuda"]:
        # VMAF is SSIM's "high == similar" scale times 100, so it needs the same
        # mapping. Passed through raw, a sensitivity of 35 became "any pair
        # scoring VMAF >= 35 is a duplicate" -- true of nearly every consecutive
        # pair -- and raising the flag made dedup *less* aggressive, the opposite
        # of what it does for every other method.
        return 100.0 - (sensitivity / 10.0)
    if method in ["flownets"]:
        return sensitivity / 100
    # MSE takes the raw value.
    return sensitivity


def _mapAutoclipSensitivity(method, sensitivity):
    """Map the 0-100 CLI sensitivity onto each autoclip backend's threshold.

    pyscenedetect: AdaptiveDetector's ``adaptive_threshold`` is a frame-score
    ratio whose usable range is roughly 0.5-6 (library default 3.0). The old
    ``100 - sens`` mapping put the DEFAULT sensitivity of 50 at threshold 50 —
    a value the ratio virtually never reaches — so a default autoclip run
    detected nothing, not even an artificial testsrc-to-smptebars hard cut.
    Map linearly so the default lands on the library default: 0 -> 6.0 (least
    sensitive), 50 -> 3.0, 100 -> 0.5 (floored; a zero threshold would cut on
    every frame).

    maxxvit / transnetv2 emit a cut probability, so the threshold is the
    inverted fraction: 50 -> 0.5.
    """
    if method == "pyscenedetect":
        return max(0.5, 3.0 * (100.0 - float(sensitivity)) / 50.0)
    return float(1.0 - (float(sensitivity) / 100.0))


def _configureProcessingSettings(args):
    if args.slowmo:
        cs.AUDIO = False
        logging.info("Slow motion enabled, audio processing disabled")

    if args.static_step and isinstance(args.interpolate_factor, float):
        logging.info("Interpolate Factor is a float, static step will be disabled")
        args.static_step = False

    # --dynamic_scale is read by the RIFE torch arches only (src/rifearches/),
    # i.e. by RifeCuda and RifeMPS. The TensorRT / DirectML / OpenVINO / NCNN
    # classes and the gmfss / distildrba families never receive the flag, so
    # leaving it set there was a silent no-op instead of an error. rife_elexor
    # is caught later, in RifeCuda.handleModel -- its arch takes no such argument.
    if getattr(args, "dynamic_scale", False):
        backend = args.interpolate_method.rsplit("-", 1)[-1]
        if not args.interpolate:
            logging.warning("--dynamic_scale has no effect without --interpolate")
            args.dynamic_scale = False
        elif backend in _NON_TORCH_INTERP_BACKENDS:
            logAndPrint(
                f"--dynamic_scale is not supported by the {backend} interpolation "
                "backend (CUDA and MPS only), disabling it",
                "yellow",
            )
            args.dynamic_scale = False
        elif not args.interpolate_method.startswith("rife"):
            logAndPrint(
                f"--dynamic_scale is not supported by {args.interpolate_method}, "
                "disabling it",
                "yellow",
            )
            args.dynamic_scale = False

    if getattr(args, "smooth_dedup", False):
        # --dedup is already forced off and --interpolate on, in
        # _applyImpliedFlags, ahead of every guard that reads them.
        #
        # The cap chooses copies over inference for a gap wider than it; it is
        # NOT a memory bound, and never was. What bounds memory is the
        # interpolate-first sink draining itself as it fills -- see
        # main.py:_FrameCollector, which is where a long hold used to keep the
        # whole gap alive at once. 0 would disable the cap entirely, so the
        # floor of 1 stays.
        args.smooth_dedup_max_span = max(
            1, int(getattr(args, "smooth_dedup_max_span", 6))
        )

        args.smooth_dedup_method = _downgradeCudaDetector(
            args.smooth_dedup_method, "smooth_dedup_method"
        )

        # Duration is preserved, so audio stays in sync and is left enabled.
        args.smooth_dedup_sens = _mapDedupSensitivity(
            args.smooth_dedup_method, args.smooth_dedup_sens
        )
        logging.info(
            f"New smooth_dedup sensitivity for {args.smooth_dedup_method} is: {args.smooth_dedup_sens}"
        )

    if args.dedup:
        cs.AUDIO = False
        logging.info("Deduplication enabled, audio processing disabled")

        # Before the sensitivity mapping, which is grouped by metric and so
        # unaffected by the swap.
        args.dedup_method = _downgradeCudaDetector(args.dedup_method, "dedup_method")
        args.dedup_sens = _mapDedupSensitivity(args.dedup_method, args.dedup_sens)
        logging.info(
            f"New dedup sensitivity for {args.dedup_method} is: {args.dedup_sens}"
        )

    if getattr(args, "stabilize", False) and args.stabilize_method == "dut":
        # DUT initializes four torch models on torch.device("cuda"); same
        # visible-downgrade contract as the CUDA frame comparators above.
        try:
            from src.infra.isCudaInit import CudaChecker

            cudaAvailable = CudaChecker().cudaAvailable
        except Exception:
            cudaAvailable = False
        if not cudaAvailable:
            logAndPrint(
                "stabilize_method dut requires CUDA, falling back to classic",
                "yellow",
            )
            args.stabilize_method = "classic"

    if args.autoclip:
        args.autoclip_sens = _mapAutoclipSensitivity(
            args.autoclip_method, args.autoclip_sens
        )
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


def _adjustMethodsBasedOnCuda(args, availableModels=None, methodChoices=None):
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

        if methodChoices is None:
            # The CLI choices, not the weight registry, decide whether a
            # non-CUDA sibling exists (see fallbackMethod's docstring).
            from src.cli.parser import _buildParser, capabilityMethods

            methodChoices = capabilityMethods(_buildParser("."))

        applyBackendFallbacks(
            args,
            availableModels,
            preferMps=cs.SYSTEM == "Darwin",
            methodChoices=methodChoices,
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
        args = createPreset(args, providedOptions, parser)

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
            logAndPrint(f"Failed to start AE comms server: {e}", "red")

    logging.info("\n============== Arguments Checker ==============")
    _handleDependencies(args)

    _applyImpliedFlags(args)

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

    if args.encode_method == "gif" or args.encode_method in SEQUENCE_EXTENSIONS:
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

    _resolveNeluxEncoder(args)

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
