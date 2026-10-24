import logging
import os
import sys

import src.constants as cs
from src.cli.config import CliConfig
from src.cli.stageOrder import (
    stageDrafts,
    stageOrderWarning,
    syncStageGlobals,
)
from src.cli.startup import _handleDependencies, _resolveRequestedProfile
from src.cli.validation import CliValidationError, applyRuntimeValidation
from src.infra.logAndPrint import logAndPrint
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
        ]
    )


# Sentinel for "no CUDA probe was handed in". Startup probes once (see
# prepareRuntimeArgs) and passes the result down, instead of every guard
# constructing its own CudaChecker. As a *value*, None means the probe itself
# failed -- no torch at all, e.g. a bare CI venv -- which the dedup guards
# treat as "leave the pick alone".
_UNPROBED = object()


def _probeCudaAvailability():
    """One CudaChecker construction; None when torch is absent (bare CI)."""
    try:
        from src.infra.isCudaInit import CudaChecker

        return bool(CudaChecker().cudaAvailable)
    except Exception:
        return None


def _downgradeCudaDetector(method: str, flagName: str, cudaAvailable=_UNPROBED) -> str:
    """Swap a CUDA-only frame comparator for its CPU twin on a CUDA-less box.

    `--dedup`, `--smooth_dedup` and `--scenechange` drive the same comparators
    through `src/factories/dedup.py`, and the CUDA ones initialize on
    `torch.device("cuda")`. Only `--smooth_dedup` used to guard them, so
    `--dedup --dedup_method flownets` on a Mac or a CUDA-less Linux box ran all
    the way through startup, downloaded the FlowNetS weights, and then died at
    model init with a raw "Torch not compiled with CUDA enabled" that never
    named the flag.

    `cudaAvailable` is the shared startup probe (True/False), or None when the
    probe itself failed. Omitted, it probes once here, so direct callers keep
    working.
    """
    if cudaAvailable is _UNPROBED:
        cudaAvailable = _probeCudaAvailability()
    if cudaAvailable is None:
        # No torch at all (a bare CI venv): leave the pick alone rather than
        # rewriting it on the strength of a failed probe.
        return method

    if cudaAvailable:
        return method

    downgrade = {
        "ssim-cuda": "ssim",
        "mse-cuda": "mse",
        "vmaf-cuda": "vmaf",
        "ssim-rocm": "ssim",
        "mse-rocm": "mse",
        "vmaf-rocm": "vmaf",
    }
    if method in downgrade:
        logging.warning(f"CUDA unavailable; {flagName} {method} -> {downgrade[method]}")
        return downgrade[method]
    if method in ("flownets", "flownets-rocm"):
        logAndPrint(
            f"{flagName} {method} requires CUDA/ROCm, falling back to ssim",
            "yellow",
        )
        return "ssim"
    return method


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
    "rife-rocm": "rife4.22-rocm",
}


# Trailing tokens that name a non-torch interpolation backend. "-lite" /
# "-heavy" are model variants, not backends, so they stay off this list.
_NON_TORCH_INTERP_BACKENDS = frozenset({"tensorrt", "directml", "openvino", "ncnn"})


def _normalizeMethodAliases(args):
    drafts = stageDrafts(args)
    attributes = ("moblur_method",)
    if drafts is None:
        attributes = ("interpolate_method", "moblur_method")
    else:
        for draft in drafts:
            if draft.kind != "interpolate":
                continue
            current = draft.settings["method"]
            canonical = _BARE_RIFE_ALIASES.get(current)
            if canonical is not None:
                logging.info(
                    f"{draft.label}: '{current}' is an alias for '{canonical}'"
                )
                draft.settings["method"] = canonical
        syncStageGlobals(args, drafts)
    for attribute in attributes:
        current = getattr(args, attribute, None)
        canonical = _BARE_RIFE_ALIASES.get(current)
        if canonical is not None:
            logging.info(f"{attribute} '{current}' is an alias for '{canonical}'")
            setattr(args, attribute, canonical)


def _handleDepthSettings(args):
    if args.depth:
        logging.info("Depth enabled, audio processing will be disabled")
        cs.AUDIO = False

    isLimbo = args.depth_method.removeprefix("video_").split("-")[0].startswith("limbo")

    # "openvino" belongs here too: it is not a separate backend, it is a
    # provider branch inside the same DepthDirectMLV2 class the "-directml"
    # methods use (src/factories/standalone.py), so it has the same constraint.
    # The --depth_batch clamp below already listed it. Limbo is exempt: its
    # ONNX backends export a graph at the quality's size themselves
    # (_shared.limboOnnxPath).
    backend = args.depth_method.split("-")[-1]
    if (
        args.depth_quality not in ["low"]
        and backend in ["tensorrt", "directml", "openvino"]
        and not isLimbo
    ):
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
        elif isLimbo and backend == "tensorrt":
            # Limbo's ONNX bakes batch 1 into the graph, so the engine cannot be
            # built for more. The CUDA and MPS paths run the torch model and do
            # batch normally, hence the backend check rather than a blanket one.
            logAndPrint(
                f"--depth_batch is not supported for {args.depth_method} (the "
                "export is fixed at batch 1), using 1",
                "yellow",
            )
            args.depth_batch = 1


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

    drafts = stageDrafts(args)
    if drafts is not None:
        # Every dedup stage goes, wherever it was typed; it gets no place in
        # the chain. The smooth stage is an interpolate stage, so syncing
        # turns --interpolate on.
        if any(draft.kind == "dedup" for draft in drafts):
            logAndPrint(
                "--smooth_dedup handles duplicate frames itself, disabling --dedup",
                "yellow",
            )
            drafts[:] = [draft for draft in drafts if draft.kind != "dedup"]
        syncStageGlobals(args, drafts)
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


# Capabilities that run outside the frame loop, in the order main.py picks
# them (VideoProcessor._selectProcessingMethod).
_STANDALONE_CAPABILITIES = (
    "autoclip",
    "depth",
    "segment",
    "obj_detect",
    "stabilize",
    "moblur",
)


def _finalizeChain(args):
    """Set ``args.chain``, the per-kind booleans and the first-instance globals.

    Must run after :func:`_applyImpliedFlags` and every other enable/disable
    step, because it reads the final state. Also prints the one order warning
    (a dedup stage after an interpolate stage). Raises
    :class:`CliValidationError` for a standalone capability given more than
    one ``--resize``.
    """
    drafts = stageDrafts(args)
    if drafts is None:
        args.chain = ()
        return args.chain
    syncStageGlobals(args, drafts)

    standalone = next(
        (name for name in _STANDALONE_CAPABILITIES if getattr(args, name, False)),
        None,
    )
    resizes = [spec for spec in args.chain if spec.kind == "resize"]
    if standalone and len(resizes) > 1:
        raise CliValidationError(
            f"--{standalone} runs outside the frame loop and takes at most one "
            f"--resize (the decoder applies it); got {len(resizes)}."
        )

    # A backend that cannot run as two live instances: refused here, naming
    # the stages, rather than when the second driver is built.
    from src.pipeline.instances import checkChainInstances

    try:
        checkChainInstances(args.chain)
    except ValueError as error:
        raise CliValidationError(str(error)) from None

    warning = stageOrderWarning(args.chain)
    if warning:
        logAndPrint(warning, "yellow")
    return args.chain


def _mapDedupSensitivity(method, sensitivity):
    """Map a 0-100 CLI sensitivity onto the threshold a detector backend wants.

    Shared by --dedup and --smooth_dedup: both drive the same detectors from
    src/factories/dedup.py, so they must agree on the scale.
    """
    if method in ["ssim", "ssim-cuda", "ssim-rocm"]:
        return 1.0 - (sensitivity / 1000)
    if method in ["vmaf", "vmaf-cuda", "vmaf-rocm"]:
        # VMAF is SSIM's "high == similar" scale times 100, so it needs the same
        # mapping. Passed through raw, a sensitivity of 35 became "any pair
        # scoring VMAF >= 35 is a duplicate" -- true of nearly every consecutive
        # pair -- and raising the flag made dedup *less* aggressive, the opposite
        # of what it does for every other method.
        return 100.0 - (sensitivity / 10.0)
    if method in ["flownets", "flownets-rocm"]:
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


class _LegacyStage:
    """A namespace's global settings dressed as one stage, for callers that
    hand ``_configureProcessingSettings`` a namespace with no chain (tests,
    other entrypoints). The real run always has drafts."""

    def __init__(self, kind, settings, smooth=False):
        self.kind = kind
        self.settings = settings
        self.smooth = smooth
        self.label = "smooth_dedup" if smooth else kind
        self.repeated = False

    def flagName(self, legacy):
        return legacy


def _legacyStages(args):
    stages = []
    if getattr(args, "smooth_dedup", False):
        stages.append(
            _LegacyStage(
                "interpolate",
                {
                    "method": getattr(args, "interpolate_method", "rife4.6"),
                    "factor": getattr(args, "interpolate_factor", 2),
                    "staticStep": getattr(args, "static_step", False),
                    "dynamicScale": getattr(args, "dynamic_scale", False),
                    "ensemble": getattr(args, "ensemble", False),
                    "detector": args.smooth_dedup_method,
                    "sens": args.smooth_dedup_sens,
                    "maxSpan": getattr(args, "smooth_dedup_max_span", 6),
                },
                smooth=True,
            )
        )
    elif getattr(args, "interpolate", False):
        stages.append(
            _LegacyStage(
                "interpolate",
                {
                    "method": getattr(args, "interpolate_method", "rife4.6"),
                    "factor": getattr(args, "interpolate_factor", 2),
                    "staticStep": getattr(args, "static_step", False),
                    "dynamicScale": getattr(args, "dynamic_scale", False),
                    "ensemble": getattr(args, "ensemble", False),
                },
            )
        )
    if getattr(args, "dedup", False):
        stages.append(
            _LegacyStage(
                "dedup", {"method": args.dedup_method, "sens": args.dedup_sens}
            )
        )
    return stages


def _writeLegacyStages(args, stages):
    for stage in stages:
        settings = stage.settings
        if stage.kind == "interpolate":
            if hasattr(args, "static_step"):
                args.static_step = settings["staticStep"]
            if hasattr(args, "dynamic_scale"):
                args.dynamic_scale = settings["dynamicScale"]
            if hasattr(args, "ensemble"):
                args.ensemble = settings["ensemble"]
        if stage.smooth:
            args.smooth_dedup_max_span = settings["maxSpan"]
            args.smooth_dedup_method = settings["detector"]
            args.smooth_dedup_sens_raw = settings["sens"]
            args.smooth_dedup_sens = settings["threshold"]
        if stage.kind == "dedup":
            args.dedup_method = settings["method"]
            args.dedup_sens_raw = settings["sens"]
            args.dedup_sens = settings["threshold"]


def _resolveStaticStep(stage):
    settings = stage.settings
    if settings.get("staticStep") and isinstance(settings["factor"], float):
        where = f" ({stage.label})" if stage.repeated else ""
        logging.info(
            f"Interpolate Factor is a float, static step will be disabled{where}"
        )
        settings["staticStep"] = False


def _resolveDynamicScale(stage):
    """--dynamic_scale is read by the RIFE torch arches only (src/rifearches/),
    i.e. by RifeCuda, RifeMPS and RifeROCm. The TensorRT / DirectML / OpenVINO
    / NCNN classes and the gmfss / distildrba families never receive the flag,
    so leaving it set there was a silent no-op instead of an error. rife_elexor
    is caught later, in RifeCuda.handleModel -- its arch takes no such argument.
    """
    settings = stage.settings
    if not settings.get("dynamicScale"):
        return
    method = settings["method"]
    backend = method.rsplit("-", 1)[-1]
    where = f" on {stage.label}" if stage.repeated else ""
    if backend in _NON_TORCH_INTERP_BACKENDS:
        logAndPrint(
            f"--dynamic_scale is not supported by the {backend} interpolation "
            f"backend (CUDA, ROCm and MPS only), disabling it{where}",
            "yellow",
        )
        settings["dynamicScale"] = False
    elif not method.startswith("rife"):
        logAndPrint(
            f"--dynamic_scale is not supported by {method}, disabling it{where}",
            "yellow",
        )
        settings["dynamicScale"] = False


def _resolveFlipViews(stage):
    """rife4.26-ultra already averages four flipped views at a fixed scale, so
    --ensemble (a fifth, time-reversed pass) and --dynamic_scale have no place
    in it; the arch refuses the first and ignores the second."""
    settings = stage.settings
    if not settings["method"].startswith("rife4.26-ultra"):
        return
    where = f" on {stage.label}" if stage.repeated else ""
    for key, flag in (("ensemble", "--ensemble"), ("dynamicScale", "--dynamic_scale")):
        if settings.get(key):
            logAndPrint(
                f"{flag} is not supported by rife4.26-ultra, which already averages "
                f"four flipped views; disabling it{where}",
                "yellow",
            )
            settings[key] = False


def _configureProcessingSettings(args, cudaAvailable=_UNPROBED):
    if args.slowmo:
        cs.AUDIO = False
        logging.info("Slow motion enabled, audio processing disabled")

    drafts = stageDrafts(args)
    stages = drafts if drafts is not None else _legacyStages(args)

    for stage in stages:
        if stage.kind == "interpolate":
            _resolveStaticStep(stage)
            _resolveDynamicScale(stage)
            _resolveFlipViews(stage)
    if getattr(args, "dynamic_scale", False) and not args.interpolate:
        logging.warning("--dynamic_scale has no effect without --interpolate")
        args.dynamic_scale = False

    for stage in stages:
        if not stage.smooth:
            continue
        # --dedup is already forced off and --interpolate on, in
        # _applyImpliedFlags, ahead of every guard that reads them.
        #
        # The cap chooses copies over inference for a gap wider than it; it is
        # NOT a memory bound, and never was. What bounds memory is the
        # interpolation stage handing a gap on as it fills -- see
        # src/pipeline/executor.py:ExpandNode.collectLimit, which keeps a long
        # hold from holding the whole gap at once. 0 disables the cap (every run is
        # interpolated, however long); negatives normalize to 0 rather than
        # inverting the span comparison into "hold everything".
        settings = stage.settings
        settings["maxSpan"] = max(0, int(settings["maxSpan"]))

        # First CUDA guard in this function resolves the shared probe; every
        # later guard below reuses it instead of constructing its own checker.
        if cudaAvailable is _UNPROBED:
            cudaAvailable = _probeCudaAvailability()
        settings["detector"] = _downgradeCudaDetector(
            settings["detector"], stage.flagName("smooth_dedup_method"), cudaAvailable
        )

        # Duration is preserved, so audio stays in sync and is left enabled.
        settings["threshold"] = _mapDedupSensitivity(
            settings["detector"], settings["sens"]
        )
        logging.info(
            f"New smooth_dedup sensitivity for {settings['detector']} is: {settings['threshold']}"
        )

    dedupStages = [stage for stage in stages if stage.kind == "dedup"]
    if dedupStages:
        cs.AUDIO = False
        logging.info("Deduplication enabled, audio processing disabled")
    for stage in dedupStages:
        # Before the sensitivity mapping, which is grouped by metric and so
        # unaffected by the swap.
        settings = stage.settings
        if cudaAvailable is _UNPROBED:
            cudaAvailable = _probeCudaAvailability()
        settings["method"] = _downgradeCudaDetector(
            settings["method"], stage.flagName("dedup_method"), cudaAvailable
        )
        settings["threshold"] = _mapDedupSensitivity(
            settings["method"], settings["sens"]
        )
        logging.info(
            f"New dedup sensitivity for {settings['method']} is: {settings['threshold']}"
        )

    if drafts is not None:
        syncStageGlobals(args, drafts)
    else:
        _writeLegacyStages(args, stages)

    if getattr(args, "stabilize", False) and args.stabilize_method in (
        "dut",
        "dut-rocm",
    ):
        # DUT initializes four torch models on torch.device("cuda") (HIP on
        # ROCm); same visible-downgrade contract as the CUDA frame comparators
        # above. A failed probe (None, no torch at all) counts as unavailable
        # and falls back, matching the old try/except here.
        if cudaAvailable is _UNPROBED:
            cudaAvailable = _probeCudaAvailability()
        if not cudaAvailable:
            logAndPrint(
                f"stabilize_method {args.stabilize_method} requires CUDA/ROCm, "
                "falling back to classic",
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
            # threshold-neutral. A failed probe (None, no torch at all) also
            # downgrades instead of raising out of the checker construction,
            # which the old unguarded probe did on a bare CI venv.
            if cudaAvailable is _UNPROBED:
                cudaAvailable = _probeCudaAvailability()
            if not cudaAvailable:
                downgrade = {
                    "ssim-cuda": "ssim",
                    "mse-cuda": "mse",
                    "ssim-rocm": "ssim",
                    "mse-rocm": "mse",
                    "maxxvit-tensorrt": "maxxvit-directml",
                }
                if method in downgrade:
                    logging.warning(
                        f"CUDA unavailable; scenechange_method {method} -> "
                        f"{downgrade[method]}"
                    )
                    method = downgrade[method]
                    args.scenechange_method = method
            if method in ("ssim", "ssim-cuda", "ssim-rocm"):
                # cut when ssim < threshold; higher sensitivity -> higher threshold
                args.scenechange_threshold = sens / 100.0
            elif method in ("mse", "mse-cuda", "mse-rocm"):
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


def _adjustMethodsBasedOnCuda(
    args, availableModels=None, methodChoices=None, parser=None, cudaChecker=_UNPROBED
):
    """Rewrite CUDA-only methods to sibling backends when CUDA is unavailable.

    `parser` is the live startup parser: its choice lists decide whether a
    non-CUDA sibling exists, so it is reused here instead of building a second
    parser. `cudaChecker` is the shared startup probe (a CudaChecker, or None
    when the probe failed); omitted, it probes once here, so direct callers
    keep working.
    """
    supportsCuda = getattr(args, "supportsCuda", None)

    preferRocm = False
    if supportsCuda is None:
        from src.infra.isCudaInit import detectGPUArchitecture

        if cudaChecker is _UNPROBED:
            try:
                from src.infra.isCudaInit import CudaChecker

                cudaChecker = CudaChecker()
            except Exception:
                # No torch at all (a bare CI venv): without a usable
                # torch.cuda there is no CUDA path, so fall back.
                cudaChecker = None

        isCuda = cudaChecker

        needsFallback = False
        if isCuda is not None and isCuda.cudaAvailable:
            if isCuda.rocmAvailable:
                # ROCm/HIP torch on AMD: torch.cuda works, but NVML/nvidia-smi
                # see no NVIDIA GPU, so detectGPUArchitecture would report
                # "not modern" and needlessly downgrade to DirectML. The CUDA
                # paths run via HIP; explicit -rocm picks stay as-is.
                logging.info("ROCm/HIP torch detected; keeping CUDA-capable methods")
            else:
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
            try:
                from src.infra.isCudaInit import isRocmTorch

                # No usable torch.cuda at all; still prefer -rocm siblings in
                # the message ordering if the torch build is HIP (e.g. AMD box
                # where HIP init failed) — the suggestions stay honest.
                preferRocm = bool(isRocmTorch())
            except Exception:
                preferRocm = False
    else:
        needsFallback = not supportsCuda

    if needsFallback:
        from src.infra.backendFallback import applyBackendFallbacks

        if availableModels is None:
            from src.model.registry import modelsList

            availableModels = modelsList()

        if methodChoices is None:
            if parser is not None:
                # The CLI choices, not the weight registry, decide whether a
                # non-CUDA sibling exists (see fallbackMethod's docstring).
                from src.cli.parser import capabilityMethods

                methodChoices = capabilityMethods(parser)
            else:
                # Back-compat for direct calls without a parser (tests).
                from src.cli.parser import _buildParser, capabilityMethods

                methodChoices = capabilityMethods(_buildParser("."))

        drafts = stageDrafts(args)
        applyBackendFallbacks(
            args,
            availableModels,
            preferMps=cs.SYSTEM == "Darwin",
            methodChoices=methodChoices,
            preferRocm=preferRocm,
            stages=drafts,
        )
        if drafts is not None:
            syncStageGlobals(args, drafts)


def prepareRuntimeArgs(args, outputPath, parser):
    from src.version import __version__

    args.png_passthrough = False
    args.single_image_input = False

    if args.list_methods is not None:
        from src.cli.parser import _listMethods

        sys.exit(_listMethods(parser, args.list_methods))

    cliConfig = CliConfig.fromArgs(args, parser, sys.argv[1:])
    args = cliConfig.args

    if args.download_requirements is not None:
        from src.infra.dependencyHandler import DependencyChecker

        _handleDependencies(args)
        selectedProfile = _resolveRequestedProfile(
            args.download_requirements, args.dependency_profile
        )

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
            from src.infra.isCudaInit import (
                detectGPUArchitecture,
                detectNVidiaGPU,
                detectRocmGPU,
            )

            isNvidia = detectNVidiaGPU()
            supportsCuda = False
            if isNvidia:
                supportsCuda, _, _ = detectGPUArchitecture()
            try:
                isRocm = detectRocmGPU()
            except Exception:
                isRocm = False
            extension = getRequirementsFileForProfile(
                getDependencyProfile(cs.SYSTEM, supportsCuda, isRocm)
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

        def reportAeServerFailure(error):
            logging.error(f"Failed to start AE comms server: {error}")
            logAndPrint(f"Failed to start AE comms server: {error}", "red")

        # The bind is checked in the background: startup does not wait the
        # ~0.1 s the server child needs to boot and report.
        try:
            startServerInThread(host=args.ae, onFailure=reportAeServerFailure)
        except Exception as e:
            reportAeServerFailure(e)

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

    # One CUDA probe for the whole startup: the dedup, smooth-dedup,
    # stabilize and scenechange guards plus the backend fallback below used to
    # construct one CudaChecker each. None means the probe itself failed -- no
    # torch at all, e.g. a bare CI venv -- which the guards treat as "no CUDA"
    # (the dedup guards leave CUDA picks alone rather than rewriting them on
    # a failed probe).
    try:
        from src.infra.isCudaInit import CudaChecker

        cudaChecker = CudaChecker()
    except Exception:
        cudaChecker = None
    cudaAvailable = bool(cudaChecker.cudaAvailable) if cudaChecker is not None else None

    _configureProcessingSettings(args, cudaAvailable=cudaAvailable)

    _adjustMethodsBasedOnCuda(args, parser=parser, cudaChecker=cudaChecker)

    # After the CUDA fallback, so the quality/batch clamps see the backend that
    # will actually run rather than the one the user typed. Before this move, a
    # CUDA-less box running `--depth --depth_quality high` passed the clamp (the
    # method was still "small_v2"), and the fallback then rewrote it to
    # "small_v2-directml" with depth_quality still "high".
    _handleDepthSettings(args)

    _handleSegmentSettings(args)

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

    if warning:
        logAndPrint(warning, "yellow")
        logging.info(warning)

    # Last enable/disable step is done: the chain reads the final enabled set.
    try:
        _finalizeChain(args)
    except CliValidationError as e:
        logAndPrint(str(e), "red")
        sys.exit(1)

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
