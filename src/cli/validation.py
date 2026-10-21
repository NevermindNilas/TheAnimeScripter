import math
import os


class CliValidationError(ValueError):
    pass


def validateTrimRange(args):
    from src.io.getVideoMetadata import (
        isFramePoint,
        isTrimUnset,
        parseTrimArg,
    )

    try:
        inpoint = parseTrimArg(getattr(args, "inpoint", 0))
        outpoint = parseTrimArg(getattr(args, "outpoint", 0))
    except Exception as e:
        raise CliValidationError(str(e)) from e

    # Normalize so downstream reads the canonical form (" 100F " -> "100f").
    args.inpoint = inpoint
    args.outpoint = outpoint

    if isTrimUnset(outpoint):
        return

    if isFramePoint(inpoint) and isFramePoint(outpoint):
        inFrames = int(str(inpoint).strip()[:-1].strip())
        outFrames = int(str(outpoint).strip()[:-1].strip())
        if outFrames <= inFrames:
            raise CliValidationError(
                "Invalid trim range: outpoint must be greater than inpoint "
                f"when set (inpoint={inpoint}, outpoint={outpoint})"
            )
        return

    if not isFramePoint(inpoint) and not isFramePoint(outpoint):
        inSeconds = float(inpoint or 0)
        outSeconds = float(outpoint or 0)
        if outSeconds <= inSeconds:
            raise CliValidationError(
                "Invalid trim range: outpoint must be greater than inpoint "
                f"when set (inpoint={inpoint}, outpoint={outpoint})"
            )
        return

    # Mixed seconds/frames need fps to compare; getVideoMetadata and
    # BuildBuffer raise per file via trimFrameRange instead.


def _prefix(label):
    return f"{label}: " if label else ""


def upscaleFactorValue(rawFactor, label=None):
    """``(factor, warning)`` for one upscale stage's factor.

    Config loading rejects fractional values before integer coercion; this
    guard is for callers supplying a namespace directly.
    """
    flag = "the upscale factor" if label else "--upscale_factor"
    if isinstance(rawFactor, bool) or (
        isinstance(rawFactor, float) and not rawFactor.is_integer()
    ):
        raise CliValidationError(
            f"{_prefix(label)}Invalid {flag}: {rawFactor!r}. "
            "Expected a whole number: 2, 3, or 4."
        )
    if isinstance(rawFactor, float):
        rawFactor = int(rawFactor)

    try:
        coerced = int(rawFactor)
    except Exception:
        return 2, f"{_prefix(label)}Invalid upscale_factor provided; defaulting to 2"

    if isinstance(rawFactor, str):
        try:
            if float(rawFactor) != coerced:
                raise CliValidationError(
                    f"{_prefix(label)}Invalid {flag}: {rawFactor!r}. "
                    "Expected a whole number: 2, 3, or 4."
                )
        except ValueError:
            return (
                2,
                f"{_prefix(label)}Invalid upscale_factor provided; defaulting to 2",
            )

    if coerced >= 2:
        return coerced, None

    return 2, (
        f"{_prefix(label)}Upscale factor must be at least 2 when --upscale is "
        f"enabled; defaulting to 2 (was {rawFactor})"
    )


def normalizeUpscaleFactor(args):
    if not args.upscale or not hasattr(args, "upscale_factor"):
        return None
    args.upscale_factor, warning = upscaleFactorValue(args.upscale_factor)
    return warning


def validatePreviewPort(args):
    """Reject a `--preview_port` outside the valid TCP range.

    argparse bounds nothing here, and ``--json`` assigns onto the
    namespace, so an out-of-range port would reach ``socket.bind`` and be
    reported as a generic bind failure. Raising here rather than exiting inside
    ``prepareRuntimeArgs`` also means it fires before ``checkSystem()`` and
    dependency handling, instead of after the slow startup work has run.
    """
    if not getattr(args, "preview", False):
        return
    port = getattr(args, "preview_port", None)
    if port is None:
        return
    try:
        port = int(port)
    except TypeError, ValueError:
        raise CliValidationError(
            f"--preview_port must be a whole number (got {port!r})."
        ) from None
    if not 1 <= port <= 65535:
        raise CliValidationError(
            f"--preview_port must be between 1 and 65535 (got {port})."
        )


def interpolateFactorWarning(rawFactor, smooth, label=None):
    """Reject an interpolation factor below 1; leave 1 and above alone.

    Nothing bounded this flag. ``gapPlan`` owes 0 frames for any factor <= 1
    over a unit gap while the frame loop still writes every source frame, so
    ``--interpolate_factor 0.5`` emitted the source frames at half the rate --
    a video twice as long as its untouched audio, exit 0, no warning. 0 and
    negatives reach the encoder as ``-r 0.0`` / ``-r -47.95``.

    Factor 1 is deliberately left enabled: on the smooth-dedup stage it is the
    supported path that regenerates each duplicated slot as a true in-between
    at unchanged duration and fps. Disabling interpolation here would silently
    switch that feature off.
    """
    flag = "the interpolation factor" if label else "--interpolate_factor"
    try:
        factor = float(rawFactor)
    except TypeError, ValueError:
        raise CliValidationError(
            f"{_prefix(label)}Invalid {flag}: {rawFactor!r}. "
            "Expected a number of at least 1."
        ) from None

    if not math.isfinite(factor):
        raise CliValidationError(
            f"{_prefix(label)}{flag} must be a finite number (got {rawFactor!r})."
        )
    if factor <= 0:
        raise CliValidationError(
            f"{_prefix(label)}{flag} must be greater than 0 (got {factor:g})."
        )
    if factor < 1:
        raise CliValidationError(
            f"{_prefix(label)}{flag} must be at least 1 (got {factor:g}). "
            "A factor below 1 inserts no frames and only lowers the output "
            "frame rate, which desynchronizes the audio. To slow footage down "
            "use --slowmo, which keeps the source frame rate."
        )
    if factor == 1 and not smooth:
        return (
            f"{_prefix(label)}{flag} 1 without --smooth_dedup writes the source "
            "frames unchanged; the interpolation model is still loaded."
        )
    return None


def validateInterpolateFactor(args):
    if not getattr(args, "interpolate", False):
        return None
    return interpolateFactorWarning(
        getattr(args, "interpolate_factor", 2), getattr(args, "smooth_dedup", False)
    )


def selectedUpscaleBackend(upscaleMethod):
    backendSuffixes = ("-directml", "-openvino", "-tensorrt", "-ncnn")
    for suffix in backendSuffixes:
        if upscaleMethod.endswith(suffix):
            return suffix[1:], upscaleMethod[: -len(suffix)]
    return "pytorch", upscaleMethod


def validateCustomModelPath(customModel, upscaleMethod, label=None):
    """The absolute path of an existing custom model that ``upscaleMethod``'s
    backend can load. Raises :class:`CliValidationError`."""
    customModel = os.path.abspath(customModel)
    if not os.path.isfile(customModel):
        raise CliValidationError(
            f"{_prefix(label)}Custom model file not found: {customModel}"
        )

    extension = os.path.splitext(customModel)[1].lower()
    pytorchExtensions = {".pt", ".pth", ".ckpt", ".safetensors"}
    onnxExtensions = {".onnx"}
    selectedBackend, baseMethod = selectedUpscaleBackend(upscaleMethod)

    if extension in onnxExtensions:
        if selectedBackend not in {"directml", "openvino", "tensorrt"}:
            raise CliValidationError(
                f"{_prefix(label)}Custom ONNX upscale models require an ONNX "
                "backend. Use an upscale method ending in -directml, -openvino, "
                f"or -tensorrt, for example {baseMethod}-directml."
            )
        return customModel

    if extension in pytorchExtensions:
        if selectedBackend != "pytorch":
            raise CliValidationError(
                f"{_prefix(label)}Custom PyTorch upscale models require a "
                "CUDA/PyTorch upscale method without a backend suffix. "
                f"Use {baseMethod} for .pt/.pth/.ckpt/.safetensors files."
            )
        return customModel

    raise CliValidationError(
        f"{_prefix(label)}Unsupported custom upscale model format. Supported "
        "extensions are .pt, .pth, .ckpt, .safetensors, and .onnx."
    )


def validateCustomUpscaleModel(args):
    if not args.custom_model:
        return
    args.custom_model = validateCustomModelPath(args.custom_model, args.upscale_method)


def _repeated(drafts, draft):
    """True when ``draft``'s kind appears more than once, so messages must
    say which stage they mean."""
    return sum(1 for other in drafts if other.kind == draft.kind) > 1


def validateStageFactors(drafts):
    """Per-stage factor checks on the resolved drafts. Returns the warnings.

    Every message names its stage, ``interpolate #2 (rife4.26)``, when its
    kind repeats; a kind typed once keeps the wording it had before stages
    could repeat.
    """
    warnings = []
    for draft in drafts:
        label = draft.label if _repeated(drafts, draft) else None
        settings = draft.settings
        if draft.kind == "upscale":
            settings["factor"], warning = upscaleFactorValue(settings["factor"], label)
            warnings.append(warning)
    for draft in drafts:
        label = draft.label if _repeated(drafts, draft) else None
        if draft.kind == "interpolate":
            warnings.append(
                interpolateFactorWarning(draft.settings["factor"], draft.smooth, label)
            )
    return [warning for warning in warnings if warning]


def applyRuntimeValidation(args):
    from src.cli.stageOrder import stageDrafts, syncStageGlobals

    drafts = stageDrafts(args)
    if drafts is None:
        validateCustomUpscaleModel(args)
    else:
        for draft in drafts:
            if draft.kind == "upscale" and draft.settings.get("customModel"):
                draft.settings["customModel"] = validateCustomModelPath(
                    draft.settings["customModel"],
                    draft.settings["method"],
                    draft.label if _repeated(drafts, draft) else None,
                )
    validateTrimRange(args)
    validatePreviewPort(args)
    # Both normalizers can warn; join rather than let one shadow the other.
    if drafts is None:
        warnings = [normalizeUpscaleFactor(args), validateInterpolateFactor(args)]
    else:
        warnings = validateStageFactors(drafts)
        syncStageGlobals(args, drafts)
    return "\n".join(w for w in warnings if w) or None
