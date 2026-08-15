import os


class CliValidationError(ValueError):
    pass


def parseOutputScale(value):
    if not value:
        return None, None

    try:
        width, height = value.split("x")
        outputWidth = int(width)
        outputHeight = int(height)
    except (ValueError, AttributeError) as e:
        raise CliValidationError(
            f"Invalid output_scale format: {value}. Expected format: WIDTHxHEIGHT (e.g., 2560x1440)"
        ) from e

    if outputWidth <= 0 or outputHeight <= 0:
        raise CliValidationError(
            f"Invalid output_scale format: {value}. Expected format: WIDTHxHEIGHT (e.g., 2560x1440)"
        )

    return outputWidth, outputHeight


def applyOutputScale(args):
    args.output_scale_width, args.output_scale_height = parseOutputScale(
        args.output_scale
    )


def validateTrimRange(args):
    inpoint = float(getattr(args, "inpoint", 0) or 0)
    outpoint = float(getattr(args, "outpoint", 0) or 0)

    if outpoint != 0 and outpoint <= inpoint:
        raise CliValidationError(
            f"Invalid trim range: outpoint must be greater than inpoint when set "
            f"(inpoint={inpoint}, outpoint={outpoint})"
        )


def normalizeUpscaleFactor(args):
    if not args.upscale or not hasattr(args, "upscale_factor"):
        return None

    try:
        if int(args.upscale_factor) >= 2:
            return None
    except Exception:
        args.upscale_factor = 2
        return "Invalid upscale_factor provided; defaulting to 2"

    oldFactor = args.upscale_factor
    args.upscale_factor = 2
    return (
        "Upscale factor must be at least 2 when --upscale is enabled; "
        f"defaulting to 2 (was {oldFactor})"
    )


def validatePreviewPort(args):
    """Reject a `--preview_port` outside the valid TCP range.

    argparse bounds nothing here, and ``--json``/``--preset`` assign onto the
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


def validateInterpolateFactor(args):
    """Reject an interpolation factor below 1; leave 1 and above alone.

    Nothing bounded this flag. ``gapPlan`` owes 0 frames for any factor <= 1
    over a unit gap while the frame loop still writes every source frame, so
    ``--interpolate_factor 0.5`` emitted the source frames at half the rate --
    a video twice as long as its untouched audio, exit 0, no warning. 0 and
    negatives reach the encoder as ``-r 0.0`` / ``-r -47.95``.

    Factor 1 is deliberately left enabled: with ``--smooth_dedup`` it is the
    supported path that regenerates each duplicated slot as a true in-between
    at unchanged duration and fps. Disabling ``--interpolate`` here would
    silently switch that feature off.
    """
    if not getattr(args, "interpolate", False):
        return None

    try:
        factor = float(getattr(args, "interpolate_factor", 2))
    except TypeError, ValueError:
        raise CliValidationError(
            f"Invalid --interpolate_factor: {args.interpolate_factor!r}. "
            "Expected a number of at least 1."
        ) from None

    if factor <= 0:
        raise CliValidationError(
            f"--interpolate_factor must be greater than 0 (got {factor:g})."
        )
    if factor < 1:
        raise CliValidationError(
            f"--interpolate_factor must be at least 1 (got {factor:g}). "
            "A factor below 1 inserts no frames and only lowers the output "
            "frame rate, which desynchronizes the audio. To slow footage down "
            "use --slowmo, which keeps the source frame rate."
        )
    if factor == 1 and not getattr(args, "smooth_dedup", False):
        return (
            "--interpolate_factor 1 without --smooth_dedup writes the source "
            "frames unchanged; the interpolation model is still loaded."
        )
    return None


def selectedUpscaleBackend(upscaleMethod):
    backendSuffixes = ("-directml", "-openvino", "-tensorrt", "-ncnn")
    for suffix in backendSuffixes:
        if upscaleMethod.endswith(suffix):
            return suffix[1:], upscaleMethod[: -len(suffix)]
    return "pytorch", upscaleMethod


def validateCustomUpscaleModel(args):
    if not args.custom_model:
        return

    args.custom_model = os.path.abspath(args.custom_model)
    if not os.path.isfile(args.custom_model):
        raise CliValidationError(f"Custom model file not found: {args.custom_model}")

    extension = os.path.splitext(args.custom_model)[1].lower()
    pytorchExtensions = {".pt", ".pth", ".ckpt", ".safetensors"}
    onnxExtensions = {".onnx"}
    selectedBackend, baseMethod = selectedUpscaleBackend(args.upscale_method)

    if extension in onnxExtensions:
        if selectedBackend not in {"directml", "openvino", "tensorrt"}:
            raise CliValidationError(
                "Custom ONNX upscale models require an ONNX backend. Use an upscale method ending in -directml, -openvino, or -tensorrt, for example "
                f"{baseMethod}-directml."
            )
        return

    if extension in pytorchExtensions:
        if selectedBackend != "pytorch":
            raise CliValidationError(
                "Custom PyTorch upscale models require a CUDA/PyTorch upscale method without a backend suffix. "
                f"Use {baseMethod} for .pt/.pth/.ckpt/.safetensors files."
            )
        return

    raise CliValidationError(
        "Unsupported custom upscale model format. Supported extensions are .pt, .pth, .ckpt, .safetensors, and .onnx."
    )


def applyRuntimeValidation(args):
    validateCustomUpscaleModel(args)
    applyOutputScale(args)
    validateTrimRange(args)
    validatePreviewPort(args)
    # Both normalizers can warn; join rather than let one shadow the other.
    warnings = [normalizeUpscaleFactor(args), validateInterpolateFactor(args)]
    return "\n".join(w for w in warnings if w) or None
