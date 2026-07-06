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
    return normalizeUpscaleFactor(args)
