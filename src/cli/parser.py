import argparse
import os
import sys
from difflib import get_close_matches

from src.version import __version__

_logAndPrint = None


def logAndPrint(message, colorFunc="cyan", level="INFO"):
    global _logAndPrint
    if _logAndPrint is None:
        from src.infra.logAndPrint import logAndPrint as _lap

        _logAndPrint = _lap
    _logAndPrint(message, colorFunc, level)


def _supportsColorStdout():
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
        return False
    if sys.platform == "win32":
        try:
            return sys.getwindowsversion().build >= 14393
        except Exception:
            return False
    return True


def str2bool(arg):
    if isinstance(arg, bool):
        return arg
    if arg.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif arg.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


class DidYouMeanArgumentParser(argparse.ArgumentParser):
    """Custom ArgumentParser that provides "did you mean?" suggestions for invalid choices."""

    def getSuggestions(self, invalidValue, validChoices, maxSuggestions=5):
        choices = {choice.lower(): choice for choice in validChoices}
        matches = get_close_matches(
            invalidValue.lower(), choices, n=maxSuggestions, cutoff=0.3
        )
        return [choices[match] for match in matches]

    def _palette(self):
        useColor = _supportsColorStdout()

        def code(c):
            return c if useColor else ""

        return {
            "RED": code("\x1b[1;31m"),
            "YELLOW": code("\x1b[1;33m"),
            "GREEN": code("\x1b[1;32m"),
            "CYAN": code("\x1b[1;36m"),
            "DIM": code("\x1b[2m"),
            "RESET": code("\x1b[0m"),
        }

    def _collectOptionStrings(self):
        options = []
        for action in self._actions:
            options.extend(action.option_strings)
        return options

    def _suggestOptions(self, invalidOption, validOptions, maxSuggestions=3):
        isLong = invalidOption.startswith("--")
        invalidCore = invalidOption.lstrip("-")
        options = {
            option.lstrip("-").lower(): option
            for option in validOptions
            if option.startswith("--") == isLong
        }
        matches = get_close_matches(
            invalidCore.lower(), options, n=maxSuggestions, cutoff=0.4
        )
        return [options[match] for match in matches]

    def _suggestOptionNames(self, message):
        tokens = message[len("unrecognized arguments:") :].split()
        badOptions = [t for t in tokens if t.startswith("-") and len(t) > 1]
        if not badOptions:
            return False

        validOptions = self._collectOptionStrings()
        c = self._palette()

        print(
            f"\n  {c['RED']}Error:{c['RESET']} unrecognized "
            f"{'option' if len(badOptions) == 1 else 'options'}: "
            f"{c['RED']}{' '.join(badOptions)}{c['RESET']}",
            file=sys.stderr,
        )

        for bad in badOptions:
            core = bad.split("=", 1)[0]
            suggestions = self._suggestOptions(core, validOptions)
            if suggestions:
                print(
                    f"\n  {c['YELLOW']}Did you mean:{c['RESET']} "
                    f"{c['GREEN']}{', '.join(suggestions)}{c['RESET']} "
                    f"{c['DIM']}(instead of '{core}'){c['RESET']}",
                    file=sys.stderr,
                )

        print(
            f"\n  {c['CYAN']}Tip:{c['RESET']} {c['DIM']}run "
            f"'main.py --help' for the full option list.{c['RESET']}",
            file=sys.stderr,
        )

        sys.exit(2)

    def error(self, message):
        if "invalid choice:" in message and "choose from" in message:
            import re

            match = re.search(
                r"argument (--?[\w-]+):\s*invalid choice:\s*'([^']+)'\s*\(choose from\s*(.+)\)",
                message,
            )

            if match:
                argName = match.group(1)
                invalidValue = match.group(2)
                choicesStr = match.group(3)

                choices = [
                    c.strip().strip("'\"") for c in choicesStr.split(",") if c.strip()
                ]

                if choices:
                    suggestions = self.getSuggestions(invalidValue, choices)

                    useColor = _supportsColorStdout()
                    RED = "\x1b[1;31m" if useColor else ""
                    YELLOW = "\x1b[1;33m" if useColor else ""
                    GREEN = "\x1b[1;32m" if useColor else ""
                    CYAN = "\x1b[1;36m" if useColor else ""
                    DIM = "\x1b[2m" if useColor else ""
                    RESET = "\x1b[0m" if useColor else ""

                    print(
                        f"\n  {RED}Error:{RESET} argument {argName}: "
                        f"invalid choice: {RED}'{invalidValue}'{RESET}",
                        file=sys.stderr,
                    )

                    if suggestions:
                        print(
                            f"\n  {YELLOW}Did you mean:{RESET} "
                            f"{GREEN}{', '.join(repr(s) for s in suggestions)}{RESET}",
                            file=sys.stderr,
                        )

                    displayedChoices = choices[:10]
                    choicesStrDisplay = ", ".join(repr(c) for c in displayedChoices)
                    if len(choices) > 10:
                        choicesStrDisplay += f", ... ({len(choices) - 10} more)"
                    print(
                        f"\n  {CYAN}Valid choices:{RESET} {DIM}{choicesStrDisplay}{RESET}",
                        file=sys.stderr,
                    )

                    sys.exit(2)
                    return

        if message.startswith("unrecognized arguments:"):
            self._suggestOptionNames(message)

        c = self._palette()
        print(f"\n  {c['RED']}Error:{c['RESET']} {message}", file=sys.stderr)
        sys.exit(2)


class TASHelpFormatter(argparse.HelpFormatter):
    """Compact, colored help formatter that groups long choice lists by backend."""

    _INLINE_THRESHOLD = 6
    _KNOWN_BACKENDS = ("tensorrt", "directml", "ncnn", "openvino")

    def __init__(self, prog, indent_increment=2, max_help_position=36, width=None):
        if width is None:
            try:
                width = os.get_terminal_size().columns
            except (ValueError, OSError) as _e:
                width = 100
        super().__init__(
            prog, indent_increment, max_help_position, min(max(width, 60), 120)
        )

    def _metavar_formatter(self, action, default_metavar):
        if action.choices is not None:
            choices = [str(c) for c in action.choices]
            if len(choices) > self._INLINE_THRESHOLD:
                result = action.dest.upper().rsplit("_", 1)[-1]
            else:
                result = "{{{}}}".format(", ".join(choices))

            def format(tuple_size):
                return (result,) * tuple_size

            return format
        return super()._metavar_formatter(action, default_metavar)

    def _get_help_string(self, action):
        help_text = action.help or ""
        extras = []
        if (
            action.default is not None
            and action.default is not argparse.SUPPRESS
            and action.default not in (None, False, [], "")
            and not isinstance(action.default, bool)
            and action.option_strings
            and "default:" not in help_text.lower()
            and "default is" not in help_text.lower()
        ):
            if isinstance(action.default, list):
                default_str = ", ".join(str(d) for d in action.default)
            else:
                default_str = str(action.default)
            extras.append(f"default: {default_str}")
        if extras:
            help_text += " ({})".format(", ".join(extras))
        return help_text

    def _format_action(self, action):
        result = super()._format_action(action)
        if not action.choices or len(list(action.choices)) <= self._INLINE_THRESHOLD:
            return result
        indent = " " * self._max_help_position
        groups = self._group_choices([str(c) for c in action.choices])
        lines = []
        if len(groups) == 1:
            items_str = ", ".join(list(groups.values())[0])
            lines.append(indent + items_str)
        else:
            for backend, items in groups.items():
                label = f"{f'[{backend}]':<10} "
                items_str = ", ".join(items)
                lines.append(indent + label + items_str)
        return result + "\n".join(lines) + "\n"

    @classmethod
    def _group_choices(cls, choices):
        from collections import OrderedDict

        groups = OrderedDict()
        for choice in choices:
            parts = choice.rsplit("-", 1)
            if len(parts) == 2 and parts[1] in cls._KNOWN_BACKENDS:
                backend = parts[1]
            else:
                backend = "cuda"
            groups.setdefault(backend, []).append(choice)
        return groups

    def format_help(self):
        text = super().format_help()

        if "usage:" not in text.lower():
            return text

        isUsageOnly = "\n\n" not in text.strip("\n")

        color = _supportsColorStdout()

        if isUsageOnly:
            header = ""
        elif color:
            R = "\033[0m"
            B = "\033[1m"
            C = "\033[96m"
            D = "\033[2m"
            header = (
                f"\n  {B}{C}The Anime Scripter{R} {D}v{__version__}{R}\n"
                f"  {D}AI-powered video enhancement toolkit{R}\n\n"
            )
        else:
            header = (
                f"\n  The Anime Scripter v{__version__}\n"
                f"  AI-powered video enhancement toolkit\n\n"
            )

        if not color:
            return header + text

        import re

        R = "\033[0m"
        B = "\033[1m"
        C = "\033[96m"
        G = "\033[92m"
        Y = "\033[93m"

        lines = text.split("\n")
        out = []
        for line in lines:
            m = re.match(r"^(\s{0,2})([\w][\w /()-]*):(\s*)$", line)
            if m:
                out.append(f"{m.group(1)}{B}{C}{m.group(2)}{R}")
                continue

            line = re.sub(r"^(usage:\s*|Usage:\s*)", f"{B}\\1{R}", line)
            line = re.sub(r"(--[\w][\w-]*)", f"{G}\\1{R}", line)
            line = re.sub(r"(?<=[\s,])(-[a-zA-Z])(?=[\s,\]])", f"{G}\\1{R}", line)
            line = re.sub(r"(\{[^}]+\})", f"{Y}\\1{R}", line)

            out.append(line)

        return header + "\n".join(out)


def capabilityMethods(parser, excluded=("decode_method",)):
    excluded = set(excluded)
    capabilities = {}
    for action in parser._actions:
        if (
            action.dest.endswith("_method")
            and action.choices
            and action.dest not in excluded
        ):
            capability = action.dest[: -len("_method")]
            capabilities[capability] = [str(choice) for choice in action.choices]
    return capabilities


def _listMethods(parser, requested):
    capabilities = capabilityMethods(parser)

    palette = parser._palette() if hasattr(parser, "_palette") else {}
    CYAN = palette.get("CYAN", "")
    DIM = palette.get("DIM", "")
    GREEN = palette.get("GREEN", "")
    YELLOW = palette.get("YELLOW", "")
    RED = palette.get("RED", "")
    RESET = palette.get("RESET", "")

    requested = (requested or "all").lower()

    if requested != "all" and requested not in capabilities:
        print(f"{RED}Unknown capability:{RESET} '{requested}'", file=sys.stderr)
        suggestions = (
            parser.getSuggestions(requested, list(capabilities))
            if hasattr(parser, "getSuggestions")
            else []
        )
        if suggestions:
            print(
                f"  {YELLOW}Did you mean:{RESET} "
                f"{GREEN}{', '.join(suggestions)}{RESET}",
                file=sys.stderr,
            )
        print(
            f"  {DIM}Capabilities: {', '.join(sorted(capabilities))}{RESET}",
            file=sys.stderr,
        )
        return 2

    targets = (
        capabilities if requested == "all" else {requested: capabilities[requested]}
    )

    for capability, methods in targets.items():
        print(f"\n{CYAN}{capability}{RESET} {DIM}({len(methods)} methods){RESET}")
        groups = TASHelpFormatter._group_choices(methods)
        if len(groups) == 1:
            print("  " + ", ".join(next(iter(groups.values()))))
        else:
            for backend, items in groups.items():
                label = f"{f'[{backend}]':<10}"
                print(f"  {DIM}{label}{RESET} {', '.join(items)}")

    return 0


def _buildParser(outputPath):
    argParser = DidYouMeanArgumentParser(
        usage="main.py [options]",
        formatter_class=TASHelpFormatter,
    )

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

    presetGroup = argParser.add_argument_group("Preset Configuration")
    presetGroup.add_argument(
        "--preset",
        type=str,
        help="Create and use a preset configuration file based on the current arguments",
    )
    presetGroup.add_argument(
        "--list_presets", action="store_true", help="List all available presets"
    )
    presetGroup.add_argument(
        "--list_methods",
        type=str,
        nargs="?",
        const="all",
        default=None,
        metavar="CAPABILITY",
        help="List available methods for a capability (e.g. --list_methods upscale) "
        "or every capability when used alone, then exit.",
    )

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

    _addInterpolationOptions(argParser)
    _addUpscalingOptions(argParser)
    _addDedupOptions(argParser)
    _addVideoProcessingOptions(argParser)
    _addMotionBlurOptions(argParser)
    _addMaskOptions(argParser)
    _addSegmentationOptions(argParser)
    _addSceneDetectionOptions(argParser)
    _addDepthOptions(argParser)
    _addEncodingOptions(argParser)

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
            "yolov9_small-tensorrt",
            "yolov9_medium-tensorrt",
            "yolov9_large-tensorrt",
        ],
    )
    objectGroup.add_argument(
        "--obj_detect_disable_annotations",
        type=str2bool,
        default=False,
        help="Disable class labels and confidence percentages on detection boxes (default: False)",
    )

    _addMiscOptions(argParser)

    return argParser


def createParser(outputPath):
    from src.cli.validator import prepareRuntimeArgs

    argParser = _buildParser(outputPath)
    args = argParser.parse_args()
    return prepareRuntimeArgs(args, outputPath, argParser)


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
        "distildrba-directml",
        "distildrba-lite-directml",
        "distildrba-openvino",
        "distildrba-lite-openvino",
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
        "rife_elexor",
        "rife_elexor-tensorrt",
        "rife4.6-directml",
        "rife4.15-directml",
        "rife4.17-directml",
        "rife4.18-directml",
        "rife4.20-directml",
        "rife4.21-directml",
        "rife4.22-directml",
        "rife4.22-lite-directml",
        "rife4.25-directml",
        "rife4.25-lite-directml",
        "rife4.25-heavy-directml",
        "rife_elexor-directml",
        "rife_elexor-openvino",
        "rife4.6-openvino",
        "rife4.15-openvino",
        "rife4.17-openvino",
        "rife4.18-openvino",
        "rife4.20-openvino",
        "rife4.21-openvino",
        "rife4.22-openvino",
        "rife4.22-lite-openvino",
        "rife4.25-openvino",
        "rife4.25-lite-openvino",
        "rife4.25-heavy-openvino",
        "rife-mps",
        "rife4.6-mps",
        "rife4.15-lite-mps",
        "rife4.16-lite-mps",
        "rife4.17-mps",
        "rife4.18-mps",
        "rife4.20-mps",
        "rife4.21-mps",
        "rife4.22-mps",
        "rife4.22-lite-mps",
        "rife4.25-mps",
        "rife4.25-lite-mps",
        "rife4.25-heavy-mps",
        "rife_elexor-mps",
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
        "adore",
        "adore-tensorrt",
        "adore-directml",
        "adore-openvino",
        "fallin_soft",
        "fallin_soft-tensorrt",
        "fallin_soft-directml",
        "fallin_strong",
        "fallin_strong-tensorrt",
        "fallin_strong-directml",
        "span",
        "shufflespan-directml",
        "span-directml",
        "shufflecugan-ncnn",
        "shufflecugan-directml",
        "shufflecugan-openvino",
        "span-ncnn",
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
        "figsr",
        "smosr",
        "smosr-tensorrt",
        "smosr-directml",
        "smosr-openvino",
        "artcnn_c4f16-tensorrt",
        "artcnn_c4f16-directml",
        "artcnn_c4f16-openvino",
        "artcnn_c4f16_dn-tensorrt",
        "artcnn_c4f16_dn-directml",
        "artcnn_c4f16_dn-openvino",
        "artcnn_c4f16_ds-tensorrt",
        "artcnn_c4f16_ds-directml",
        "artcnn_c4f16_ds-openvino",
        "artcnn_c4f32-tensorrt",
        "artcnn_c4f32-directml",
        "artcnn_c4f32-openvino",
        "artcnn_c4f32_dn-tensorrt",
        "artcnn_c4f32_dn-directml",
        "artcnn_c4f32_dn-openvino",
        "artcnn_c4f32_ds-tensorrt",
        "artcnn_c4f32_ds-directml",
        "artcnn_c4f32_ds-openvino",
        "artcnn_r8f64-tensorrt",
        "artcnn_r8f64-directml",
        "artcnn_r8f64-openvino",
        "artcnn_r16f96-tensorrt",
        "artcnn_r16f96-directml",
        "artcnn_r16f96-openvino",
        "shufflecugan-mps",
        "adore-mps",
        "span-mps",
        "open-proteus-mps",
        "aniscale2-mps",
        "shufflespan-mps",
        "rtmosr-mps",
        "saryn-mps",
        "fallin_soft-mps",
        "fallin_strong-mps",
        "gauss-mps",
        "figsr-mps",
        "smosr-mps",
        "maxine-bicubic",
        "maxine-low",
        "maxine-medium",
        "maxine-high",
        "maxine-ultra",
        "maxine-highbitrate_low",
        "maxine-highbitrate_medium",
        "maxine-highbitrate_high",
        "maxine-highbitrate_ultra",
    ]

    upscaleGroup.add_argument(
        "--upscale_method",
        type=str,
        choices=upscaleMethods,
        default="shufflecugan",
        help="Upscaling method",
    )
    upscaleGroup.add_argument(
        "--custom_model",
        type=str,
        default="",
        help="Path to a custom upscale model. Use .pt/.pth/.ckpt/.safetensors with CUDA methods and .onnx with -directml, -openvino, or -tensorrt methods",
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
        "--restore", action="store_true", help="Restore the video"
    )
    processingGroup.add_argument(
        "--stabilize", action="store_true", help="Stabilize the video"
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
        "autocas",
        "gater3",
        "gater3-directml",
        "gater3-openvino",
        "deepdeband-f",
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
        "dehalo",
        "dehalo-tensorrt",
        "dehalo-directml",
        "dehalo-openvino",
        "scunet-mps",
        "nafnet-mps",
        "dpir-mps",
        "real-plksr-mps",
        "anime1080fixer-mps",
        "gater3-mps",
        "deh264_real-mps",
        "deh264_span-mps",
        "hurrdeblur-mps",
        "dehalo-mps",
        "linethinner-lite",
        "linethinner-medium",
        "linethinner-heavy",
        "linethinner-lite-cuda",
        "linethinner-medium-cuda",
        "linethinner-heavy-cuda",
        "maxine-denoise_low",
        "maxine-denoise_medium",
        "maxine-denoise_high",
        "maxine-denoise_ultra",
        "maxine-deblur_low",
        "maxine-deblur_medium",
        "maxine-deblur_high",
        "maxine-deblur_ultra",
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


def _addMotionBlurOptions(argParser):
    moblurGroup = argParser.add_argument_group("Motion Blur")
    moblurGroup.add_argument(
        "--moblur",
        action="store_true",
        help="Add motion blur via interpolation and frame blending",
    )
    moblurGroup.add_argument(
        "--moblur_method",
        type=str,
        default="rife4.25",
        choices=[
            "rife4.6",
            "rife4.25",
            "rife4.6-directml",
            "rife4.25-directml",
            "rife4.6-tensorrt",
            "rife4.25-tensorrt",
            "rife4.6-mps",
            "rife4.25-mps",
        ],
        help="Interpolation model + backend used to synthesize the motion-blur samples. "
        "rife4.25 is sharper/slower, rife4.6 is faster. Suffix picks the backend: "
        "none = CUDA (NVIDIA; the only path that gets the windowed-sample speedup), "
        "-tensorrt = CUDA via TensorRT, -directml = AMD/Intel GPUs, -mps = Apple Silicon.",
    )
    moblurGroup.add_argument(
        "--moblur_factor",
        type=int,
        default=8,
        help="Interpolation factor for motion blur sample generation (8-16 recommended; higher = smoother trails)",
    )
    moblurGroup.add_argument(
        "--moblur_strength",
        type=str,
        default="gaussian_sym",
        choices=["equal", "gaussian_sym", "pyramid", "ascending", "descending"],
        help="Frame-weighting scheme. gaussian_sym mimics natural shutter falloff",
    )
    moblurGroup.add_argument(
        "--moblur_shutter_angle",
        type=float,
        default=180.0,
        help="Virtual shutter angle in degrees (0-360). 180 = cinema standard, 360 = max smear, 90 = crisp, 0 = disabled",
    )
    moblurGroup.add_argument(
        "--moblur_no_linear_blend",
        action="store_true",
        help="Disable gamma-correct (linear-light) blending. Not recommended -- produces muddy highlights and lifted shadows",
    )


def _addMaskOptions(argParser):
    maskGroup = argParser.add_argument_group("Masking")
    maskGroup.add_argument(
        "--mask",
        type=str,
        default="",
        help="Path to a transparent PNG marking regions to protect. Paint opaque dark pixels on a transparent "
        "background -- those areas stay sharp and static (HUD/text/subtitles), transparent areas are processed "
        "normally. Applies to --moblur (protected areas are not blurred) and --interpolate (protected areas are "
        "not morphed between source frames).",
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
    segmentationGroup.add_argument(
        "--segment_batch",
        type=int,
        default=1,
        help="Number of frames processed per model forward. 1 = default (one "
        "frame at a time). Higher values raise throughput by amortizing "
        "kernel-launch overhead, at a proportional VRAM cost. Supported on the "
        "CUDA and TensorRT backends; forced to 1 for DirectML/OpenVINO, where "
        "batching measured slower.",
    )


def _addSceneDetectionOptions(argParser):
    sceneGroup = argParser.add_argument_group("Scene Detection")
    sceneGroup.add_argument(
        "--autoclip", action="store_true", help="Detect scene changes"
    )
    sceneGroup.add_argument(
        "--autoclip_method",
        type=str,
        default="pyscenedetect",
        choices=[
            "pyscenedetect",
            "maxxvit-directml",
            "maxxvit-tensorrt",
            "transnetv2",
        ],
        help="Autoclip detection backend",
    )
    sceneGroup.add_argument(
        "--autoclip_sens", type=float, default=50, help="Autoclip sensitivity"
    )
    sceneGroup.add_argument(
        "--scenechange",
        action="store_true",
        help="Skip interpolating across hard scene cuts (hold the frame instead "
        "of morphing across the cut). Requires --interpolate.",
    )
    sceneGroup.add_argument(
        "--scenechange_method",
        type=str,
        default="ssim-cuda",
        choices=[
            "ssim",
            "ssim-cuda",
            "mse",
            "mse-cuda",
            "maxxvit-tensorrt",
            "maxxvit-directml",
        ],
        help="Streaming scene-cut detector for --scenechange. Cheap: "
        "ssim/mse (no model). Classifier: maxxvit-tensorrt/-directml.",
    )
    sceneGroup.add_argument(
        "--scenechange_sens",
        type=float,
        default=50,
        help="Scene-cut sensitivity for --scenechange (0-100, higher = more "
        "cuts detected). Mapped to a per-method threshold.",
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
        "small_v2-mps",
        "base_v2-mps",
        "large_v2-mps",
        "giant_v2-mps",
        "distill_small_v2-mps",
        "distill_base_v2-mps",
        "distill_large_v2-mps",
        "og_small_v2",
        "og_base_v2",
        "og_large_v2",
        "og_giant_v2",
        "og_distill_small_v2",
        "og_distill_base_v2",
        "og_distill_large_v2",
        "og_small_v2-mps",
        "og_base_v2-mps",
        "og_large_v2-mps",
        "og_giant_v2-mps",
        "og_distill_small_v2-mps",
        "og_distill_base_v2-mps",
        "og_distill_large_v2-mps",
        "og_video_small_v2",
        "og_video_base_v2",
        "og_video_large_v2",
        "video_small_v2",
        "video_large_v2",
        "video_small_v2-tensorrt",
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
        "og_large_v3",
        "small_v3-mps",
        "base_v3-mps",
        "large_v3-mps",
        "og_large_v3-mps",
        "small_v3-directml",
        "base_v3-directml",
        "small_v3-tensorrt",
        "base_v3-tensorrt",
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
    depthGroup.add_argument(
        "--depth_norm",
        action="store_true",
        help="Apply ghost-free global affine stabilization to image depth methods, or shared-range calibration to temporal video depth methods",
    )
    depthGroup.add_argument(
        "--depth_window",
        type=int,
        choices=[4, 8, 16, 32],
        default=32,
        help="Temporal attention window (frames) for video depth methods (video_*). "
        "32 = full quality (default); 16 is ~1.24x faster forward with no added "
        "flicker at a slight quality tradeoff; 8 is faster still. CUDA-only, video_* methods.",
    )
    depthGroup.add_argument(
        "--depth_batch",
        type=int,
        default=1,
        help="Number of frames processed per model forward for the image depth "
        "methods. 1 = default (one frame at a time). Higher values raise "
        "throughput at lower resolutions where the model is launch-bound "
        "(e.g. --depth_quality low/medium) at a small VRAM cost; negligible "
        "gain at high quality. Supported on the CUDA, MPS, and TensorRT "
        "image backends; forced to 1 for video_* methods, "
        "distill TensorRT, and the DirectML/OpenVINO backends.",
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
        "nvenc_h264_nelux",
        "nvenc_h265_nelux",
        "nvenc_av1_nelux",
        "x264_nelux",
        "x265_nelux",
        "av1_nelux",
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
        type=str,
        nargs="?",
        const="",
        default=None,
        metavar="PROFILE",
        help=(
            "Download dependencies for a runtime profile. Supported profiles: "
            "windows-cuda, windows-lite, linux-cuda, linux-lite, macos-mps, "
            "macos-lite. When used "
            "without a profile, prompts for the current OS full CUDA / TensorRT "
            "or lite dependencies, with guidance for newer NVIDIA GPUs, Apple "
            "Silicon MPS, and CPU-only hardware."
        ),
    )
    miscGroup.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove unused libraries from the script, in case if there were migrations or changes in the script",
    )
