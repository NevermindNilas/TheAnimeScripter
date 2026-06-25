import json
import logging
import os
import sys
from dataclasses import dataclass

from src.infra.logAndPrint import logAndPrint

PARENT_FLAG_DEFAULTS = {
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
    "moblur_method": ("moblur", "rife4.25"),
    "moblur_factor": ("moblur", 8),
    "moblur_strength": ("moblur", "gaussian_sym"),
    "moblur_shutter_angle": ("moblur", 180.0),
}


@dataclass(frozen=True)
class CliConfig:
    args: object
    providedOptions: set[str]
    jsonKeys: set[str]


def providedOptions(argv):
    """Return normalized long-option names provided on the command line."""
    provided = set()
    for arg in argv:
        if arg.startswith("--"):
            provided.add(arg[2:].split("=", 1)[0].replace("-", "_"))
    return provided


def optionWasProvided(optionName, cliOptions, jsonKeys=None):
    return optionName in cliOptions or optionName in (jsonKeys or set())


def parserDefaults(parser):
    defaults = {}
    for action in parser._actions:
        if action.dest not in ["help", "version", "json"]:
            defaults[action.dest] = action.default
    return defaults


def mergeJsonConfig(args, parser, argv=None):
    argv = sys.argv[1:] if argv is None else argv
    cliOptions = providedOptions(argv)
    extraCliOptions = cliOptions - {"json"}

    if extraCliOptions:
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
        with open(jsonPath, encoding="utf-8") as f:
            jsonConfig = json.load(f)
    except json.JSONDecodeError as e:
        logAndPrint(f"Invalid JSON format in config file: {e}", "red")
        sys.exit()
    except Exception as e:
        logAndPrint(f"Error reading JSON config: {e}", "red")
        sys.exit()

    defaults = parserDefaults(parser)
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

    return loadedKeys


def autoEnableParentFlags(args, cliOptions, jsonKeys=None):
    logging.info(f"[DEBUG] jsonKeys: {jsonKeys}")

    for optionName, (parentFlag, defaultValue) in PARENT_FLAG_DEFAULTS.items():
        if not hasattr(args, optionName):
            continue

        currentValue = getattr(args, optionName)
        isExplicitlyProvided = optionWasProvided(optionName, cliOptions, jsonKeys)

        if optionName == "interpolate_method":
            logging.info(
                f"[DEBUG] interpolate_method - providedOnCLI: {optionName in cliOptions}, isExplicitlyProvided: {isExplicitlyProvided}"
            )

        if isExplicitlyProvided:
            if not getattr(args, parentFlag):
                setattr(args, parentFlag, True)
                logging.info(
                    f"Auto-enabling --{parentFlag} because --{optionName} was provided"
                )
        elif currentValue != defaultValue and not getattr(args, parentFlag):
            setattr(args, parentFlag, True)
            logging.info(
                f"Auto-enabling --{parentFlag} because {optionName} differs from default"
            )


def normalizeCliConfig(args, parser, argv=None):
    argv = sys.argv[1:] if argv is None else argv
    cliOptions = providedOptions(argv)
    jsonKeys = set()

    if args.json:
        jsonKeys = mergeJsonConfig(args, parser, argv)

    autoEnableParentFlags(args, cliOptions, jsonKeys)

    return CliConfig(args=args, providedOptions=cliOptions, jsonKeys=jsonKeys)
