import difflib
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
    "smooth_dedup_method": ("smooth_dedup", "ssim"),
    "smooth_dedup_sens": ("smooth_dedup", 35.0),
    "smooth_dedup_max_span": ("smooth_dedup", 6),
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
    parser: object
    argv: list[str]
    providedOptions: set[str]
    jsonKeys: set[str]

    @classmethod
    def fromArgs(cls, args, parser, argv=None):
        argv = list(sys.argv[1:] if argv is None else argv)
        config = cls(
            args=args,
            parser=parser,
            argv=argv,
            providedOptions=cls.collectProvidedOptions(argv),
            jsonKeys=set(),
        )
        config.normalize()
        return config

    @staticmethod
    def collectProvidedOptions(argv):
        """Return normalized long-option names provided on the command line."""
        provided = set()
        for arg in argv:
            if arg.startswith("--"):
                provided.add(arg[2:].split("=", 1)[0].replace("-", "_"))
        return provided

    def optionWasProvided(self, optionName):
        return optionName in self.providedOptions or optionName in self.jsonKeys

    @property
    def parserActionsByDest(self):
        return {
            action.dest: action
            for action in self.parser._actions
            if action.dest not in ["help", "version"]
        }

    @property
    def parserDefaults(self):
        defaults = {}
        for action in self.parser._actions:
            if action.dest not in ["help", "version", "json"]:
                defaults[action.dest] = action.default
        return defaults

    def normalize(self):
        if self.args.json:
            self.mergeJsonConfig()

        self.autoEnableParentFlags()

    def mergeJsonConfig(self):
        extraCliOptions = self.providedOptions - {"json"}

        if extraCliOptions:
            logAndPrint(
                "Cannot use --json with other command line arguments. Use --json alone.",
                "red",
            )
            sys.exit(1)

        jsonConfig = self.loadJsonConfig()
        defaults = self.parserDefaults
        loadedKeys = set()
        for key, value in jsonConfig.items():
            if key == "json":
                continue

            self.validateJsonChoice(key, value)

            if hasattr(self.args, key):
                currentValue = getattr(self.args, key)
                defaultValue = defaults.get(key)

                if currentValue == defaultValue:
                    setattr(self.args, key, value)
                    logging.info(f"Loaded from JSON: {key} = {value}")
                loadedKeys.add(key)
            else:
                logging.warning(f"Unknown option in JSON config: {key}")

        self.jsonKeys.update(loadedKeys)

    def validateJsonChoice(self, key, value):
        """Reject JSON values argparse would have rejected on the command line.

        ``--json`` assigns straight onto the namespace, so it used to bypass
        every ``choices=`` list the parser declares. A bad value therefore
        travelled all the way into the factory before anything noticed: the AE
        bridge sent ``depth_method: "small_v3-directml"``, a method with no CLI
        choice and no backend class, and the run died several seconds later in
        ``src/factories/standalone.py:depth()`` after decoding metadata.
        """
        action = self.parserActionsByDest.get(key)
        if action is None or not action.choices:
            return

        choices = list(action.choices)
        values = value if isinstance(value, list) else [value]
        for item in values:
            if item in choices:
                continue

            message = f"Invalid value for '{key}' in JSON config: {item!r}."
            if len(choices) <= 8:
                message += f" Valid choices: {', '.join(str(c) for c in choices)}."
            else:
                close = difflib.get_close_matches(
                    str(item), [str(c) for c in choices], 3
                )
                if close:
                    message += f" Did you mean: {', '.join(close)}?"
                if key.endswith("_method"):
                    capability = key[: -len("_method")]
                    message += (
                        f" Run --list_methods {capability} to see all "
                        f"{len(choices)} choices."
                    )
                else:
                    message += f" {len(choices)} valid choices, see --help."
            logAndPrint(message, "red")
            sys.exit(1)

    def loadJsonConfig(self):
        jsonPath = os.path.abspath(self.args.json)

        if not os.path.exists(jsonPath):
            logAndPrint(f"JSON config file not found: {jsonPath}", "red")
            sys.exit(1)

        try:
            with open(jsonPath, encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logAndPrint(f"Invalid JSON format in config file: {e}", "red")
            sys.exit(1)
        except Exception as e:
            logAndPrint(f"Error reading JSON config: {e}", "red")
            sys.exit(1)

    def autoEnableParentFlags(self):
        logging.debug("jsonKeys: %s", self.jsonKeys)

        for optionName, (parentFlag, defaultValue) in PARENT_FLAG_DEFAULTS.items():
            if not hasattr(self.args, optionName):
                continue

            currentValue = getattr(self.args, optionName)
            isExplicitlyProvided = self.optionWasProvided(optionName)

            if optionName == "interpolate_method":
                logging.info(
                    f"[DEBUG] interpolate_method - providedOnCLI: {optionName in self.providedOptions}, isExplicitlyProvided: {isExplicitlyProvided}"
                )

            if isExplicitlyProvided:
                if not getattr(self.args, parentFlag):
                    setattr(self.args, parentFlag, True)
                    logging.info(
                        f"Auto-enabling --{parentFlag} because --{optionName} was provided"
                    )
            elif currentValue != defaultValue and not getattr(self.args, parentFlag):
                setattr(self.args, parentFlag, True)
                logging.info(
                    f"Auto-enabling --{parentFlag} because {optionName} differs from default"
                )
