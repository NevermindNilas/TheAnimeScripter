import difflib
import json
import logging
import os
import sys
from dataclasses import dataclass, field

from src.infra.logAndPrint import logAndPrint


def validateChoiceForKey(parser, key, value, sourceLabel):
    """Reject a value argparse would have rejected on the command line.

    Config sources that assign straight onto the namespace (``--json``,
    ``--preset``) bypass every ``choices=`` list the parser declares, so a
    stale or hand-edited value used to travel all the way into a factory
    before anything noticed. Exits with the same red diagnostic argparse
    would have produced at parse time.

    Returns the value normalized to the parser's canonical casing: the AE
    panel serialized its decode dropdown label as ``"CPU"``, which the JSON
    path tolerated for years before this validation existed, so a
    case-insensitive match is accepted and mapped onto the declared choice.
    Callers must assign the return value, not the original.
    """
    action = next((a for a in parser._actions if a.dest == key), None)
    if action is None or not action.choices:
        return value

    choices = list(action.choices)
    canonicalByLower = {c.lower(): c for c in choices if isinstance(c, str)}
    values = value if isinstance(value, list) else [value]
    normalized = []
    for item in values:
        if item in choices:
            normalized.append(item)
            continue
        if isinstance(item, str) and item.lower() in canonicalByLower:
            canonical = canonicalByLower[item.lower()]
            logging.info(
                f"Normalized '{key}' in {sourceLabel}: {item!r} -> {canonical!r}"
            )
            normalized.append(canonical)
            continue

        message = f"Invalid value for '{key}' in {sourceLabel}: {item!r}."
        if len(choices) <= 8:
            message += f" Valid choices: {', '.join(str(c) for c in choices)}."
        else:
            close = difflib.get_close_matches(str(item), [str(c) for c in choices], 3)
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

    return normalized if isinstance(value, list) else normalized[0]


_TYPE_LABELS = {int: "a whole number", float: "a number", str: "a string"}


def _expectedLabel(converter):
    """A human-readable name for what a parser ``type=`` converter accepts."""
    if converter in _TYPE_LABELS:
        return _TYPE_LABELS[converter]
    if getattr(converter, "__name__", "") == "str2bool":
        return "true or false"
    return f"a value accepted by {getattr(converter, '__name__', converter)}"


def coerceValueForKey(parser, key, value, sourceLabel):
    """Apply the parser's own type conversion to a config-file value.

    Companion to :func:`validateChoiceForKey`, which guards ``choices=`` only.
    ``--json`` and ``--preset`` assign JSON scalars straight onto the
    namespace, so nothing ever ran ``action.type`` over them and a panel or
    hand-written file that quoted a number carried a ``str`` into the run: it
    died at ``main.py``'s ``self.fps * self.interpolateFactor`` with "can't
    multiply sequence by non-int of type 'float'", naming neither the option
    nor the file.

    The worse half is silent. ``store_true`` flags carry no ``type`` at all,
    so ``{"interpolate": "false"}`` used to assign the *string* ``"false"`` --
    truthy -- and turned interpolation on for a config that asked for it off.

    Exits 1 with the same red diagnostic style as the choices check. Returns
    the coerced value; callers must assign it.
    """
    action = next((a for a in parser._actions if a.dest == key), None)
    if action is None or value is None:
        return value

    def fail(item, expected):
        logAndPrint(
            f"Invalid value for '{key}' in {sourceLabel}: {item!r}. "
            f"Expected {expected}.",
            "red",
        )
        sys.exit(1)

    # store_true/store_false: argparse gives them no converter because the
    # command line never supplies a value. A config file can, and does.
    if action.nargs == 0:
        if isinstance(value, bool):
            return value
        # Keep accepting 0/1 and "true"/"yes"/... -- panels already send those
        # and they have always worked by truthiness; only ambiguity is an error.
        if isinstance(value, int) and value in (0, 1):
            return bool(value)
        if isinstance(value, str):
            from src.cli.parser import str2bool

            try:
                return str2bool(value)
            except Exception:
                fail(value, "true or false")
        fail(value, "true or false")

    converter = action.type
    if converter is None:
        return value

    values = value if isinstance(value, list) else [value]
    converted = []
    for item in values:
        # `isinstance(item, converter)` is only meaningful when the converter
        # is a real class. `--half` and `--interpolate_first` declare
        # `type=str2bool`, a plain function, and isinstance() against it raises
        # TypeError -- which would break every preset load, since presets store
        # all of vars(args). `type(x) is C` also sidesteps bool-is-an-int.
        if isinstance(converter, type) and type(item) is converter:
            converted.append(item)
            continue
        try:
            converted.append(converter(item))
        except Exception:
            # str2bool raises argparse.ArgumentTypeError, which is not a
            # ValueError, so this catch has to stay broad.
            fail(item, _expectedLabel(converter))

    return converted if isinstance(value, list) else converted[0]


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
    "stabilize_method": ("stabilize", "classic"),
    "depth_method": ("depth", "small_v2"),
    "obj_detect_method": ("obj_detect", "yolov9_small-directml"),
    "resize_factor": ("resize", 2),
    # --output_scale is deliberately absent. It is an encoder-side `scale=`
    # filter (WriteBuffer._buildFilterList) and needs no capability flag, but
    # mapping it onto "resize" here also handed it --resize_factor's default of
    # 2: `--upscale --output_scale 1280x720` on a 640x360 source bilinearly
    # doubled the DECODE to 1280x720, ran the model over 4x the pixels to
    # 2560x1440, then made FFmpeg scale back down. Slower, more VRAM, softer.
    # isAnyOtherProcessingMethodEnabled counts it directly instead, so
    # --output_scale on its own is still a valid run.
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
    # Every recognized key the JSON actually contained, whatever its value.
    # jsonKeys holds only the ones that ask for something (value != default);
    # this one answers "did the config mention this at all", which is what an
    # explicit "upscale": false needs to beat a sibling "upscale_method".
    jsonPresentKeys: set[str] = field(default_factory=set)

    @classmethod
    def fromArgs(cls, args, parser, argv=None):
        argv = list(sys.argv[1:] if argv is None else argv)
        config = cls(
            args=args,
            parser=parser,
            argv=argv,
            providedOptions=cls.collectProvidedOptions(argv),
            jsonKeys=set(),
            jsonPresentKeys=set(),
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
        presentKeys = set()
        for key, value in jsonConfig.items():
            if key == "json":
                continue

            # Coerce first: `"upscale_factor": "2"` is a valid choice once it
            # is an int, and reporting it as an invalid choice against
            # `2, 3, 4` reads like nonsense.
            value = coerceValueForKey(self.parser, key, value, "JSON config")
            value = self.validateJsonChoice(key, value)

            if hasattr(self.args, key):
                presentKeys.add(key)
                currentValue = getattr(self.args, key)
                defaultValue = defaults.get(key)

                if currentValue == defaultValue:
                    setattr(self.args, key, value)
                    logging.info(f"Loaded from JSON: {key} = {value}")
                if value != defaultValue:
                    # Only a key that actually asks for something counts as
                    # "provided". The After Effects panel serializes its whole
                    # form, so every capability's *_method arrives at its
                    # default value -- and marking those provided made
                    # autoEnableParentFlags turn on upscale, depth, segment,
                    # dedup and restore for a config that asked for
                    # interpolation alone.
                    loadedKeys.add(key)
            else:
                # logging.warning alone only reaches TAS-Log.log, so a typo'd
                # key changed nothing and said nothing. The preset path has
                # printed this since 4fd967c4; match it.
                message = f"Unknown option in JSON config: '{key}'; ignoring it."
                close = difflib.get_close_matches(key, list(defaults), 1)
                if close:
                    message += f" Did you mean '{close[0]}'?"
                logAndPrint(message, "yellow")

        self.jsonKeys.update(loadedKeys)
        self.jsonPresentKeys.update(presentKeys)

    def validateJsonChoice(self, key, value):
        """Reject JSON values argparse would have rejected on the command line.

        ``--json`` assigns straight onto the namespace, so it used to bypass
        every ``choices=`` list the parser declares. A bad value therefore
        travelled all the way into the factory before anything noticed: the AE
        bridge sent ``depth_method: "small_v3-directml"``, a method with no CLI
        choice and no backend class, and the run died several seconds later in
        ``src/factories/standalone.py:depth()`` after decoding metadata.

        Returns the value with any case-insensitive matches normalized to the
        parser's canonical casing (the AE panel sends ``decode_method: "CPU"``).
        """
        return validateChoiceForKey(self.parser, key, value, "JSON config")

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

            if parentFlag in self.jsonPresentKeys and not getattr(
                self.args, parentFlag
            ):
                # The config named the capability itself and said off. A sibling
                # key must not overrule that: a JSON carrying both
                # "upscale": false and "upscale_method" used to upscale anyway.
                logging.info(
                    f"Not auto-enabling --{parentFlag}: the config sets it explicitly"
                )
                continue

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
