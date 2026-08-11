import json
import os

import src.constants as cs
from src.infra.logAndPrint import logAndPrint


def createPreset(args, providedOptions=None, parser=None):
    ignoreList = ["input", "output", "inpoint", "outpoint"]
    # Flags the user typed on the command line must beat the preset (explicit
    # CLI flag > preset > default). Without this a loaded preset silently
    # clobbers every flag, because a preset stores all of vars(args), not just
    # the ones the user cared about when saving.
    providedOptions = providedOptions or set()
    presetsPath = os.path.join(cs.WHEREAMIRUNFROM, "presets.json")

    if not os.path.exists(presetsPath):
        with open(presetsPath, "w") as file:
            json.dump({"Presets": {}}, file)

    with open(presetsPath, "r+") as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError:
            data = {"Presets": {}}

        presets = data.get("Presets", {})

        if args.preset in presets:
            preset = presets[args.preset]
            for key, value in preset.items():
                if key in ignoreList or key in providedOptions:
                    continue
                # presets.json survives TAS upgrades, so an old preset can name
                # options or method values the current parser no longer has.
                # Validate like --json does instead of assigning blindly.
                if not hasattr(args, key):
                    logAndPrint(
                        f"Preset '{args.preset}' contains unknown option "
                        f"'{key}'; ignoring it.",
                        "yellow",
                    )
                    continue
                if parser is not None:
                    from src.cli.config import validateChoiceForKey

                    validateChoiceForKey(
                        parser, key, value, f"preset '{args.preset}'"
                    )
                setattr(args, key, value)
        else:
            filteredArgs = {k: v for k, v in vars(args).items() if k not in ignoreList}
            presets[args.preset] = filteredArgs
            data["Presets"] = presets
            file.seek(0)
            json.dump(data, file, indent=4)
            file.truncate()
            logAndPrint(f"Preset {args.preset} created successfully", "green")

    return args


def listPresets():
    presetsPath = os.path.join(cs.WHEREAMIRUNFROM, "presets.json")

    if not os.path.exists(presetsPath):
        print("No presets found")
        return

    with open(presetsPath) as file:
        data = json.load(file)
        presets = data.get("Presets", {})

        if not presets:
            print("No presets found")
            return

        for presetName, presetValues in presets.items():
            logAndPrint(f"Preset: {presetName}", "cyan")
            trueValues = [
                key
                for key, value in presetValues.items()
                if isinstance(value, bool) and value is True
            ]
            if trueValues:
                for key in trueValues:
                    print(f" - {key}")
            print()
