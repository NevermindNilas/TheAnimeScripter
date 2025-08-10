import json
import os
from src.utils.logAndPrint import logAndPrint
from src.constants import MAINPATH


def createPreset(args):
    ignoreList = ["input", "output", "inpoint", "outpoint"]
    presetsPath = os.path.join(MAINPATH, "presets.json")

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
                if key not in ignoreList:
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
    presetsPath = os.path.join(MAINPATH, "presets.json")

    if not os.path.exists(presetsPath):
        print("No presets found")
        return

    with open(presetsPath, "r") as file:
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
