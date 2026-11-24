import argparse
from pathlib import Path

from src.portableBuild import buildPortable, defaultSettings


def parseArguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build script for The Anime Scripter portable version."
    )
    parser.add_argument(
        "--develop",
        action="store_true",
        help="If active, it will overwrite the platform development install directory with the newly generated build. ONLY USE IN DEVELOPMENT!",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parseArguments()
    settings = defaultSettings(Path(__file__).resolve().parent)
    buildPortable(settings, develop=args.develop)


if __name__ == "__main__":
    main()
