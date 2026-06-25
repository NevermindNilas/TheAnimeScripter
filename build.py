import argparse

from tools.build_support.context import create_build_context
from tools.build_support.pipeline import build_portable


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build script for The Anime Scripter portable version."
    )
    parser.add_argument(
        "--develop",
        action="store_true",
        help=(
            "If active, it will overwrite the contents of "
            "F:\\TheAnimeScripter\\dist-portable\\main with the newly generated "
            "build. ONLY USE IN DEVELOPMENT!"
        ),
    )
    args = parser.parse_args()

    build_portable(create_build_context(), develop=args.develop)


if __name__ == "__main__":
    main()
