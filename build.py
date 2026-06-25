import argparse
import os
import shutil
from pathlib import Path

from tools.build_support.bundle import (
    bundle_files,
    cleanup_temp_files,
    move_extras,
    remove_portable_python,
)
from tools.build_support.context import (
    BuildContext,
    create_build_context,
    validate_requirements_files,
)
from tools.build_support.python_runtime import download_portable_python
from tools.build_support.requirements import install_requirements


def resolve_output_dir(context: BuildContext, develop: bool) -> Path:
    if not develop:
        return context.dist_path / "main"

    if context.system == "Windows":
        return Path(r"C:\Users\nilas\AppData\Roaming\TheAnimeScripter")
    if context.system == "Darwin":
        return (
            Path.home()
            / "Library"
            / "Application Support"
            / "TheAnimeScripter"
            / "TAS-Portable"
        )

    return Path.home() / ".config" / "TheAnimeScripter" / "TAS-Portable"


def prepare_output_dir(output_dir: Path, develop: bool) -> None:
    if output_dir.exists() and not develop:
        # Just so it doesn't randomly delete TAS-Portable.
        print(f"Removing existing build directory: {output_dir}")
        shutil.rmtree(output_dir)

    os.makedirs(output_dir.parent, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)


def build_portable(context: BuildContext, develop: bool = False) -> Path:
    validate_requirements_files(context)
    final_output_dir = resolve_output_dir(context, develop)
    prepare_output_dir(final_output_dir, develop)

    download_portable_python(context)
    install_requirements(context)
    bundle_files(context, final_output_dir)
    move_extras(context, final_output_dir)
    cleanup_temp_files(context, final_output_dir)
    remove_portable_python(context)

    print("Bundle process completed successfully!")
    print(f"Portable bundle is ready at {final_output_dir}")
    return final_output_dir


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
