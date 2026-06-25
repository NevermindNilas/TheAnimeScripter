import os
import shutil
from pathlib import Path

from tools.build_support.context import BuildContext


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
