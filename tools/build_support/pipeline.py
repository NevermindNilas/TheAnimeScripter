from pathlib import Path

from tools.build_support.bundle import (
    bundle_files,
    bundle_macos_ffmpeg,
    cleanup_temp_files,
    move_extras,
    remove_portable_python,
    seed_dependency_profile,
)
from tools.build_support.context import BuildContext, validate_requirements_files
from tools.build_support.paths import prepare_output_dir, resolve_output_dir
from tools.build_support.python_runtime import download_portable_python
from tools.build_support.requirements import install_requirements


def build_portable(context: BuildContext, develop: bool = False) -> Path:
    validate_requirements_files(context)
    final_output_dir = resolve_output_dir(context, develop)
    prepare_output_dir(final_output_dir, develop)

    download_portable_python(context)
    install_requirements(context)
    bundle_files(context, final_output_dir)
    bundle_macos_ffmpeg(context, final_output_dir)
    seed_dependency_profile(context, final_output_dir)
    move_extras(context, final_output_dir)
    cleanup_temp_files(context, final_output_dir)
    remove_portable_python(context)

    print("Bundle process completed successfully!")
    print(f"Portable bundle is ready at {final_output_dir}")
    return final_output_dir
