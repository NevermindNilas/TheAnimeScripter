import os
import shutil
from pathlib import Path

from tools.build_support.context import BuildContext


def bundle_files(context: BuildContext, target_dir: Path) -> None:
    print("Creating portable bundle...")

    bundle_dir = target_dir

    print("Copying Python installation...")
    for item in context.portable_python_dir.iterdir():
        if item.is_dir():
            shutil.copytree(item, bundle_dir / item.name, dirs_exist_ok=True)
        else:
            shutil.copy2(item, bundle_dir / item.name)

    print("Copying source code...")
    src_dir = context.base_dir / "src"
    src_dest = bundle_dir / "src"

    if src_dest.exists():
        print(f"Removing existing src directory: {src_dest}")
        shutil.rmtree(src_dest)

    shutil.copytree(src_dir, src_dest)

    shutil.copy2(context.base_dir / "main.py", bundle_dir / "main.py")

    print("Copying requirements files...")
    for requirements_file in context.requirements_files:
        shutil.copy2(requirements_file, bundle_dir / requirements_file.name)

    if context.system in ("Linux", "Darwin"):
        launcher_script = bundle_dir / "run.sh"
        platform_label = "Linux" if context.system == "Linux" else "macOS"
        with open(launcher_script, "w") as f:
            f.write(f"""#!/bin/bash
# The Anime Scripter - {platform_label} Launcher Script

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"

# Set up the Python path
PYTHON_EXE="$SCRIPT_DIR/bin/python3"

# Check if Python executable exists
if [ ! -f "$PYTHON_EXE" ]; then
    echo "Error: Python executable not found at $PYTHON_EXE"
    echo "Please run the build script first: python build.py"
    exit 1
fi

# Run the main application
exec "$PYTHON_EXE" "$SCRIPT_DIR/main.py" "$@"
""")
        os.chmod(launcher_script, 0o755)
        print(f"Created {platform_label} launcher script: run.sh")

    print(f"Portable bundle created at {bundle_dir}")


def move_extras(context: BuildContext, target_dir: Path) -> None:
    files_to_copy = [
        "LICENSE",
        "README.md",
        "README.txt",
        "PARAMETERS.MD",
        "CHANGELOG.MD",
    ]

    for file_name in files_to_copy:
        source_path = context.base_dir / file_name
        if not source_path.exists():
            print(f"Skipping missing extra file: {source_path}")
            continue

        shutil.copy(source_path, target_dir)


def cleanup_temp_files(context: BuildContext, target_dir: Path) -> None:
    print("Cleaning up temporary files...")

    if context.system == "Windows":
        temp_files = [
            "python.zip",
            "get-pip.py",
            "license.txt",
            "wheel",
        ]
    else:
        temp_files = [
            "python.tar.gz",
            "get-pip.py",
            "license.txt",
            "wheel",
        ]

    for temp_file in temp_files:
        temp_file_path = target_dir / temp_file
        if temp_file_path.exists():
            if temp_file_path.is_dir():
                shutil.rmtree(temp_file_path)
                print(f"Removed directory {temp_file_path}")
            else:
                os.remove(temp_file_path)
                print(f"Removed {temp_file_path}")
        else:
            print(f"{temp_file_path} does not exist, skipping removal.")


def remove_portable_python(context: BuildContext) -> None:
    if context.portable_python_dir.exists():
        shutil.rmtree(context.portable_python_dir)
        print(f"Removed {context.portable_python_dir}")
    else:
        print(f"{context.portable_python_dir} does not exist, skipping removal.")
