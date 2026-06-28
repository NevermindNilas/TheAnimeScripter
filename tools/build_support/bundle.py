import json
import os
import re
import shutil
from pathlib import Path

from tools.build_support.context import BuildContext
from tools.build_support.process import run_subprocess_result


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


def parse_macos_dylib_dependencies(binary_path: Path) -> list[str]:
    result = run_subprocess_result(["otool", "-L", str(binary_path)])
    if result.returncode != 0:
        raise RuntimeError(f"otool failed for {binary_path}: {result.stderr}")

    deps = []
    for line in result.stdout.splitlines()[1:]:
        stripped = line.strip()
        if stripped:
            deps.append(stripped.split(" ", 1)[0])
    return deps


def is_bundled_macos_dependency(dep: str) -> bool:
    return dep.endswith(".dylib") and dep.startswith(("/opt/homebrew/", "/usr/local/"))


def resolve_macos_dependency(dep: str) -> Path:
    path = Path(dep)
    if path.exists():
        return path

    resolved = shutil.which(path.name)
    if resolved:
        return Path(resolved)

    raise FileNotFoundError(f"Could not resolve macOS dependency: {dep}")


def collect_macos_dylib_closure(entry_points: list[Path]) -> dict[str, Path]:
    collected = {}
    queue = list(entry_points)
    seen_binaries = set()

    while queue:
        current = queue.pop(0)
        current_key = str(current)
        if current_key in seen_binaries:
            continue
        seen_binaries.add(current_key)

        for dep in parse_macos_dylib_dependencies(current):
            if not is_bundled_macos_dependency(dep) or dep in collected:
                continue

            source = resolve_macos_dependency(dep)
            collected[dep] = source
            queue.append(source)

    return collected


def sign_macos_binary(path: Path) -> None:
    result = run_subprocess_result(["codesign", "--force", "--sign", "-", str(path)])
    if result.returncode != 0:
        raise RuntimeError(f"codesign failed for {path}: {result.stderr}")


def patch_macos_install_names(
    binary_path: Path,
    replacements: dict[str, str],
    id_name: str | None = None,
) -> None:
    if id_name is not None:
        result = run_subprocess_result(
            ["install_name_tool", "-id", id_name, str(binary_path)]
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"install_name_tool -id failed for {binary_path}: {result.stderr}"
            )

    deps = set(parse_macos_dylib_dependencies(binary_path))
    for old_name, new_name in replacements.items():
        if old_name not in deps:
            continue

        result = run_subprocess_result(
            ["install_name_tool", "-change", old_name, new_name, str(binary_path)]
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"install_name_tool -change failed for {binary_path}: {result.stderr}"
            )

    sign_macos_binary(binary_path)


def find_macos_ffmpeg_tools() -> tuple[Path, Path]:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg and ffprobe:
        return Path(ffmpeg), Path(ffprobe)

    for bin_dir in (Path("/opt/homebrew/bin"), Path("/usr/local/bin")):
        ffmpeg_path = bin_dir / "ffmpeg"
        ffprobe_path = bin_dir / "ffprobe"
        if ffmpeg_path.exists() and ffprobe_path.exists():
            return ffmpeg_path, ffprobe_path

    raise RuntimeError(
        "macOS portable builds require native ffmpeg and ffprobe. "
        "Install them with `brew install ffmpeg`."
    )


def iter_bundle_nelux_binaries(bundle_dir: Path) -> list[Path]:
    binaries = []
    for nelux_dir in bundle_dir.glob("lib/python*/site-packages/nelux"):
        binaries.extend(
            path
            for path in nelux_dir.rglob("*")
            if path.is_file() and path.suffix in {".so", ".dylib"}
        )
    return binaries


def patch_nelux_for_bundled_macos_ffmpeg(
    target_dir: Path, install_name_to_basename: dict[str, str]
) -> None:
    nelux_binaries = iter_bundle_nelux_binaries(target_dir)
    if not nelux_binaries:
        print("Skipping nelux FFmpeg patch; nelux package not found in bundle.")
        return

    ffmpeg_lib_dir = target_dir / "ffmpeg_shared" / "lib"
    for nelux_dir in target_dir.glob("lib/python*/site-packages/nelux"):
        nelux_dylib_dir = nelux_dir / ".dylibs"
        nelux_dylib_dir.mkdir(exist_ok=True)
        for library_name in set(install_name_to_basename.values()):
            source = ffmpeg_lib_dir / library_name
            if source.exists():
                shutil.copy2(source, nelux_dylib_dir / library_name)

    cellar_pattern = re.compile(
        r"/(?:opt/homebrew|usr/local)/Cellar/ffmpeg/[^/\s]+/lib/(lib(?:av|sw)[^/\s]+\.dylib)"
    )
    opt_pattern = re.compile(
        r"/(?:opt/homebrew|usr/local)/opt/ffmpeg/lib/(lib(?:av|sw)[^/\s]+\.dylib)"
    )

    for binary_path in nelux_binaries:
        deps = parse_macos_dylib_dependencies(binary_path)
        replacements = {}
        for dep in deps:
            match = cellar_pattern.match(dep) or opt_pattern.match(dep)
            if not match:
                continue

            lib_name = match.group(1)
            if (binary_path.parent / lib_name).exists():
                replacements[dep] = f"@loader_path/{lib_name}"
            else:
                replacements[dep] = f"@loader_path/.dylibs/{lib_name}"

        id_name = None
        if binary_path.suffix == ".dylib":
            id_name = f"@rpath/{binary_path.name}"

        if replacements or id_name is not None:
            patch_macos_install_names(binary_path, replacements, id_name=id_name)


def bundle_macos_ffmpeg(context: BuildContext, target_dir: Path) -> None:
    if context.system != "Darwin":
        return

    print("Bundling macOS FFmpeg executables and dylibs...")
    ffmpeg_path, ffprobe_path = find_macos_ffmpeg_tools()
    ffmpeg_dir = target_dir / "ffmpeg_shared"
    lib_dir = ffmpeg_dir / "lib"
    lib_dir.mkdir(parents=True, exist_ok=True)

    bundled_executables = []
    for source in (ffmpeg_path, ffprobe_path):
        destination = ffmpeg_dir / source.name
        shutil.copy2(source, destination)
        os.chmod(destination, 0o755)
        bundled_executables.append(destination)

    dylibs = collect_macos_dylib_closure([ffmpeg_path, ffprobe_path])
    install_name_to_basename = {}
    for install_name, source in dylibs.items():
        destination = lib_dir / Path(install_name).name
        shutil.copy2(source, destination)
        os.chmod(destination, 0o755)
        install_name_to_basename[install_name] = destination.name

    executable_replacements = {
        old: f"@executable_path/lib/{name}"
        for old, name in install_name_to_basename.items()
    }
    library_replacements = {
        old: f"@loader_path/{name}" for old, name in install_name_to_basename.items()
    }

    for executable in bundled_executables:
        patch_macos_install_names(executable, executable_replacements)

    for library_name in install_name_to_basename.values():
        library_path = lib_dir / library_name
        patch_macos_install_names(
            library_path,
            library_replacements,
            id_name=f"@rpath/{library_name}",
        )

    patch_nelux_for_bundled_macos_ffmpeg(target_dir, install_name_to_basename)


def seed_dependency_profile(context: BuildContext, target_dir: Path) -> None:
    if context.system != "Darwin":
        return

    cache_path = target_dir / ".dependencyCache.json"
    cache_path.write_text(json.dumps({"profile": "macos-mps"}), encoding="utf-8")


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
