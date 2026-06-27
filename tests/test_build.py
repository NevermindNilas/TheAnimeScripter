import json

from tools.build_support import bundle as build_bundle
from tools.build_support import requirements
from tools.build_support.context import BuildContext


def make_context(tmp_path, system="Darwin"):
    return BuildContext(
        base_dir=tmp_path,
        dist_path=tmp_path / "dist-portable",
        requirements_path=tmp_path / "requirements.txt",
        requirements_files=[tmp_path / "requirements.txt"],
        portable_python_dir=tmp_path / "portable-python",
        python_version="3.14.5",
        standalone_release="20260510",
        system=system,
    )


def test_macos_build_installs_mps_runtime_requirements(monkeypatch, tmp_path):
    portable = tmp_path / "portable-python"
    python_executable = portable / "bin" / "python3"
    python_executable.parent.mkdir(parents=True)
    python_executable.write_text("")
    context = make_context(tmp_path)
    context.requirements_path.write_text("")
    extra_requirements = tmp_path / "extra-requirements-macos.txt"
    extra_requirements.write_text("")

    commands = []
    monkeypatch.setattr(
        requirements,
        "run_subprocess",
        lambda command, **kwargs: commands.append((command, kwargs)),
    )

    requirements.install_requirements(context)

    requirement_args = [
        command[command.index("-r") + 1] for command, _ in commands if "-r" in command
    ]
    assert str(context.requirements_path) in requirement_args
    assert str(extra_requirements) in requirement_args

    macos_install = next(
        command for command, _ in commands if str(extra_requirements) in command
    )
    assert macos_install[macos_install.index("-c") + 1] == str(
        context.requirements_path
    )


def test_macos_ffmpeg_bundle_copies_tools_libs_and_patches(monkeypatch, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    ffmpeg = source / "ffmpeg"
    ffprobe = source / "ffprobe"
    avutil = source / "libavutil.60.dylib"
    for path in (ffmpeg, ffprobe, avutil):
        path.write_bytes(b"binary")

    target = tmp_path / "bundle"
    target.mkdir()
    context = make_context(tmp_path)
    patched = []

    monkeypatch.setattr(
        build_bundle, "find_macos_ffmpeg_tools", lambda: (ffmpeg, ffprobe)
    )
    monkeypatch.setattr(
        build_bundle,
        "collect_macos_dylib_closure",
        lambda entries: {
            "/opt/homebrew/Cellar/ffmpeg/8.1.2/lib/libavutil.60.dylib": avutil
        },
    )
    monkeypatch.setattr(
        build_bundle,
        "patch_macos_install_names",
        lambda binary, replacements, id_name=None: patched.append(
            (binary, replacements, id_name)
        ),
    )
    monkeypatch.setattr(
        build_bundle, "patch_nelux_for_bundled_macos_ffmpeg", lambda *_args: None
    )

    build_bundle.bundle_macos_ffmpeg(context, target)

    assert (target / "ffmpeg_shared" / "ffmpeg").exists()
    assert (target / "ffmpeg_shared" / "ffprobe").exists()
    assert (target / "ffmpeg_shared" / "lib" / "libavutil.60.dylib").exists()
    assert any(
        binary == target / "ffmpeg_shared" / "ffmpeg"
        and replacements
        == {
            "/opt/homebrew/Cellar/ffmpeg/8.1.2/lib/libavutil.60.dylib": "@executable_path/lib/libavutil.60.dylib"
        }
        for binary, replacements, _ in patched
    )


def test_patch_nelux_uses_loader_path_for_bundled_ffmpeg(monkeypatch, tmp_path):
    bundle_dir = tmp_path / "bundle"
    nelux = bundle_dir / "lib" / "python3.13" / "site-packages" / "nelux"
    dylibs = nelux / ".dylibs"
    dylibs.mkdir(parents=True)
    binary = nelux / "_nelux.so"
    binary.write_bytes(b"binary")

    ffmpeg_lib = bundle_dir / "ffmpeg_shared" / "lib"
    ffmpeg_lib.mkdir(parents=True)
    (ffmpeg_lib / "libavutil.60.dylib").write_bytes(b"lib")

    old = "/opt/homebrew/Cellar/ffmpeg/8.1.2/lib/libavutil.60.dylib"
    patched = []

    monkeypatch.setattr(
        build_bundle, "parse_macos_dylib_dependencies", lambda _path: [old]
    )
    monkeypatch.setattr(
        build_bundle,
        "patch_macos_install_names",
        lambda binary_path, replacements, id_name=None: patched.append(
            (binary_path, replacements, id_name)
        ),
    )

    build_bundle.patch_nelux_for_bundled_macos_ffmpeg(
        bundle_dir,
        {old: "libavutil.60.dylib"},
    )

    assert (dylibs / "libavutil.60.dylib").exists()
    assert patched == [
        (
            binary,
            {old: "@loader_path/.dylibs/libavutil.60.dylib"},
            None,
        )
    ]


def test_macos_build_seeds_dependency_profile(tmp_path):
    context = make_context(tmp_path)

    build_bundle.seed_dependency_profile(context, tmp_path)

    assert json.loads((tmp_path / ".dependencyCache.json").read_text()) == {
        "profile": "macos-mps"
    }


def test_non_macos_build_does_not_seed_dependency_profile(tmp_path):
    context = make_context(tmp_path, system="Linux")

    build_bundle.seed_dependency_profile(context, tmp_path)

    assert not (tmp_path / ".dependencyCache.json").exists()
