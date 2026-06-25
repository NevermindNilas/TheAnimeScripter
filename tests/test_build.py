import json
from pathlib import Path

import build


def test_macos_build_installs_mps_runtime_requirements(monkeypatch, tmp_path):
    portable = tmp_path / "portable-python"
    python_executable = portable / "bin" / "python3"
    python_executable.parent.mkdir(parents=True)
    python_executable.write_text("")

    commands = []
    monkeypatch.setattr(build, "system", "Darwin")
    monkeypatch.setattr(build, "portablePythonDir", portable)
    monkeypatch.setattr(
        build,
        "runSubprocess",
        lambda command, **kwargs: commands.append((command, kwargs)),
    )

    build.installRequirements()

    requirement_args = [
        command[command.index("-r") + 1] for command, _ in commands if "-r" in command
    ]
    assert str(build.requirementsPath) in requirement_args
    assert str(Path(build.baseDir) / "extra-requirements-macos.txt") in requirement_args

    macos_install = next(
        command
        for command, _ in commands
        if str(Path(build.baseDir) / "extra-requirements-macos.txt") in command
    )
    assert macos_install[macos_install.index("-c") + 1] == str(build.requirementsPath)


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
    patched = []

    monkeypatch.setattr(build, "system", "Darwin")
    monkeypatch.setattr(build, "findMacosFfmpegTools", lambda: (ffmpeg, ffprobe))
    monkeypatch.setattr(
        build,
        "collectMacosDylibClosure",
        lambda entries: {
            "/opt/homebrew/Cellar/ffmpeg/8.1.2/lib/libavutil.60.dylib": avutil
        },
    )
    monkeypatch.setattr(
        build,
        "patchMacosInstallNames",
        lambda binary, replacements, idName=None: patched.append(
            (binary, replacements, idName)
        ),
    )
    monkeypatch.setattr(build, "patchNeluxForBundledMacosFfmpeg", lambda *_args: None)

    build.bundleMacosFfmpeg(target)

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
    bundle = tmp_path / "bundle"
    nelux = bundle / "lib" / "python3.13" / "site-packages" / "nelux"
    dylibs = nelux / ".dylibs"
    dylibs.mkdir(parents=True)
    binary = nelux / "_nelux.so"
    binary.write_bytes(b"binary")

    ffmpegLib = bundle / "ffmpeg_shared" / "lib"
    ffmpegLib.mkdir(parents=True)
    (ffmpegLib / "libavutil.60.dylib").write_bytes(b"lib")

    old = "/opt/homebrew/Cellar/ffmpeg/8.1.2/lib/libavutil.60.dylib"
    patched = []

    monkeypatch.setattr(build, "parseMacosDylibDependencies", lambda _path: [old])
    monkeypatch.setattr(
        build,
        "patchMacosInstallNames",
        lambda binary_path, replacements, idName=None: patched.append(
            (binary_path, replacements, idName)
        ),
    )

    build.patchNeluxForBundledMacosFfmpeg(
        bundle,
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


def test_macos_build_seeds_dependency_profile(monkeypatch, tmp_path):
    monkeypatch.setattr(build, "system", "Darwin")

    build.seedDependencyProfile(tmp_path)

    assert json.loads((tmp_path / ".dependencyCache.json").read_text()) == {
        "profile": "macos-mps"
    }


def test_non_macos_build_does_not_seed_dependency_profile(monkeypatch, tmp_path):
    monkeypatch.setattr(build, "system", "Linux")

    build.seedDependencyProfile(tmp_path)

    assert not (tmp_path / ".dependencyCache.json").exists()
