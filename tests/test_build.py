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


def test_portable_bundle_never_redistributes_ffmpeg(tmp_path):
    # FFmpeg is GPL. Copying it into the bundle would make TAS a redistributor
    # and pull in the source-offer obligations, so macOS installs it via
    # Homebrew on the user's machine instead (src/infra/getFFMPEG.py).
    for helper in (
        "bundle_macos_ffmpeg",
        "find_macos_ffmpeg_tools",
        "patch_nelux_for_bundled_macos_ffmpeg",
    ):
        assert not hasattr(build_bundle, helper), (
            f"{helper} would redistribute FFmpeg binaries in the portable bundle"
        )


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
