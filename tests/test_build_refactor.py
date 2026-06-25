from pathlib import Path

from tools.build_support import paths, pipeline
from tools.build_support.context import BuildContext


def make_context(tmp_path, system="Linux"):
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


def test_resolve_output_dir_uses_dist_for_release_build(tmp_path):
    context = make_context(tmp_path)

    assert paths.resolve_output_dir(context, develop=False) == (
        tmp_path / "dist-portable" / "main"
    )


def test_resolve_output_dir_uses_linux_develop_path(monkeypatch, tmp_path):
    context = make_context(tmp_path, system="Linux")
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")

    assert paths.resolve_output_dir(context, develop=True) == (
        tmp_path / "home" / ".config" / "TheAnimeScripter" / "TAS-Portable"
    )


def test_resolve_output_dir_uses_windows_user_roaming_path(monkeypatch, tmp_path):
    context = make_context(tmp_path, system="Windows")
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "Users" / "alice")

    assert paths.resolve_output_dir(context, develop=True) == (
        tmp_path / "Users" / "alice" / "AppData" / "Roaming" / "TheAnimeScripter"
    )


def test_build_portable_runs_steps_in_order(monkeypatch, tmp_path):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("")
    context = make_context(tmp_path)
    calls = []

    monkeypatch.setattr(
        pipeline, "download_portable_python", lambda ctx: calls.append("python")
    )
    monkeypatch.setattr(
        pipeline, "install_requirements", lambda ctx: calls.append("requirements")
    )
    monkeypatch.setattr(
        pipeline,
        "bundle_files",
        lambda ctx, output_dir: calls.append(("bundle", output_dir)),
    )
    monkeypatch.setattr(
        pipeline,
        "move_extras",
        lambda ctx, output_dir: calls.append(("extras", output_dir)),
    )
    monkeypatch.setattr(
        pipeline,
        "cleanup_temp_files",
        lambda ctx, output_dir: calls.append(("cleanup", output_dir)),
    )
    monkeypatch.setattr(
        pipeline, "remove_portable_python", lambda ctx: calls.append("remove")
    )

    output_dir = pipeline.build_portable(context)

    assert output_dir == tmp_path / "dist-portable" / "main"
    assert calls == [
        "python",
        "requirements",
        ("bundle", output_dir),
        ("extras", output_dir),
        ("cleanup", output_dir),
        "remove",
    ]
