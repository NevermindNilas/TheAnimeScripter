import src.infra.getFFMPEG as getFFMPEG


def test_dll_directory_handle_is_kept(monkeypatch, tmp_path):
    handle = object()
    monkeypatch.setattr(getFFMPEG.cs, "SYSTEM", "Windows")
    monkeypatch.setattr(
        getFFMPEG.os, "add_dll_directory", lambda path: handle, raising=False
    )

    getFFMPEG.addFfmpegToDllSearchPath(str(tmp_path / "ffmpeg.exe"))

    assert getFFMPEG._ffmpegDllDirectoryHandle is handle


def test_darwin_prefers_system_ffmpeg(monkeypatch, tmp_path):
    ffmpeg = tmp_path / "ffmpeg"
    ffprobe = tmp_path / "ffprobe"
    ffmpeg.write_text("")
    ffprobe.write_text("")

    monkeypatch.setattr(getFFMPEG.cs, "SYSTEM", "Darwin")
    monkeypatch.setattr(
        getFFMPEG.shutil,
        "which",
        lambda name, path=None: str({"ffmpeg": ffmpeg, "ffprobe": ffprobe}.get(name)),
    )

    assert getFFMPEG.downloadAndExtractFfmpeg(str(tmp_path / "unused")) == str(ffmpeg)
    assert getFFMPEG.cs.FFPROBEPATH == str(ffprobe)


def test_darwin_checks_homebrew_path_when_path_lookup_misses(monkeypatch, tmp_path):
    ffmpeg = tmp_path / "ffmpeg"
    ffprobe = tmp_path / "ffprobe"
    ffmpeg.write_text("")
    ffprobe.write_text("")

    def fake_which(name, path=None):
        if path != "/opt/homebrew/bin":
            return None
        return str({"ffmpeg": ffmpeg, "ffprobe": ffprobe}.get(name))

    monkeypatch.setattr(getFFMPEG.shutil, "which", fake_which)

    assert getFFMPEG.findSystemFfmpeg() == (str(ffmpeg), str(ffprobe))


class _CompletedProcess:
    def __init__(self, returncode):
        self.returncode = returncode


def _installsFfmpeg(monkeypatch, tmp_path, returncode=0):
    """Make brew resolvable and have `brew install ffmpeg` publish the tools."""
    ffmpeg = tmp_path / "ffmpeg"
    ffprobe = tmp_path / "ffprobe"
    installed = []

    def fake_which(name, path=None):
        if name == "brew":
            return "/opt/homebrew/bin/brew"
        if installed and name in ("ffmpeg", "ffprobe"):
            return str({"ffmpeg": ffmpeg, "ffprobe": ffprobe}[name])
        return None

    def fake_run(command, check=False):
        installed.append(command)
        if returncode == 0:
            ffmpeg.write_text("")
            ffprobe.write_text("")
        return _CompletedProcess(returncode)

    monkeypatch.setattr(getFFMPEG.cs, "SYSTEM", "Darwin")
    monkeypatch.setattr(getFFMPEG.shutil, "which", fake_which)

    import subprocess

    monkeypatch.setattr(subprocess, "run", fake_run)
    return ffmpeg, ffprobe, installed


def test_darwin_installs_ffmpeg_with_homebrew_when_missing(monkeypatch, tmp_path):
    ffmpeg, ffprobe, installed = _installsFfmpeg(monkeypatch, tmp_path)

    assert getFFMPEG.downloadAndExtractFfmpeg(str(tmp_path / "unused")) == str(ffmpeg)
    assert getFFMPEG.cs.FFPROBEPATH == str(ffprobe)
    assert installed == [["/opt/homebrew/bin/brew", "install", "ffmpeg"]]


def test_darwin_does_not_reinstall_when_ffmpeg_already_present(monkeypatch, tmp_path):
    ffmpeg = tmp_path / "ffmpeg"
    ffprobe = tmp_path / "ffprobe"
    ffmpeg.write_text("")
    ffprobe.write_text("")

    monkeypatch.setattr(getFFMPEG.cs, "SYSTEM", "Darwin")
    monkeypatch.setattr(
        getFFMPEG.shutil,
        "which",
        lambda name, path=None: str({"ffmpeg": ffmpeg, "ffprobe": ffprobe}.get(name)),
    )

    def fail(*_args, **_kwargs):
        raise AssertionError("Homebrew must not run when FFmpeg is already installed")

    monkeypatch.setattr(getFFMPEG, "installMacosFfmpegViaHomebrew", fail)

    assert getFFMPEG.downloadAndExtractFfmpeg(str(tmp_path / "unused")) == str(ffmpeg)


def test_darwin_raises_when_homebrew_install_fails(monkeypatch, tmp_path):
    _installsFfmpeg(monkeypatch, tmp_path, returncode=1)

    try:
        getFFMPEG.downloadAndExtractFfmpeg(str(tmp_path / "ffmpeg"))
    except RuntimeError as exc:
        # Homebrew is installed here; blaming a missing Homebrew would send the
        # user off diagnosing the wrong thing.
        assert "did not produce them" in str(exc)
        assert "Homebrew was not found" not in str(exc)
    else:
        raise AssertionError("Expected a failed Homebrew install to raise")


def test_darwin_raises_when_homebrew_is_absent(monkeypatch, tmp_path):
    monkeypatch.setattr(getFFMPEG.cs, "SYSTEM", "Darwin")
    monkeypatch.setattr(getFFMPEG.shutil, "which", lambda name, path=None: None)
    monkeypatch.setattr(getFFMPEG.os.path, "exists", lambda path: False)

    try:
        getFFMPEG.downloadAndExtractFfmpeg(str(tmp_path / "ffmpeg"))
    except RuntimeError as exc:
        assert "Homebrew was not found" in str(exc)
        assert "https://brew.sh" in str(exc)
    else:
        raise AssertionError("Expected a missing Homebrew to raise")
