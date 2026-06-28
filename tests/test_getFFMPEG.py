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


def test_darwin_raises_when_no_native_ffmpeg(monkeypatch, tmp_path):
    monkeypatch.setattr(getFFMPEG.cs, "SYSTEM", "Darwin")
    monkeypatch.setattr(getFFMPEG.shutil, "which", lambda name, path=None: None)

    try:
        getFFMPEG.downloadAndExtractFfmpeg(str(tmp_path / "ffmpeg"))
    except RuntimeError as exc:
        assert "brew install ffmpeg" in str(exc)
    else:
        raise AssertionError("Expected missing macOS FFmpeg to raise")
