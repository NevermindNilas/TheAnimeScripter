import src.infra.getFFMPEG as getFFMPEG


def test_dll_directory_handle_is_kept(monkeypatch, tmp_path):
    handle = object()
    monkeypatch.setattr(getFFMPEG.cs, "SYSTEM", "Windows")
    monkeypatch.setattr(getFFMPEG.os, "add_dll_directory", lambda path: handle)

    getFFMPEG.addFfmpegToDllSearchPath(str(tmp_path / "ffmpeg.exe"))

    assert getFFMPEG._ffmpegDllDirectoryHandle is handle
