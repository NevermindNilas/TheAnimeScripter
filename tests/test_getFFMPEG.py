import json
import os
import zipfile

import pytest

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


# --------------------------------------------------------------------------- #
# Staying on the pinned build
#
# ffmpeg_shared/ was populated once, on a machine's first ever TAS run, and
# never revisited: the startup gate was `os.path.exists` on the two
# executables. Anyone upgrading kept whatever FFmpeg they got the first time,
# so re-pinning this file only ever reached new installs. These pin the rule
# that migrates existing users -- an install that cannot identify itself is
# replaced.
# --------------------------------------------------------------------------- #


def _installed(tmp_path, stamp=None, raw=None):
    """An ffmpeg_shared/ holding both executables, optionally stamped."""
    shared = tmp_path / "ffmpeg_shared"
    shared.mkdir(exist_ok=True)
    ffmpeg = shared / "ffmpeg.exe"
    ffprobe = shared / "ffprobe.exe"
    ffmpeg.write_text("")
    ffprobe.write_text("")

    if raw is not None:
        (shared / getFFMPEG._STAMP_NAME).write_text(raw)
    elif stamp is not None:
        (shared / getFFMPEG._STAMP_NAME).write_text(json.dumps(stamp))

    return str(ffmpeg), str(ffprobe)


def test_missing_binaries_need_an_install(tmp_path):
    shared = tmp_path / "ffmpeg_shared"
    shared.mkdir()

    assert getFFMPEG.ffmpegNeedsInstall(
        str(shared / "ffmpeg.exe"), str(shared / "ffprobe.exe")
    )


def test_ffprobe_alone_is_not_enough(tmp_path):
    ffmpeg, ffprobe = _installed(tmp_path, stamp={"buildId": getFFMPEG.FFMPEG_BUILD_ID})
    os.remove(ffprobe)

    assert getFFMPEG.ffmpegNeedsInstall(ffmpeg, ffprobe)


def test_unstamped_install_is_replaced(tmp_path):
    """The upgrade case: a gyan/BtbN/johnvansickle FFmpeg from an older TAS."""
    ffmpeg, ffprobe = _installed(tmp_path)

    assert getFFMPEG.ffmpegNeedsInstall(ffmpeg, ffprobe)


def test_stamp_naming_another_build_is_replaced(tmp_path):
    ffmpeg, ffprobe = _installed(tmp_path, stamp={"buildId": "tas-ffmpeg-0.0.0"})

    assert getFFMPEG.ffmpegNeedsInstall(ffmpeg, ffprobe)


def test_unreadable_stamp_is_replaced(tmp_path):
    ffmpeg, ffprobe = _installed(tmp_path, raw="{ this is not json")

    assert getFFMPEG.ffmpegNeedsInstall(ffmpeg, ffprobe)


def test_matching_stamp_is_left_alone(tmp_path):
    ffmpeg, ffprobe = _installed(tmp_path, stamp={"buildId": getFFMPEG.FFMPEG_BUILD_ID})

    assert not getFFMPEG.ffmpegNeedsInstall(ffmpeg, ffprobe)


@pytest.mark.parametrize(
    "system,machine,expected",
    [
        ("Windows", "AMD64", "windows-x86_64"),
        ("Linux", "x86_64", "linux-x86_64"),
        ("Linux", "aarch64", "linux-aarch64"),
        ("Linux", "arm64", "linux-aarch64"),
    ],
)
def test_every_shipped_platform_has_a_pinned_build(
    monkeypatch, system, machine, expected
):
    monkeypatch.setattr(getFFMPEG.cs, "SYSTEM", system)
    monkeypatch.setattr(getFFMPEG.platform, "machine", lambda: machine)

    assert getFFMPEG.currentBuildKey() == expected
    assert expected in getFFMPEG._FFMPEG_BUILDS


def _pinnedArchive(path, root):
    """A stand-in for the release zip, with the layout tas-ffmpeg publishes."""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{root}/bin/ffmpeg.exe", "ffmpeg")
        archive.writestr(f"{root}/bin/ffprobe.exe", "ffprobe")
        archive.writestr(f"{root}/bin/avcodec-62.dll", "avcodec")
        archive.writestr(f"{root}/licenses/ffmpeg/COPYING.GPLv2", "gpl")
        archive.writestr(f"{root}/manifest.json", '{"av_version_info": "8.1.2-tas"}')
        archive.writestr(f"{root}/include/libavutil/avutil.h", "header")


def _fakeInstall(monkeypatch, tmp_path):
    """Run downloadAndExtractFfmpeg against a local archive, no network."""
    monkeypatch.setattr(getFFMPEG.cs, "SYSTEM", "Windows")
    monkeypatch.setattr(getFFMPEG.platform, "machine", lambda: "AMD64")
    # The real one runs the unpacked binary; these are text files.
    monkeypatch.setattr(getFFMPEG, "_verifyInstalledBuild", lambda path: None)

    _, root, _ = getFFMPEG._FFMPEG_BUILDS["windows-x86_64"]

    def fakeDownload(url, destination, label, expectedSha256=None):
        _pinnedArchive(destination, root)

    monkeypatch.setattr(getFFMPEG, "_downloadFile", fakeDownload)

    shared = tmp_path / "ffmpeg_shared"
    shared.mkdir(exist_ok=True)
    return shared


def test_install_flattens_binaries_and_stamps_the_directory(monkeypatch, tmp_path):
    shared = _fakeInstall(monkeypatch, tmp_path)
    _, root, _ = getFFMPEG._FFMPEG_BUILDS["windows-x86_64"]

    result = getFFMPEG.downloadAndExtractFfmpeg(str(shared / "ffmpeg.exe"))

    assert result == str(shared / "ffmpeg.exe")
    # bin/* lands beside ffmpeg.exe, which is what addFfmpegToDllSearchPath and
    # nelux both expect.
    assert (shared / "ffmpeg.exe").exists()
    assert (shared / "ffprobe.exe").exists()
    assert (shared / "avcodec-62.dll").exists()
    # GPL text and the build manifest survive the unpacked tree being deleted.
    assert (shared / "licenses" / "ffmpeg" / "COPYING.GPLv2").exists()
    assert (shared / "manifest.json").exists()
    assert not (shared / root).exists()

    stamp = getFFMPEG.readFfmpegStamp(str(shared))
    assert stamp["buildId"] == getFFMPEG.FFMPEG_BUILD_ID
    assert stamp["platform"] == "windows-x86_64"
    assert not getFFMPEG.ffmpegNeedsInstall(
        str(shared / "ffmpeg.exe"), str(shared / "ffprobe.exe")
    )


def test_install_removes_the_previous_build(monkeypatch, tmp_path):
    """A leftover soname from the old build next to the new one is a crash:
    Windows binds the first matching name, and nelux aborts on a mismatch."""
    shared = _fakeInstall(monkeypatch, tmp_path)
    (shared / "avcodec-61.dll").write_text("stale")
    (shared / "ffplay.exe").write_text("stale")

    getFFMPEG.downloadAndExtractFfmpeg(str(shared / "ffmpeg.exe"))

    assert not (shared / "avcodec-61.dll").exists()
    assert not (shared / "ffplay.exe").exists()
    assert (shared / "avcodec-62.dll").exists()


def test_failed_download_leaves_the_previous_install_alone(monkeypatch, tmp_path):
    """Wiping before the replacement is on disk would strand an offline user
    with no FFmpeg, where previously they simply kept the one they had."""
    from urllib.error import URLError

    shared = _fakeInstall(monkeypatch, tmp_path)
    (shared / "ffmpeg.exe").write_text("the FFmpeg they already had")
    (shared / "ffprobe.exe").write_text("the FFprobe they already had")

    def offline(url, destination, label, expectedSha256=None):
        raise URLError("no route to host")

    monkeypatch.setattr(getFFMPEG, "_downloadFile", offline)

    with pytest.raises(URLError):
        getFFMPEG.downloadAndExtractFfmpeg(str(shared / "ffmpeg.exe"))

    assert (shared / "ffmpeg.exe").read_text() == "the FFmpeg they already had"
    assert (shared / "ffprobe.exe").exists()


def test_install_refuses_when_the_old_build_cannot_be_removed(monkeypatch, tmp_path):
    """Another TAS (or After Effects, which keeps one running) holding a DLL
    open must not leave half a build behind.

    The locked file is found with a rename probe BEFORE anything is deleted, so
    the refusal leaves the previous install whole: deleting first and raising
    afterwards would strand the user with neither FFmpeg nor a working start.
    """
    shared = _fakeInstall(monkeypatch, tmp_path)
    (shared / "avcodec-61.dll").write_text("locked")
    (shared / "ffmpeg.exe").write_text("the FFmpeg they already had")
    (shared / "ffprobe.exe").write_text("the FFprobe they already had")

    realReplace = os.replace

    def refuse(src, dst):
        if os.path.basename(str(src)) == "avcodec-61.dll":
            raise PermissionError(13, "Permission denied")
        return realReplace(src, dst)

    monkeypatch.setattr(getFFMPEG.os, "replace", refuse)

    with pytest.raises(RuntimeError, match="avcodec-61.dll"):
        getFFMPEG.downloadAndExtractFfmpeg(str(shared / "ffmpeg.exe"))

    # Nothing was destroyed on the way to the refusal.
    assert (shared / "ffmpeg.exe").read_text() == "the FFmpeg they already had"
    assert (shared / "ffprobe.exe").exists()
    assert (shared / "avcodec-61.dll").exists()


def test_install_leaves_a_concurrent_instances_download_alone(monkeypatch, tmp_path):
    """The pid suffix only helps if the wipe also spares the other pid's file."""
    shared = _fakeInstall(monkeypatch, tmp_path)
    sibling = shared / "ffmpeg-download-999999.zip"
    sibling.write_text("another instance is mid-download")

    getFFMPEG.downloadAndExtractFfmpeg(str(shared / "ffmpeg.exe"))

    assert sibling.exists()


def test_linux_install_flattens_the_shared_objects_beside_the_binaries(
    monkeypatch, tmp_path
):
    """The binaries' runpath is `$ORIGIN:$ORIGIN/../lib`, so lib/*.so* has to
    land next to ffmpeg for it to start at all without LD_LIBRARY_PATH.
    pkgconfig/ is build-time only and must not be dragged along."""
    import tarfile

    monkeypatch.setattr(getFFMPEG.cs, "SYSTEM", "Linux")
    monkeypatch.setattr(getFFMPEG.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(getFFMPEG, "_verifyInstalledBuild", lambda path: None)

    _, root, _ = getFFMPEG._FFMPEG_BUILDS["linux-x86_64"]
    staging = tmp_path / "staging" / root
    for relative in (
        "bin/ffmpeg",
        "bin/ffprobe",
        "lib/libavcodec.so.62",
        "lib/libavcodec.so.62.28.102",
        "lib/pkgconfig/libavcodec.pc",
        "licenses/ffmpeg/COPYING.GPLv2",
        "manifest.json",
    ):
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(relative)

    def fakeDownload(url, destination, label, expectedSha256=None):
        with tarfile.open(destination, "w:xz") as archive:
            archive.add(str(staging), arcname=root)

    monkeypatch.setattr(getFFMPEG, "_downloadFile", fakeDownload)

    shared = tmp_path / "ffmpeg_shared"
    shared.mkdir()
    getFFMPEG.downloadAndExtractFfmpeg(str(shared / "ffmpeg"))

    assert (shared / "ffmpeg").exists()
    assert (shared / "ffprobe").exists()
    assert (shared / "libavcodec.so.62").exists()
    assert (shared / "libavcodec.so.62.28.102").exists()
    assert not (shared / "pkgconfig").exists()
    assert (shared / "licenses" / "ffmpeg" / "COPYING.GPLv2").exists()
    assert not (shared / root).exists()
    assert getFFMPEG.readFfmpegStamp(str(shared))["platform"] == "linux-x86_64"


class _Response:
    """The slice of urlopen's result _downloadFile actually uses."""

    headers = {"content-length": "5"}

    def __init__(self, payload=b"wrong"):
        self._chunks = [payload, b""]

    def getcode(self):
        return 200

    def read(self, _size):
        return self._chunks.pop(0)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def test_checksum_mismatch_installs_nothing(monkeypatch, tmp_path):
    """An archive that is not the pinned build must never reach extraction --
    that is the whole point of pinning it."""
    monkeypatch.setattr(getFFMPEG, "urlopen", lambda url: _Response())
    destination = tmp_path / "ffmpeg.zip"

    with pytest.raises(RuntimeError, match="does not match its pinned checksum"):
        getFFMPEG._downloadFile(
            "https://example.invalid/ffmpeg.zip",
            str(destination),
            "Downloading FFmpeg",
            "0" * 64,
        )

    assert not destination.exists()
