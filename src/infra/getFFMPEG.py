import json
import logging
import os
import platform
import shutil
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import src.constants as cs

_ffmpegDllDirectoryHandle = None

# --------------------------------------------------------------------------- #
# The pinned FFmpeg build
#
# TAS downloads FFmpeg from NevermindNilas/TAS-FFMPEG: one build of FFmpeg
# 8.1.2, produced from pinned sources for every platform TAS supports, with the
# corresponding source published beside it so the GPL obligation is actually
# satisfiable. It replaces BtbN's `latest` autobuild and johnvansickle's rolling
# static tarball, both of which meant the FFmpeg a user ended up with depended
# on the day they first ran TAS.
#
# Why one build matters, and not just one version: these binaries carry no
# --build-suffix, so nelux's bundled libraries and the ones TAS unpacks into
# ffmpeg_shared/ have identical file names. On Windows whichever loads first
# into the process serves both. Identical builds are safe; mismatched builds are
# undefined behaviour, and nothing at the OS level prevents it. The binaries are
# built with --extra-version=tas, so `ffmpeg -version` reports exactly
# FFMPEG_AV_VERSION_INFO and no distro, gyan or BtbN build can be mistaken for
# one -- which is what _verifyInstalledBuild() asserts after every install.
#
# NOT FFmpeg 9/master, deliberately: 9 expired FF_API_NVDEC_OLD_PIX_FMTS and
# nelux's CUDA decoder still throws on the replacement NVDEC pixel formats, so
# HEVC 4:4:4 high-bit-depth NVDEC decode breaks there.
#
# The upstream source of truth for every value below is the `ffmpeg-pin.lock`
# asset on the same release; nelux consumes the same file. To move to a new
# release, update the whole block -- URLs, hashes, FFMPEG_BUILD_ID and
# FFMPEG_AV_VERSION_INFO together -- and bumping FFMPEG_BUILD_ID is what makes
# every existing install replace itself (see ffmpegNeedsInstall).
#
# macOS is deliberately NOT here: TAS does not redistribute FFmpeg on macOS and
# installs it with Homebrew instead (see installMacosFfmpegViaHomebrew). It is
# therefore the one platform not pinned to this build.
# --------------------------------------------------------------------------- #
FFMPEG_BUILD_ID = "tas-ffmpeg-8.1.2"
FFMPEG_AV_VERSION_INFO = "8.1.2-tas"

_RELEASE_BASE = "https://github.com/NevermindNilas/TAS-FFMPEG/releases/download/v8.1.2"

# key -> (archive file name, directory the archive unpacks to, SHA256)
_FFMPEG_BUILDS = {
    "windows-x86_64": (
        "tas-ffmpeg-8.1.2-win64.zip",
        "tas-ffmpeg-8.1.2-win64",
        "1d570273dc8be2b3f386467657591a22045d058a830b1eb1eaa881dafb689bfb",
    ),
    "linux-x86_64": (
        "tas-ffmpeg-8.1.2-linux64.tar.xz",
        "tas-ffmpeg-8.1.2-linux64",
        "6205ef68cd9fdf2d1bd6478766c9c2b0c0363d901caea09aa91634a92a21ed28",
    ),
    "linux-aarch64": (
        "tas-ffmpeg-8.1.2-linuxarm64.tar.xz",
        "tas-ffmpeg-8.1.2-linuxarm64",
        "6a0759046f0a0537743e6fc366eb19f5fcb91ba757a84b036e1b04bcf380a3b9",
    ),
}

# Written into ffmpeg_shared/ after a successful install; read on every startup
# to decide whether that install is still the pinned one.
_STAMP_NAME = ".tas-ffmpeg.json"

# FFmpeg is GPL and its terms have to travel with the binaries. The archives put
# every component's licence under licenses/; the prefixes cover the loose files
# other builds keep at the archive root.
_LICENSE_PREFIXES = ("license", "copying", "gpl")
# manifest.json's provenance block points at share/tas-ffmpeg/config.h and
# configure-command.txt, so keeping the manifest without share/ would preserve a
# record whose references dangle. ~940KB, almost all of it config.log.
_KEEP_FROM_ARCHIVE = ("licenses", "manifest.json", "share")


def addFfmpegToDllSearchPath(ffmpegPath: str | None = None) -> None:
    global _ffmpegDllDirectoryHandle

    if cs.SYSTEM != "Windows":
        return

    ffmpegPath = ffmpegPath or cs.FFMPEGPATH
    ffmpeg_dir = os.path.dirname(ffmpegPath)
    if not ffmpeg_dir or not os.path.exists(ffmpeg_dir):
        return

    try:
        _ffmpegDllDirectoryHandle = os.add_dll_directory(ffmpeg_dir)
        logging.info(f"Added FFmpeg directory to DLL search path: {ffmpeg_dir}")
    except Exception as e:
        logging.warning(f"Failed to add FFmpeg to DLL search path: {e}")


def getFFMPEG():
    ffmpegPath = downloadAndExtractFfmpeg(cs.FFMPEGPATH)
    cs.FFMPEGPATH = ffmpegPath
    if not cs.FFPROBEPATH or not os.path.exists(cs.FFPROBEPATH):
        ffProbeExe = "ffprobe.exe" if cs.SYSTEM == "Windows" else "ffprobe"
        cs.FFPROBEPATH = os.path.join(os.path.dirname(ffmpegPath), ffProbeExe)
    addFfmpegToDllSearchPath(cs.FFMPEGPATH)


def _normalizeArch(machine: str) -> str:
    machine = machine.lower()
    if machine in ("amd64", "x86_64", "x64"):
        return "x86_64"
    if machine in ("aarch64", "arm64"):
        return "aarch64"
    return machine


def currentBuildKey() -> str:
    """The `_FFMPEG_BUILDS` key for this machine, whether or not it exists."""
    system = "windows" if cs.SYSTEM == "Windows" else cs.SYSTEM.lower()
    return f"{system}-{_normalizeArch(platform.machine())}"


def readFfmpegStamp(ffmpegDir: str) -> dict | None:
    """The provenance record left by the last successful install, if readable."""
    try:
        with open(os.path.join(ffmpegDir, _STAMP_NAME), encoding="utf-8") as file:
            stamp = json.load(file)
    except OSError, ValueError:
        return None
    return stamp if isinstance(stamp, dict) else None


def ffmpegNeedsInstall(ffmpegPath: str, ffprobePath: str) -> bool:
    """Whether ffmpeg_shared/ has to be (re)populated before this run.

    This used to be a bare `os.path.exists` on the two executables, which meant
    ffmpeg_shared/ was filled once, on the first run TAS ever did on that
    machine, and never revisited. Anyone upgrading from an earlier TAS kept the
    FFmpeg they happened to get back then -- a gyan or BtbN build, or a
    johnvansickle 7.0.2 static -- no matter what this file was pinned to, so
    changing the pin only ever reached brand-new installs.

    An install that predates the stamp has no way to identify itself, so a
    missing stamp means "not ours, replace it". That is the rule that migrates
    every existing user, and it needs no version parsing: it costs one failed
    open() per startup and no subprocess.
    """
    if not ffmpegPath or not os.path.exists(ffmpegPath):
        return True
    if not ffprobePath or not os.path.exists(ffprobePath):
        return True

    stamp = readFfmpegStamp(os.path.dirname(ffmpegPath))
    if stamp is None:
        logging.info(
            "FFmpeg in ffmpeg_shared has no TAS build stamp; replacing it with "
            f"{FFMPEG_BUILD_ID}"
        )
        return True

    installed = stamp.get("buildId")
    if installed != FFMPEG_BUILD_ID:
        logging.info(
            f"FFmpeg build changed ({installed} -> {FFMPEG_BUILD_ID}); reinstalling"
        )
        return True

    return False


def findSystemFfmpeg() -> tuple[str, str] | None:
    """Return system ffmpeg/ffprobe when both are available."""
    searchPaths = [
        None,
        "/opt/homebrew/bin",
        "/usr/local/bin",
        "/usr/bin",
    ]

    ffmpegPath = None
    ffprobePath = None
    for searchPath in searchPaths:
        ffmpegCandidate = shutil.which("ffmpeg", path=searchPath)
        ffprobeCandidate = shutil.which("ffprobe", path=searchPath)
        if ffmpegCandidate and ffprobeCandidate:
            ffmpegPath = ffmpegCandidate
            ffprobePath = ffprobeCandidate
            break

    if ffmpegPath and ffprobePath:
        return ffmpegPath, ffprobePath
    return None


def _downloadFile(
    url: str, destination: str, label: str, expectedSha256: str | None = None
) -> None:
    """Download `url` to `destination`, verifying `expectedSha256` if given.

    A checksum mismatch deletes the partial download and raises: an unexpected
    FFmpeg is exactly what pinning exists to prevent, and a tampered archive is
    the other explanation. Never fall through to extraction on a mismatch.
    """
    import hashlib

    from src.infra.progressBarLogic import ProgressBarDownloadLogic

    digest = hashlib.sha256()

    with urlopen(url) as response:
        # Check for HTTP errors manually (like raise_for_status)
        if response.getcode() != 200:
            raise HTTPError(url, response.getcode(), None, None, None)

        totalSizeInBytes = int(response.headers.get("content-length", 0))

        with (
            ProgressBarDownloadLogic(totalSizeInBytes or 1, label) as bar,
            open(destination, "wb") as file,
        ):
            while True:
                data = response.read(1024 * 1024)
                if not data:
                    break
                digest.update(data)
                file.write(data)
                bar(len(data))

    if expectedSha256 is None:
        return

    actual = digest.hexdigest()
    if actual != expectedSha256:
        try:
            os.remove(destination)
        except OSError:
            pass
        raise RuntimeError(
            f"The FFmpeg archive downloaded from {url} does not match its pinned "
            f"checksum (expected {expectedSha256}, got {actual}). Nothing was "
            f"installed. Either the download was corrupted or that URL no longer "
            f"serves the build TAS is pinned to."
        )
    logging.info(f"Verified SHA256 of {os.path.basename(destination)}")


def _safeExtractZip(archive, destination: str) -> None:
    """Extract a zip, refusing any member that would land outside `destination`.

    Mirrors tools/build_support/python_runtime.py, which already guards its
    archives this way.
    """
    resolved = os.path.realpath(destination)
    for member in archive.infolist():
        target = os.path.realpath(os.path.join(destination, member.filename))
        if os.path.commonpath([resolved, target]) != resolved:
            raise RuntimeError(
                f"Refusing to extract '{member.filename}': it escapes {destination}"
            )
    archive.extractall(destination)


def _safeExtractTar(archive, destination: str) -> None:
    archive.extractall(destination, filter="data")


def _keepArchiveExtras(extractedRoot: str, ffmpegDir: str) -> None:
    """Rescue the licence texts and the build manifest before the unpacked tree
    is deleted.

    FFmpeg is GPL, so its terms and every linked component's have to stay beside
    the binaries; manifest.json is the machine-readable identity of the build
    (version, commit, sonames, per-file hashes) and is what makes a support
    report about "which FFmpeg do you have" answerable.
    """
    if not os.path.isdir(extractedRoot):
        return

    kept = []
    for name in os.listdir(extractedRoot):
        source = os.path.join(extractedRoot, name)
        wanted = name in _KEEP_FROM_ARCHIVE or name.lower().startswith(
            _LICENSE_PREFIXES
        )
        if not wanted:
            continue
        destination = os.path.join(ffmpegDir, name)
        try:
            if os.path.isdir(source):
                shutil.copytree(source, destination, dirs_exist_ok=True)
            else:
                shutil.copy2(source, destination)
            kept.append(name)
        except OSError as e:
            logging.warning(f"Failed to preserve {name}: {e}")

    if kept:
        logging.info(f"Preserved from the FFmpeg archive: {', '.join(sorted(kept))}")
    else:
        logging.warning(f"No licence files found in {extractedRoot}")


def _removePath(target: str) -> None:
    """Delete a file, link or directory, tolerating read-only members."""
    if os.path.isdir(target) and not os.path.islink(target):
        shutil.rmtree(target, onerror=remove_readonly)
    else:
        os.remove(target)


def _flattenInto(sourceDir: str, ffmpegDir: str, keepIf=None) -> None:
    """Move `sourceDir`'s entries up into ffmpeg_shared/, replacing what is there.

    The archives ship bin/ (ffmpeg, ffprobe, and on Windows the seven DLLs) and,
    on Linux, lib/ with the shared objects those binaries link. Both are
    flattened into one directory on purpose: the binaries' runpath is
    `$ORIGIN:$ORIGIN/../lib`, so side-by-side satisfies it without
    LD_LIBRARY_PATH, and it keeps the layout TAS has always had -- ffmpeg.exe
    next to its DLLs, which is what addFfmpegToDllSearchPath and nelux expect.
    """
    if not os.path.isdir(sourceDir):
        return

    for item in sorted(os.listdir(sourceDir)):
        if keepIf is not None and not keepIf(item):
            continue

        source = os.path.join(sourceDir, item)
        destination = os.path.join(ffmpegDir, item)
        if os.path.exists(destination) or os.path.islink(destination):
            try:
                _removePath(destination)
            except OSError as e:
                logging.warning(f"Failed to remove existing file {destination}: {e}")

        shutil.move(source, destination)


def _clearStaleFfmpeg(ffmpegDir: str, keep: str | None = None) -> None:
    """Empty ffmpeg_shared/ before laying down the pinned build.

    Unpacking over the top is not enough. An install from before this pin
    carries whatever library names that build used, and anything the new archive
    does not happen to overwrite survives beside it -- a leftover avcodec-61.dll
    next to a fresh avcodec-62.dll. Windows binds the first matching name the
    loader finds, and nelux delay-loads FFmpeg and aborts the process when the
    sonames disagree with what its wheel was built against, so a mixed directory
    is a crash rather than clutter.

    ffmpeg_shared/ is created and owned by TAS -- it is gitignored and no build
    step populates it -- so emptying it is safe. This runs before nelux is
    imported and before addFfmpegToDllSearchPath, so nothing in *this* process
    holds the DLLs; another TAS can (the Adobe Edition relaunches TAS while a
    render is running), and Windows refuses to delete a loaded DLL.

    That case is checked BEFORE anything is deleted. Deleting first and raising
    afterwards would leave the user with a half-erased install *and* a failed
    startup -- worse than the mixed directory this exists to prevent -- so the
    locked files are found with a rename probe (the same operation Windows
    refuses on an open image) while the old install is still whole.

    `keep` is the already-downloaded archive, which lives in this directory and
    is the reason nothing is deleted until the replacement is on disk and its
    checksum has been verified: a user whose download fails keeps the FFmpeg
    they had rather than being left with none.
    """
    if not os.path.isdir(ffmpegDir):
        return

    def isOurs(name: str) -> bool:
        # A concurrent instance's in-flight download is not ours to delete.
        return not (name == keep or name.startswith("ffmpeg-download-"))

    doomed = [name for name in os.listdir(ffmpegDir) if isOurs(name)]

    stuck = []
    for name in doomed:
        target = os.path.join(ffmpegDir, name)
        if os.path.isdir(target) and not os.path.islink(target):
            continue
        probe = f"{target}.replacing"
        try:
            os.replace(target, probe)
            os.replace(probe, target)
        except OSError:
            stuck.append(name)

    if stuck:
        raise RuntimeError(
            f"Could not replace the FFmpeg in {ffmpegDir}: "
            f"{', '.join(sorted(stuck))} is in use, so nothing was changed. "
            f"Another TAS instance (or After Effects, which keeps TAS running) "
            f"most likely has these files open. Close it and run TAS again."
        )

    for name in doomed:
        target = os.path.join(ffmpegDir, name)
        try:
            _removePath(target)
        except OSError as e:
            logging.warning(f"Failed to remove {target}: {e}")


def _verifyInstalledBuild(ffmpegPath: str) -> None:
    """Run the freshly unpacked ffmpeg once and assert it is the pinned build.

    Cheap where it counts: this happens on install, not on startup, so the
    steady-state path still spends no subprocess. It catches a truncated or
    misassembled unpack, and on Linux it also proves the flattened layout
    resolves -- the binary cannot start at all if $ORIGIN does not find the
    shared objects.
    """
    import subprocess

    try:
        result = subprocess.run(
            [ffmpegPath, "-version"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as e:
        raise RuntimeError(
            f"The FFmpeg just installed to {ffmpegPath} could not be run: {e}"
        ) from e

    banner = f"{result.stdout or ''}{result.stderr or ''}"
    if FFMPEG_AV_VERSION_INFO not in banner:
        firstLine = banner.strip().splitlines()[0] if banner.strip() else "no output"
        raise RuntimeError(
            f"The FFmpeg installed to {ffmpegPath} reports '{firstLine}', not the "
            f"pinned {FFMPEG_AV_VERSION_INFO} build. The install was not "
            f"completed; delete that directory and run TAS again."
        )
    logging.info(f"Verified FFmpeg build {FFMPEG_AV_VERSION_INFO}")


def _writeStamp(ffmpegDir: str, buildKey: str, url: str, sha256: str) -> None:
    stamp = {
        "buildId": FFMPEG_BUILD_ID,
        "avVersionInfo": FFMPEG_AV_VERSION_INFO,
        "platform": buildKey,
        "url": url,
        "sha256": sha256,
    }
    try:
        with open(os.path.join(ffmpegDir, _STAMP_NAME), "w", encoding="utf-8") as file:
            json.dump(stamp, file, indent=2)
    except OSError as e:
        # A missing stamp only costs one redundant reinstall next startup, so it
        # is not worth failing an otherwise good install over.
        logging.warning(f"Failed to write the FFmpeg build stamp: {e}")


def findHomebrew() -> str | None:
    return shutil.which("brew") or next(
        (
            candidate
            for candidate in ("/opt/homebrew/bin/brew", "/usr/local/bin/brew")
            if os.path.exists(candidate)
        ),
        None,
    )


def installMacosFfmpegViaHomebrew() -> tuple[str, str] | None:
    """Install FFmpeg with Homebrew. Returns (ffmpeg, ffprobe) on success.

    TAS must not ship FFmpeg itself: Homebrew's build is GPL, and redistributing
    it would carry the GPL's source-offer obligations. Installing it on the
    user's machine keeps TAS free of any FFmpeg bytes. It also gives nelux the
    shared FFmpeg dylibs it links against, which a static ffmpeg binary cannot.

    The cost is that macOS is the one platform not pinned to FFMPEG_BUILD_ID:
    `brew install ffmpeg` hands over whatever bottle exists that day.
    """
    import subprocess

    brew = findHomebrew()
    if brew is None:
        return None

    logging.info("FFmpeg not found, installing it with Homebrew")
    try:
        result = subprocess.run([brew, "install", "ffmpeg"], check=False)
    except OSError as e:
        logging.error(f"Failed to run `brew install ffmpeg`: {e}")
        return None

    if result.returncode != 0:
        logging.error(f"`brew install ffmpeg` exited with {result.returncode}")
        return None

    return findSystemFfmpeg()


def downloadAndExtractFfmpeg(ffmpegPath):
    logging.info("Downloading FFMPEG")
    ffmpegDir = os.path.dirname(ffmpegPath)
    if cs.SYSTEM == "Darwin":
        systemFfmpeg = findSystemFfmpeg() or installMacosFfmpegViaHomebrew()
        if systemFfmpeg is not None:
            cs.FFPROBEPATH = systemFfmpeg[1]
            logging.info(f"Using system FFmpeg: {systemFfmpeg[0]}")
            return systemFfmpeg[0]

        if findHomebrew() is None:
            raise RuntimeError(
                "FFmpeg and FFprobe are required on macOS, and TAS could not "
                "install them automatically because Homebrew was not found. "
                "Install Homebrew from https://brew.sh and run "
                "`brew install ffmpeg`, or place ffmpeg and ffprobe on PATH."
            )

        raise RuntimeError(
            "FFmpeg and FFprobe are required on macOS, and `brew install ffmpeg` "
            "did not produce them. Run it by hand to see why it failed, or place "
            "ffmpeg and ffprobe on PATH."
        )

    buildKey = currentBuildKey()
    build = _FFMPEG_BUILDS.get(buildKey)
    if build is None:
        raise RuntimeError(
            f"No pinned FFmpeg build exists for {buildKey}. TAS-FFMPEG publishes "
            f"{', '.join(sorted(_FFMPEG_BUILDS))}; install ffmpeg and ffprobe "
            f"yourself and put them on PATH, or open an issue asking for this "
            f"platform."
        )

    archiveName, extractedRootName, sha256 = build
    url = f"{_RELEASE_BASE}/{archiveName}"

    os.makedirs(ffmpegDir, exist_ok=True)

    # The pid keeps two TAS instances that start together from writing the same
    # partial archive over each other.
    suffix = ".zip" if cs.SYSTEM == "Windows" else ".tar.xz"
    archiveFileName = f"ffmpeg-download-{os.getpid()}{suffix}"
    ffmpegArchivePath = os.path.join(ffmpegDir, archiveFileName)

    try:
        _downloadFile(url, ffmpegArchivePath, "Downloading FFmpeg", sha256)
    except (URLError, HTTPError) as e:
        logging.error(f"Failed to download FFMPEG: {e}")
        raise

    # Only now that a verified replacement is on disk: whatever is there is
    # either a different build or a half-finished one, and either way it goes
    # before anything is unpacked over it. Doing this first would leave a user
    # whose download failed with no FFmpeg at all, where before they at least
    # kept a working one.
    _clearStaleFfmpeg(ffmpegDir, keep=archiveFileName)

    extractFunc = extractFfmpegZip if cs.SYSTEM == "Windows" else extractFfmpegTar
    extractFunc(ffmpegArchivePath, ffmpegDir, extractedRootName)

    _verifyInstalledBuild(str(ffmpegPath))
    _writeStamp(ffmpegDir, buildKey, url, sha256)
    return str(ffmpegPath)


def extractFfmpegZip(ffmpegZipPath, ffmpegDir, extractedRootName):
    import zipfile

    try:
        with zipfile.ZipFile(ffmpegZipPath, "r") as zipRef:
            _safeExtractZip(zipRef, ffmpegDir)

        extractedRoot = os.path.join(ffmpegDir, extractedRootName)
        _keepArchiveExtras(extractedRoot, ffmpegDir)
        _flattenInto(os.path.join(extractedRoot, "bin"), ffmpegDir)

        if os.path.exists(extractedRoot):
            shutil.rmtree(extractedRoot, onerror=remove_readonly)

    except zipfile.BadZipFile as e:
        logging.error(f"Failed to extract ZIP: {e}")
        raise
    finally:
        if os.path.exists(ffmpegZipPath):
            os.remove(ffmpegZipPath)


def extractFfmpegTar(ffmpegTarPath, ffmpegDir, extractedRootName):
    import stat
    import tarfile

    try:
        with tarfile.open(ffmpegTarPath, "r:xz") as tarRef:
            _safeExtractTar(tarRef, ffmpegDir)

        extractedRoot = os.path.join(ffmpegDir, extractedRootName)
        _keepArchiveExtras(extractedRoot, ffmpegDir)
        _flattenInto(os.path.join(extractedRoot, "bin"), ffmpegDir)
        # pkgconfig/ and the .a import libraries are build-time only; the
        # runtime needs the shared objects and the symlinks pointing at them.
        _flattenInto(
            os.path.join(extractedRoot, "lib"),
            ffmpegDir,
            keepIf=lambda item: ".so" in item,
        )

        for executable in ("ffmpeg", "ffprobe"):
            target = os.path.join(ffmpegDir, executable)
            if os.path.exists(target):
                os.chmod(target, os.stat(target).st_mode | stat.S_IEXEC)

        if os.path.exists(extractedRoot):
            shutil.rmtree(extractedRoot, onerror=remove_readonly)

    except tarfile.TarError as e:
        logging.error(f"Failed to extract TAR: {e}")
        raise
    finally:
        if os.path.exists(ffmpegTarPath):
            os.remove(ffmpegTarPath)


def remove_readonly(func, path, excinfo):
    import logging
    import stat
    import time

    try:
        os.chmod(path, stat.S_IWRITE)
    except Exception:
        pass

    try:
        func(path)
    except Exception:
        time.sleep(1)
        try:
            func(path)
        except Exception as e:
            logging.warning(f"Failed to remove {path}: {e}")
