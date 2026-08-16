import logging
import os

import src.constants as cs


def _promptDownloadRequirementsSelection() -> str:
    import sys

    from inquirer import List, prompt

    from src.infra.logAndPrint import logAndPrint

    currentPlatform = "windows" if cs.SYSTEM == "Windows" else "linux"
    if cs.SYSTEM == "Darwin":
        currentPlatform = "macos"
        choices = [
            ("Full MPS dependencies (Apple Silicon GPU)", "macos-mps"),
            ("Lite CPU dependencies (no MPS)", "macos-lite"),
        ]
    else:
        choices = [
            (
                "Full CUDA / TensorRT dependencies (GTX 16xx, RTX 20xx+, newer NVIDIA)",
                f"{currentPlatform}-cuda",
            ),
            (
                "Lite dependencies (GTX 10xx, AMD, Intel)",
                f"{currentPlatform}-lite",
            ),
        ]
    answers = prompt(
        [
            List(
                "dependency_profile",
                message=(
                    f"Select which {currentPlatform.title()} dependencies to install for your hardware"
                ),
                choices=choices,
            )
        ]
    )

    if not answers:
        logAndPrint("No dependency profile selected, exiting.", "red")
        sys.exit(1)

    return answers["dependency_profile"]


def _handleDependencies(args):
    import shutil

    from src.infra.getFFMPEG import (
        addFfmpegToDllSearchPath,
        ffmpegNeedsInstall,
        remove_readonly,
    )

    legacyFFMPEG = os.path.join(cs.WHEREAMIRUNFROM, "ffmpeg")
    if os.path.isdir(legacyFFMPEG):
        try:
            shutil.rmtree(legacyFFMPEG, onerror=remove_readonly)
            logging.info(f"Removed legacy FFmpeg folder: {legacyFFMPEG}")
        except Exception as e:
            logging.warning(f"Failed to remove legacy FFmpeg folder: {e}")

    ffmpegBaseDir = cs.WHEREAMIRUNFROM
    ffmpegSharedDir = os.path.join(ffmpegBaseDir, "ffmpeg_shared")

    cs.FFMPEGPATH = os.path.join(
        ffmpegSharedDir,
        "ffmpeg.exe" if cs.SYSTEM == "Windows" else "ffmpeg",
    )

    cs.FFPROBEPATH = os.path.join(
        ffmpegSharedDir,
        "ffprobe.exe" if cs.SYSTEM == "Windows" else "ffprobe",
    )

    probeName = "ffprobe.exe" if cs.SYSTEM == "Windows" else "ffprobe"
    if probeName not in os.environ["PATH"]:
        os.environ["PATH"] += os.pathsep + os.path.dirname(cs.FFPROBEPATH)

    # Not a bare existence check: ffmpeg_shared/ used to be filled on the first
    # run TAS ever did on a machine and never looked at again, so an upgrading
    # user kept whichever third-party FFmpeg they got back then no matter what
    # src/infra/getFFMPEG.py was pinned to. ffmpegNeedsInstall also replaces an
    # install whose build stamp is missing or names a different build.
    if ffmpegNeedsInstall(cs.FFMPEGPATH, cs.FFPROBEPATH):
        from src.infra.getFFMPEG import getFFMPEG

        haveWorkingFfmpeg = os.path.exists(cs.FFMPEGPATH) and os.path.exists(
            cs.FFPROBEPATH
        )
        try:
            getFFMPEG()
        except Exception as e:
            # An upgrade is not worth taking the run down for. Most users
            # reaching here already have a working (if unpinned) FFmpeg from an
            # older TAS, and before the build stamp existed they were never
            # asked to download anything at all -- so an offline machine, a
            # pruned release or a checksum mismatch must not turn a run that
            # used to work into a startup failure. Nothing is deleted until the
            # replacement is downloaded and verified, so the old install is
            # still intact here. Loud, because they are not on the pinned build.
            if not haveWorkingFfmpeg:
                raise
            from src.infra.logAndPrint import logAndPrint

            logging.warning(f"Failed to install the pinned FFmpeg: {e}")
            logAndPrint(
                f"Could not install the pinned FFmpeg ({e}). Continuing with "
                f"the FFmpeg already in ffmpeg_shared, which is not the build "
                f"TAS is tested against.",
                "yellow",
            )
            addFfmpegToDllSearchPath(cs.FFMPEGPATH)
    else:
        addFfmpegToDllSearchPath(cs.FFMPEGPATH)

    try:
        from src.infra.isCudaInit import detectGPUArchitecture, detectNVidiaGPU

        isNvidia = detectNVidiaGPU()
        supportsCuda = False
        if isNvidia:
            supportsCuda, _, _ = detectGPUArchitecture()
        args.supportsCuda = supportsCuda
    except ImportError:
        isNvidia = False
        supportsCuda = False
        args.supportsCuda = False

    from src.infra.dependencyHandler import getDependencyProfile

    args.dependency_profile = getDependencyProfile(cs.SYSTEM, supportsCuda)

    if args.download_requirements is None and not args.cleanup:
        from src.infra.dependencyHandler import DependencyChecker

        checker = DependencyChecker()
        checker.ensureDependencies()
