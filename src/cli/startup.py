import logging
import os

import src.constants as cs


def _promptDownloadRequirementsSelection() -> str:
    from inquirer import List, prompt
    from src.infra.logAndPrint import logAndPrint
    import sys

    if cs.SYSTEM == "Darwin":
        return "macos-mps"

    currentPlatform = "windows" if cs.SYSTEM == "Windows" else "linux"
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
        sys.exit()

    return answers["dependency_profile"]


def _handleDependencies(args):
    import shutil
    from src.infra.getFFMPEG import remove_readonly

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

    if cs.SYSTEM == "Windows":
        if "ffprobe.exe" not in os.environ["PATH"]:
            os.environ["PATH"] += os.pathsep + os.path.dirname(cs.FFPROBEPATH)
    else:
        if "ffprobe" not in os.environ["PATH"]:
            os.environ["PATH"] += os.pathsep + os.path.dirname(cs.FFPROBEPATH)

    if not os.path.exists(cs.FFMPEGPATH) or not os.path.exists(cs.FFPROBEPATH):
        from src.infra.getFFMPEG import getFFMPEG

        getFFMPEG()

    if cs.SYSTEM == "Windows":
        ffmpeg_dir = os.path.dirname(cs.FFMPEGPATH)
        if os.path.exists(ffmpeg_dir):
            try:
                os.add_dll_directory(ffmpeg_dir)
                logging.info(f"Added FFmpeg directory to DLL search path: {ffmpeg_dir}")
            except Exception as e:
                logging.warning(f"Failed to add FFmpeg to DLL search path: {e}")

    try:
        from src.infra.isCudaInit import detectNVidiaGPU, detectGPUArchitecture

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
