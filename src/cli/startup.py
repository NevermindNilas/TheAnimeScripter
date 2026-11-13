import logging
import os

import src.constants as cs


def _promptDownloadRequirementsSelection() -> str:
    import sys

    from inquirer import List, prompt

    from src.infra.dependencyHandler import (
        DependenciesMissingError,
        canPromptInteractively,
    )
    from src.infra.logAndPrint import logAndPrint

    # Without a terminal the picker cannot be answered: on Windows it blocks
    # forever on the console, on macOS/Linux it fails at once.
    if not canPromptInteractively():
        raise DependenciesMissingError("No terminal to choose a dependency profile in")

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
                f"ROCm dependencies (AMD Radeon on {currentPlatform.title()}, experimental)",
                f"{currentPlatform}-rocm",
            ),
            (
                "Lite dependencies (NCNN/Vulkan and CPU, no CUDA)"
                if currentPlatform == "linux"
                else "Lite dependencies (GTX 10xx, AMD, Intel)",
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


def _resolveRequestedProfile(requested: str, detectedProfile: str) -> str:
    """The profile a `--download_requirements [PROFILE]` run installs.

    A bare flag asks at a terminal; with no terminal it takes the profile
    detected from the hardware instead of waiting on a picker nobody can answer.
    """
    from src.infra.dependencyHandler import canPromptInteractively
    from src.infra.logAndPrint import logAndPrint

    profile = requested.strip().lower()
    if profile:
        return profile
    if canPromptInteractively():
        return _promptDownloadRequirementsSelection()
    logAndPrint(
        f"No dependency profile given and no terminal to choose one in; "
        f"installing the detected '{detectedProfile}' profile.",
        "yellow",
    )
    return detectedProfile


def _ensureDependenciesOrExit() -> None:
    """Make sure the stored profile's dependencies are installed before a run.

    Where that needs a profile choice and there is no terminal to make it in,
    print DEPENDENCIES_MISSING_MARKER and exit, so the host panel can run its
    own installer instead of watching a run that never starts.
    """
    import sys

    from src.infra.dependencyHandler import (
        DEPENDENCIES_MISSING_EXIT_CODE,
        DEPENDENCIES_MISSING_MARKER,
        DependenciesMissingError,
        DependencyChecker,
    )
    from src.infra.logAndPrint import logAndPrint

    try:
        DependencyChecker().ensureDependencies()
    except DependenciesMissingError as e:
        logAndPrint(
            f"{DEPENDENCIES_MISSING_MARKER}: {e}. Install them from the TAS "
            f"panel, or run TAS once with --download_requirements.",
            "red",
            "ERROR",
        )
        sys.exit(DEPENDENCIES_MISSING_EXIT_CODE)


def _useCertifiWithoutSystemCertificates() -> None:
    """Let HTTPS work on a minimal Linux system with no CA bundle installed.

    Model and FFmpeg downloads use urllib, which only reads the system's store;
    pip gets away with it by shipping its own.
    """
    import ssl

    if cs.SYSTEM != "Linux" or "SSL_CERT_FILE" in os.environ:
        return
    paths = ssl.get_default_verify_paths()
    if os.path.isfile(paths.openssl_cafile):
        return
    if os.path.isdir(paths.openssl_capath) and os.listdir(paths.openssl_capath):
        return
    try:
        import certifi  # ty: ignore[unresolved-import]
    except ImportError:
        return
    logging.info("No system CA certificates found; using certifi's bundle")
    os.environ["SSL_CERT_FILE"] = certifi.where()


def _torchCanRunCuda() -> bool:
    try:
        import torch
    except ImportError:
        # Nothing installed yet: an install run has no stage to fall back.
        return True
    return torch.cuda.is_available()


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

    # Where yt-dlp's FFmpeg lives. Only named here: it is installed on the first
    # URL input (getFFMPEG.ensureFFMPEG), since nothing else in TAS runs it.
    ffmpegSharedDir = os.path.join(cs.WHEREAMIRUNFROM, "ffmpeg_shared")
    cs.FFMPEGPATH = os.path.join(
        ffmpegSharedDir,
        "ffmpeg.exe" if cs.SYSTEM == "Windows" else "ffmpeg",
    )
    cs.FFPROBEPATH = os.path.join(
        ffmpegSharedDir,
        "ffprobe.exe" if cs.SYSTEM == "Windows" else "ffprobe",
    )

    try:
        from src.infra.isCudaInit import (
            detectGPUArchitecture,
            detectNVidiaGPU,
            detectRocmGPU,
            torchCudaSupport,
        )

        isRocm = detectRocmGPU()
        # The NVML / nvidia-smi probe only when torch cannot answer: before
        # the dependencies are installed, or with a CPU-only torch.
        supportsCuda = torchCudaSupport()
        if supportsCuda is None:
            supportsCuda = False
            if detectNVidiaGPU():
                supportsCuda, _, _ = detectGPUArchitecture()
        args.supportsCuda = supportsCuda or isRocm
    except ImportError:
        supportsCuda = False
        isRocm = False
        args.supportsCuda = False

    from src.infra.dependencyHandler import getDependencyProfile

    args.dependency_profile = getDependencyProfile(cs.SYSTEM, supportsCuda, isRocm)

    if args.download_requirements is None and not args.cleanup:
        _ensureDependenciesOrExit()

    _useCertifiWithoutSystemCertificates()

    # The hardware answer above picks the profile; a run also needs a torch that
    # can reach it (not a CPU-only build, a broken driver or a hidden GPU).
    if args.supportsCuda and not isRocm and not _torchCanRunCuda():
        args.supportsCuda = False
