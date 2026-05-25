import os
import logging
import re
import glob

EXTENSIONS = [".mp4", ".mkv", ".webm", ".avi", ".mov", ".gif"]
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tiff", ".tif", ".exr", ".dpx"]
OUTPUT_FILE_EXTENSIONS = tuple(EXTENSIONS + IMAGE_EXTENSIONS)

# Characters that are illegal in filenames on Windows (and a good idea to avoid
# everywhere). Argument values get folded into output names, so scrub them.
_UNSAFE_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

_SEQUENCE_PATTERN = re.compile(r"^(.+?)(\d+)(\.[^.]+)$")


def _isURL(value):
    """True if the input string is an http(s) URL."""
    return any(proto in str(value) for proto in ("https://", "http://"))


def _sanitize(value):
    """Replace filesystem-unsafe characters in a name fragment."""
    return _UNSAFE_CHARS.sub("_", str(value))


def _listImages(folderPath):
    """Return a sorted, de-duplicated list of image files in a folder.

    Globbing is case-insensitive on Windows/NTFS, so matching both ``*.png``
    and ``*.PNG`` returns each file twice. A set collapses the duplicates;
    without it, frame counts (and sequence detection) double on Windows.
    """
    found = set()
    for ext in IMAGE_EXTENSIONS:
        found.update(glob.glob(os.path.join(folderPath, f"*{ext}")))
        found.update(glob.glob(os.path.join(folderPath, f"*{ext.upper()}")))
    return sorted(found)


def detectImageSequence(folderPath):
    """
    Detects if a folder contains an image sequence and returns the sequence pattern.

    Args:
        folderPath: Path to the folder to check

    Returns:
        tuple: (sequencePattern, firstFrame, lastFrame, frameCount) or None if not a sequence
    """
    if not os.path.isdir(folderPath):
        return None

    imageFiles = _listImages(folderPath)
    if len(imageFiles) < 2:
        return None

    firstFile = os.path.basename(imageFiles[0])
    match = _SEQUENCE_PATTERN.match(firstFile)
    if not match:
        return None

    prefix, number, extension = match.groups()
    padding = len(number)
    expectedPattern = f"{prefix}%0{padding}d{extension}"

    frameNumbers = set()
    for imgFile in imageFiles:
        basename = os.path.basename(imgFile)
        m = _SEQUENCE_PATTERN.match(basename)
        if m and m.group(1) == prefix and m.group(3).lower() == extension.lower():
            try:
                frameNumbers.add(int(m.group(2)))
            except ValueError:
                continue

    if len(frameNumbers) < 2:
        return None

    frameNumbers = sorted(frameNumbers)
    sequencePath = os.path.join(folderPath, expectedPattern)
    return (sequencePath, frameNumbers[0], frameNumbers[-1], len(frameNumbers))


def getFirstImageInSequence(folderPath):
    """
    Gets the first image file in a folder for sequence detection.

    Args:
        folderPath: Path to the folder

    Returns:
        Path to the first image file, or None if no images found
    """
    if not os.path.isdir(folderPath):
        return None

    imageFiles = _listImages(folderPath)
    return imageFiles[0] if imageFiles else None


def _baseName(videoInput):
    """Derive the leading filename component from an input path/URL."""
    if _isURL(videoInput):
        return "TAS-YTDLP"
    if not videoInput:
        return "TAS"

    name = os.path.splitext(os.path.basename(videoInput))[0]
    # Image-sequence inputs look like ``frames_%05d`` after splitext; keep only
    # the human-readable prefix instead of leaking the printf pattern.
    if "%" in name:
        name = name.split("%", 1)[0].rstrip("_-. ") or "TAS"
    return name


def _resolveExtension(args, videoInput):
    """Pick the output file extension consistently for every input kind."""
    if getattr(args, "single_image_input", False):
        return ".png"
    if (
        getattr(args, "segment", False)
        or getattr(args, "encode_method", "") == "prores"
    ):
        return ".mov"
    if getattr(args, "encode_method", "") == "png":
        return ""  # png -> image-sequence directory, handled by the caller
    # URLs and printf-style sequence patterns have no trustworthy extension to
    # copy, so fall back to the container default rather than parsing garbage.
    if _isURL(videoInput) or (videoInput and "%" in str(videoInput)):
        return ".mp4"
    if videoInput:
        return os.path.splitext(videoInput)[1] or ".mp4"
    return ".mp4"


def generateOutputName(args, videoInput):
    """Generates output filename based on input and processing arguments."""
    baseName = _baseName(videoInput)

    features = [
        ("resize", "Resize", "resize_factor"),
        ("dedup", "Dedup", "dedup_sens"),
        ("interpolate", "Int", "interpolate_factor"),
        ("upscale", "Up", "upscale_factor"),
        ("sharpen", "Sh", "sharpen_sens"),
        ("restore", "Restore", "restore_method"),
        ("segment", "Segment", None),
        ("depth", "Depth", None),
        ("ytdlp", "YTDLP", None),
    ]

    isUrl = _isURL(videoInput)
    suffixes = []
    for arg, label, valAttr in features:
        # URL base name already carries the YTDLP tag; don't duplicate it.
        if arg == "ytdlp" and isUrl:
            continue
        if getattr(args, arg, False):
            val = getattr(args, valAttr, "") if valAttr else ""
            suffixes.append(f"-{label}{_sanitize(val)}")

    extension = _resolveExtension(args, videoInput)
    return f"{_sanitize(baseName)}{''.join(suffixes)}{extension}"


def _makeUniquePath(path, usedPaths, checkFs=True):
    """Return ``path``, or ``path-1``/``path-2``/... if it would collide.

    A name is considered taken if it was already handed out in this run
    (``usedPaths``) or, when ``checkFs`` is set, if it already exists on disk.
    ``checkFs=False`` is used for explicit user-named outputs: those should be
    overwritable, but still must not clobber each other inside one batch.
    """

    def taken(candidate):
        return candidate in usedPaths or (checkFs and os.path.exists(candidate))

    if not taken(path):
        usedPaths.add(path)
        return path

    root, ext = os.path.splitext(path)
    index = 1
    while taken(f"{root}-{index}{ext}"):
        index += 1
    unique = f"{root}-{index}{ext}"
    usedPaths.add(unique)
    return unique


def _makeUniqueDir(path, usedPaths):
    """Like :func:`_makeUniquePath` but for image-sequence output folders."""

    def taken(candidate):
        return candidate in usedPaths or os.path.exists(candidate)

    candidate = path
    index = 0
    while taken(candidate):
        index += 1
        candidate = f"{path}-{index}"
    usedPaths.add(candidate)
    return candidate


def generateOutputPath(video, output, defaultOutputPath, args, usedPaths):
    """Generates appropriate output path based on input parameters.

    ``usedPaths`` is a set shared across one batch; it guarantees that two
    inputs never resolve to the same output file (e.g. same basename in two
    folders, or one explicit ``--output file.mp4`` for many inputs).
    """
    # Explicit, fully-specified output file: honour the exact name the user
    # gave (allow overwriting a prior file), but disambiguate within a batch.
    if output and output.lower().endswith(OUTPUT_FILE_EXTENSIONS):
        return _makeUniquePath(output, usedPaths, checkFs=False)

    baseDir = output if output and os.path.isdir(output) else defaultOutputPath

    pngSequence = getattr(args, "encode_method", "") == "png" and not getattr(
        args, "png_passthrough", False
    )
    if pngSequence:
        outputName = generateOutputName(args, video)
        outputFolder = _makeUniqueDir(os.path.join(baseDir, outputName), usedPaths)
        os.makedirs(outputFolder, exist_ok=True)
        return os.path.join(outputFolder, "frames_%05d.png")

    return _makeUniquePath(
        os.path.join(baseDir, generateOutputName(args, video)), usedPaths
    )


WEBM_COMPATIBLE_ENCODERS = (
    "vp9",
    "qsv_vp9",
    "av1",
    "slow_av1",
    "nvenc_av1",
    "slow_nvenc_av1",
)


def validateEncoder(video, encodeMethod, customEncoder):
    """Validates and potentially adjusts the encoder method based on file type."""
    if (
        str(video).endswith(".webm")
        and not customEncoder
        and encodeMethod not in WEBM_COMPATIBLE_ENCODERS
    ):
        from src.utils.logAndPrint import logAndPrint

        logAndPrint(
            f"Video {video} is a Webm file, encode method was not set to {list(WEBM_COMPATIBLE_ENCODERS)} and `--custom_encoder` is None, defaulting to 'vp9'.",
            colorFunc="yellow",
        )
        return "vp9"
    return encodeMethod


def getVideoFiles(videosInput):
    """Extract list of video files from input specification."""
    # Handle semicolon separated paths
    if ";" in str(videosInput):
        paths = [v.strip() for v in str(videosInput).split(";") if v.strip()]
        all_files = []
        for p in paths:
            all_files.extend(getVideoFiles(p))
        return all_files

    # Handle URL input
    if _isURL(videosInput):
        return [videosInput]

    # Handle image sequence pattern (e.g., frames_%05d.png)
    if "%" in str(videosInput):
        return [videosInput]

    # Handle directory or file
    absPath = os.path.abspath(videosInput)
    if os.path.isdir(absPath):
        # First, check if this directory contains an image sequence
        sequenceInfo = detectImageSequence(absPath)
        if sequenceInfo:
            sequencePath, firstFrame, lastFrame, frameCount = sequenceInfo
            logging.info(
                f"Detected image sequence: {sequencePath} "
                f"(frames {firstFrame}-{lastFrame}, {frameCount} total)"
            )
            return [sequencePath]

        # Otherwise, look for video files in the directory (sorted for a stable,
        # reproducible batch order regardless of filesystem listing order).
        return sorted(
            os.path.join(absPath, f)
            for f in os.listdir(absPath)
            if os.path.splitext(f)[1].lower() in EXTENSIONS
        )

    if os.path.isfile(absPath):
        if absPath.endswith(".txt"):
            with open(absPath, "r") as f:
                return [
                    os.path.abspath(line.strip().strip('"'))
                    for line in f
                    if line.strip()
                ]
        return [absPath]

    # Fallback
    return [absPath]


def processInputOutputPaths(args, defaultOutputPath):
    """Processes input and output paths for video processing.

    Returns a list of per-video dicts (``videoPath``, ``outputPath``,
    ``encodeMethod``, ``customEncoder``), one entry per resolved input.
    """
    os.makedirs(defaultOutputPath, exist_ok=True)

    output = args.output
    if output:
        output = os.path.abspath(output)
        if not output.lower().endswith(OUTPUT_FILE_EXTENSIONS):
            os.makedirs(output, exist_ok=True)
        else:
            parent_dir = os.path.dirname(output)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)

    videoFiles = getVideoFiles(args.input)

    usedPaths = set()
    results = []
    for video in videoFiles:
        if not _isURL(video):
            # Skip existence check for image sequence patterns (they contain %)
            if "%" not in str(video) and not os.path.exists(video):
                raise FileNotFoundError(f"File {video} does not exist")

        results.append(
            {
                "videoPath": video,
                "outputPath": generateOutputPath(
                    video, output, defaultOutputPath, args, usedPaths
                ),
                "encodeMethod": validateEncoder(
                    video, args.encode_method, args.custom_encoder
                ),
                "customEncoder": args.custom_encoder,
            }
        )

    return results
