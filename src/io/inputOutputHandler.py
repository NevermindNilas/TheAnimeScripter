import glob
import logging
import os
import re

# Containers TAS will write. Every one of these is a container the default
# encoder can mux, because `_resolveExtension` copies an input's extension onto
# the output when the user gives no --output.
EXTENSIONS = [".mp4", ".mkv", ".webm", ".avi", ".mov", ".gif", ".m4v"]
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tiff", ".tif", ".exr", ".dpx"]

# Containers a directory batch will pick up. A superset of EXTENSIONS: TAS can
# decode plenty it cannot sensibly mux back out, and the two lists used to be
# one, so a folder batch silently dropped every .m4v (which PARAMETERS.MD
# advertises) and `--output clip.m4v` silently became a directory.
# `.ts` is deliberately absent: it is TypeScript at least as often as MPEG-TS,
# and scanning a source folder would feed .ts files to the decoder. It is
# listed in _SKIPPED_MEDIA_EXTENSIONS instead, so a batch says so out loud.
INPUT_EXTENSIONS = EXTENSIONS + [
    ".m2ts",
    ".mts",
    ".mpg",
    ".mpeg",
    ".m2v",
    ".flv",
    ".wmv",
    ".ogv",
]

# Media containers a directory batch deliberately does NOT pick up. Only used
# to make the skip audible -- warning about every .srt/.nfo/Thumbs.db next to
# the videos would be noise users learn to ignore.
_SKIPPED_MEDIA_EXTENSIONS = frozenset(
    {".ts", ".vob", ".asf", ".rm", ".rmvb", ".3gp", ".divx", ".mxf", ".f4v", ".y4m"}
)

# Extensions that look like a file rather than an output directory. Anything
# here is treated as "--output names a file"; see _looksLikeMediaFile for the
# warning that covers containers just outside it.
# .txt is an output-only extension: autoclip writes a cut list, and users may
# name it explicitly with --output cuts.txt.
OUTPUT_FILE_EXTENSIONS = tuple(EXTENSIONS + IMAGE_EXTENSIONS) + (".txt",)


def _looksLikeMediaFile(path):
    """True if ``path``'s extension names a media container TAS knows of.

    Used to decide whether an unrecognised ``--output`` deserves a warning:
    ``--output out.wmv`` is a user asking for a file, while
    ``--output D:\\renders\\ep01.v2`` is an ordinary directory that merely has
    a dot in its name.
    """
    ext = os.path.splitext(str(path))[1].lower()
    if not ext:
        return False
    return (
        ext in INPUT_EXTENSIONS
        or ext in IMAGE_EXTENSIONS
        or ext in _SKIPPED_MEDIA_EXTENSIONS
    )


# Characters that are illegal in filenames on Windows (and a good idea to avoid
# everywhere). Argument values get folded into output names, so scrub them.
_UNSAFE_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

# A video title becomes a file name: path separators and pipes read as " - ",
# the other unsafe characters are dropped. `%` (an image-sequence pattern to
# getVideoFiles, a field to yt-dlp's outtmpl) and `;` (getVideoFiles' path
# separator) would change how the downloaded file is read back as --input.
_TITLE_SEPARATORS = re.compile(r"\s*[|/\\]+\s*")
_TITLE_DROPPED = re.compile(r'[<>:"?*%;\x00-\x1f]')
_TITLE_MAX = 100

# The prefix may be empty: with `.+?` a digit-only name (`0001.png`, Blender's
# default) gave up its first digit as the prefix, so `0%03d.png` silently
# dropped every frame from `1000.png` on.
_SEQUENCE_PATTERN = re.compile(r"^(.*?)(\d+)(\.[^.]+)$")

# Image-sequence encoders and the per-frame extension each one writes. The
# single source for every sequence special case: extension resolution and
# folder containment here, the writer's pattern rewrite in
# src/io/ffmpegSettings.py, the audio-off rule in src/cli/validator.py, and
# the single-image guard in src/io/inputNormalization.py.
SEQUENCE_EXTENSIONS = {
    "png": ".png",
    "jpeg": ".jpg",
}


def isSingleImageInput(path) -> bool:
    """Whether ``path`` is one still image rather than a video or a sequence.

    Only a PNG is accepted as a single image (src/io/inputNormalization.py
    rejects the other still formats), and a ``%`` pattern is a sequence.
    The writer factory asks this too, because the standalone drivers (depth,
    segment, ...) build their own writers without saying so.
    """
    path = str(path or "")
    return "%" not in path and path.lower().endswith(".png")


_URL_PREFIXES = ("http://", "https://", "www.")


def isUrlInput(value) -> bool:
    """Whether an ``--input`` names a URL to download rather than a local path.

    The one URL test: input normalization routes on it and output naming
    reads it. It used to be two -- a bare ``startswith(("http", "www"))`` in
    normalization, so a local ``http_clip.mp4`` or a ``www`` folder was sent
    to the downloader and rejected as "not a YouTube URL". A path that exists
    is never a URL.
    """
    text = str(value or "").strip()
    return text.lower().startswith(_URL_PREFIXES) and not os.path.exists(text)


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
    # The narrowest number is the padding: the first name sorted is not it for
    # unpadded frames (`u_10.png` sorts before `u_7.png`).
    firstPadding = padding = len(number)
    matched = set()

    frameNumbers = set()
    for imgFile in imageFiles:
        basename = os.path.basename(imgFile)
        m = _SEQUENCE_PATTERN.match(basename)
        if m and m.group(1) == prefix and m.group(3).lower() == extension.lower():
            try:
                frameNumbers.add(int(m.group(2)))
                padding = min(padding, len(m.group(2)))
                matched.add(m.group(2))
            except ValueError:
                continue
    if any(f"{value:0{padding}d}" not in matched for value in frameNumbers):
        # Mixed padded and unpadded names: no one pattern names them all, so
        # keep the first name's, which reads the frames that share its width.
        padding = firstPadding
    expectedPattern = f"{prefix}%0{padding}d{extension}"

    if len(frameNumbers) < 2:
        return None

    frameNumbers = sorted(frameNumbers)
    expectedFrames = set(range(frameNumbers[0], frameNumbers[-1] + 1))
    missingFrames = sorted(expectedFrames.difference(frameNumbers))
    if missingFrames:
        preview = ", ".join(str(frame) for frame in missingFrames[:10])
        if len(missingFrames) > 10:
            preview = f"{preview}, ..."
        logging.warning(
            "Rejected image sequence %s: missing frame number(s): %s",
            folderPath,
            preview,
        )
        return None

    sequencePath = os.path.join(folderPath, expectedPattern)
    return (sequencePath, frameNumbers[0], frameNumbers[-1], len(frameNumbers))


# FFmpeg's image2 reader looks for a sequence's first frame at indices 0-4
# only (its start_number_range), and nelux exposes no start_number option.
_SEQUENCE_START_RANGE = 5
_PRINTF_FIELD = re.compile(r"%(?:0(\d+))?d")


# Staged pattern -> the pattern the user's frames live under. Output naming
# and the output-is-an-input guard read the original: the staged frames are
# links to it, so writing over the source would corrupt what is being decoded.
_STAGED_SOURCES = {}


def _patternStart(sequencePath):
    """The lowest frame number on disk for a printf pattern, or None."""
    folder, name = os.path.split(sequencePath)
    fields = list(_PRINTF_FIELD.finditer(name))
    if len(fields) != 1 or not os.path.isdir(folder or "."):
        return None
    field = fields[0]
    # `%d` prints no leading zero, so it never names a padded file.
    digits = rf"\d{{{int(field.group(1))},}}" if field.group(1) else r"0|[1-9]\d*"
    matcher = re.compile(
        re.escape(name[: field.start()])
        + f"({digits})"
        + re.escape(name[field.end() :])
        + "$"
    )
    numbers = [
        int(m.group(1)) for f in os.listdir(folder or ".") if (m := matcher.match(f))
    ]
    return min(numbers, default=None)


def _stageSequenceFromZero(sequencePath, firstFrame):
    """A 0-based linked copy of a sequence the decoder cannot find the start of.

    A sequence numbered from 5 or later (a VFX `shot_1001.png`, an export from
    frame 100) was detected and then failed to open with "No such file or
    directory". The frames are linked into a temporary folder removed at exit:
    the system one, or one beside the source folder where that is the only
    place a link works (Windows hard links stay on one drive and symlinks
    need a privilege). They are copied only when neither takes a link. The
    source folder itself is never written to.
    """
    import atexit
    import shutil
    import tempfile

    # Absolute: a symlink keeps the path it is given, and a relative one
    # would be read from the staging folder.
    folder, name = os.path.split(os.path.abspath(sequencePath))
    first = os.path.join(folder, name % firstFrame)
    if not os.path.exists(first):
        # Nothing matched after all; the original path gives the plain error.
        return sequencePath

    def link(source, target):
        for make in (os.link, os.symlink):
            try:
                make(source, target)
                return True
            except OSError:
                continue
        return False

    def newFolder(parent):
        made = tempfile.mkdtemp(prefix="tas-sequence-", dir=parent)
        atexit.register(shutil.rmtree, made, ignore_errors=True)
        return made

    stageFolder = None
    for parent in (None, os.path.dirname(os.path.abspath(folder))):
        try:
            candidate = newFolder(parent)
        except OSError:
            continue
        probe = os.path.join(candidate, name % 0)
        if link(first, probe):
            os.remove(probe)
            stageFolder = candidate
            break
        shutil.rmtree(candidate, ignore_errors=True)
    copied = stageFolder is None
    if copied:
        stageFolder = newFolder(None)

    index = firstFrame
    while os.path.exists(source := os.path.join(folder, name % index)):
        target = os.path.join(stageFolder, name % (index - firstFrame))
        if copied or not link(source, target):
            shutil.copy2(source, target)
            copied = True
        index += 1

    staged = os.path.join(stageFolder, name)
    _STAGED_SOURCES[staged] = sequencePath
    logging.info(
        f"Image sequence {sequencePath} starts at frame {firstFrame}; "
        f"reading it through {stageFolder}"
        + (" (frames copied: links are not available there)" if copied else "")
    )
    return staged


def _readableSequence(sequencePath, firstFrame=None):
    """``sequencePath``, or a 0-based staged twin when it starts too late."""
    try:
        if firstFrame is None:
            firstFrame = _patternStart(sequencePath)
        if firstFrame is None or firstFrame < _SEQUENCE_START_RANGE:
            return sequencePath
        return _stageSequenceFromZero(sequencePath, firstFrame)
    except (OSError, TypeError, ValueError) as e:
        logging.warning(f"Could not stage image sequence {sequencePath}: {e}")
        return sequencePath


def titleFileName(title):
    """A video title as a file name stem, or "" when nothing usable is left."""
    name = _TITLE_SEPARATORS.sub(" - ", str(title or ""))
    name = " ".join(_TITLE_DROPPED.sub("", name).split())
    # Windows drops trailing dots and spaces from a name.
    return name[:_TITLE_MAX].strip(" .-")


def _baseName(videoInput, title=None):
    """Derive the leading filename component from an input path/URL."""
    if isUrlInput(videoInput):
        return titleFileName(title) or "TAS-YTDLP"
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
    encodeMethod = getattr(args, "encode_method", "") or ""
    if getattr(args, "autoclip", False):
        # autoclip writes a cut list, not a video, whatever the input or
        # --encode_method: a PNG input named it <name>-Autoclip.png.
        return ".txt"
    if getattr(args, "single_image_input", False):
        # A single-image run keeps a sequence-format --encode_method for its
        # one-frame output (png or jpeg); every other method writes PNG.
        return SEQUENCE_EXTENSIONS.get(encodeMethod, ".png")
    if getattr(args, "segment", False) or encodeMethod == "prores":
        return ".mov"
    if encodeMethod in SEQUENCE_EXTENSIONS:
        return ""  # image-sequence directory, handled by the caller
    if encodeMethod == "gif":
        return ".gif"  # -c:v gif into the input's container fails the muxer
    # URLs and printf-style sequence patterns have no trustworthy extension to
    # copy, so fall back to the container default rather than parsing garbage.
    if isUrlInput(videoInput) or (videoInput and "%" in str(videoInput)):
        return ".mp4"
    if videoInput:
        # Only copy a container TAS can actually mux into -- EXTENSIONS is
        # exactly that set. A .wmv or .flv source would otherwise name a
        # .wmv/.flv output and die at header write, which is how widening the
        # batch scan turns a silent skip into a late crash after models load.
        extension = os.path.splitext(videoInput)[1]
        if extension.lower() not in EXTENSIONS:
            return ".mp4"
        if not _containerTakes(extension, encodeMethod):
            # e.g. an .avi source with x265, a .mov one with av1/vp9: the
            # muxer refused the codec at the first frame, after every model
            # had loaded. .mp4 takes every codec that reaches here.
            return ".mp4"
        return extension
    return ".mp4"


# Codec families each output container can mux, as nelux's muxer reports them
# (its "cannot be written into a '<fmt>' container" check). .webm is absent on
# purpose: validateEncoder turns a non-WebM method into vp9 for a .webm output.
_CONTAINER_CODECS = {
    ".mp4": {"h264", "hevc", "av1", "vp9"},
    ".m4v": {"h264"},
    ".mov": {"h264", "hevc", "prores", "gif"},
    ".mkv": {"h264", "hevc", "av1", "vp9", "prores"},
    ".avi": {"h264", "av1", "vp9"},
}


def _codecFamily(encodeMethod):
    """The codec an --encode_method writes (libx265 / hevc_nvenc -> "hevc")."""
    from src.io.encodingSettings import matchNeluxEncoder

    mapping = matchNeluxEncoder(encodeMethod) if encodeMethod else None
    codec = str((mapping or {}).get("codec", ""))
    for family, markers in (
        ("h264", ("x264", "h264")),
        ("hevc", ("x265", "hevc")),
        ("av1", ("av1",)),
        ("vp9", ("vp9",)),
        ("prores", ("prores",)),
        ("gif", ("gif",)),
    ):
        if any(marker in codec for marker in markers):
            return family
    return None


def _containerTakes(extension, encodeMethod):
    """Whether a copied input extension can hold this method's codec; True
    when either side is unknown, so nothing new is refused."""
    accepted = _CONTAINER_CODECS.get(extension.lower())
    family = _codecFamily(encodeMethod)
    return accepted is None or family is None or family in accepted


def _number(value):
    """Integral floats print as ints (35.0 -> "35", not "35.0")."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _resizeSuffix(target):
    if target[0] == "size":
        return f"{target[1]}x{target[2]}"
    if target[0] == "height":
        return f"{target[1]}p"
    return str(_number(target[1]))


def _chainSuffixes(chain):
    """One suffix per stage, kinds in the fixed order below and stages of a
    kind in chain order, so a chain without repeats or resize names its output
    exactly as before stages could repeat. Every value is what the user typed
    (sensitivity before mapping)."""
    suffixes = []
    for spec in chain:
        if spec.kind == "resize":
            suffixes.append(f"-Resize{_sanitize(_resizeSuffix(spec['target']))}")
    for spec in chain:
        if spec.kind == "interpolate" and spec.smooth:
            suffixes.append(f"-SmoothDedup{_sanitize(_number(spec['sens']))}")
    for spec in chain:
        if spec.kind == "interpolate":
            suffixes.append(f"-Int{_sanitize(_number(spec['factor']))}")
    for spec in chain:
        # Only when on: chains without resampling keep their names.
        if spec.kind == "interpolate" and spec.settings.get("resample", 0):
            suffixes.append(f"-Resample{spec.settings['resample']}")
    for spec in chain:
        if spec.kind == "upscale":
            suffixes.append(f"-Up{_sanitize(_number(spec['factor']))}")
    for spec in chain:
        if spec.kind == "restore":
            # list(), not the tuple: the name has always carried the list repr.
            suffixes.append(f"-Restore{_sanitize(list(spec['methods']))}")
    return suffixes


def generateOutputName(args, videoInput, title=None):
    """Generates output filename based on input and processing arguments.

    With ``args.chain`` (every CLI run) the stage suffixes come from the chain;
    a namespace without one (other entrypoints, tests) still names its output
    from the global flags. ``title`` names a URL input's output after the
    video instead of ``TAS-YTDLP``.
    """
    baseName = _baseName(videoInput, title)
    chain = getattr(args, "chain", None)

    stageFeatures = [
        ("smooth_dedup", "SmoothDedup", "smooth_dedup_sens"),
        ("interpolate", "Int", "interpolate_factor"),
        ("upscale", "Up", "upscale_factor"),
        ("restore", "Restore", "restore_method"),
    ]
    features = [
        ("segment", "Segment", None),
        ("depth", "Depth", None),
        ("autoclip", "Autoclip", None),
        ("ytdlp", "YTDLP", None),
    ]
    if chain is None:
        features = stageFeatures + features
    # --dedup is a run of its own, never a chain stage, so its suffix always
    # comes from the flag; it keeps its old place ahead of the stage suffixes.
    features = [("dedup", "Dedup", "dedup_sens")] + features

    isUrl = isUrlInput(videoInput)
    suffixes = [] if chain is None else _chainSuffixes(chain)
    for arg, label, valAttr in features:
        # A URL's name is the video title or TAS-YTDLP; no YTDLP suffix.
        if arg == "ytdlp" and isUrl:
            continue
        if getattr(args, arg, False):
            # Sensitivity flags are remapped in-place by the validator
            # (35 -> 0.965 for ssim), but the filename must carry what the
            # user typed, so prefer the raw snapshot when it exists.
            # Integral floats print as ints (35.0 -> "35", not "35.0").
            if valAttr:
                rawAttr = f"{valAttr}_raw"
                val = _number(getattr(args, rawAttr, getattr(args, valAttr, "")))
            else:
                val = ""
            suffixes.append(f"-{label}{_sanitize(val)}")

    extension = _resolveExtension(args, videoInput)
    return f"{_sanitize(baseName)}{''.join(suffixes)}{extension}"


def _pathMatcher(paths):
    """A test for "is this one of ``paths``": by normalized name (a pattern
    compares as text), or as the same file on disk (a link, another spelling)."""
    keys = {os.path.normcase(os.path.abspath(p)) for p in paths}
    onDisk = [p for p in paths if os.path.exists(p)]

    def matches(candidate):
        if os.path.normcase(os.path.abspath(candidate)) in keys:
            return True
        if not onDisk or not os.path.exists(candidate):
            return False
        for p in onDisk:
            try:
                if os.path.samefile(candidate, p):
                    return True
            except OSError:
                continue
        return False

    return matches


def _makeUniquePath(path, usedPaths, checkFs=True, avoid=()):
    """Return ``path``, or ``path-1``/``path-2``/... if it would collide.

    A name is considered taken if it was already handed out in this run
    (``usedPaths``) or, when ``checkFs`` is set, if it already exists on disk.
    ``checkFs=False`` is used for explicit user-named outputs: those should be
    overwritable, but still must not clobber each other inside one batch, nor
    be one of ``avoid`` -- the run's inputs.
    """

    isAvoided = _pathMatcher(avoid)

    def taken(candidate):
        return (
            candidate in usedPaths
            or (checkFs and os.path.exists(candidate))
            or isAvoided(candidate)
        )

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


def generateOutputPath(video, output, defaultOutputPath, args, usedPaths, inputs=None):
    """Generates appropriate output path based on input parameters.

    ``usedPaths`` is a set shared across one batch; it guarantees that two
    inputs never resolve to the same output file (e.g. same basename in two
    folders, or one explicit ``--output file.mp4`` for many inputs).
    ``inputs`` are the batch's input files (default: ``video``), none of which
    an explicit output may overwrite.
    """
    sequenceExt = SEQUENCE_EXTENSIONS.get(getattr(args, "encode_method", ""))
    if getattr(args, "png_passthrough", False) or getattr(args, "autoclip", False):
        # A single-image run writes one file, never a sequence folder, and
        # autoclip writes a cut list whatever --encode_method says: with png
        # or jpeg it went into a `<name>-Autoclip.txt` FOLDER as a text file
        # named `frames_%05d.png`.
        sequenceExt = None
    if getattr(args, "segment", False):
        # A segment matte is always ProRes 4444 with alpha (the writer turns
        # every method into prores_segment), so it is a .mov, not a sequence.
        sequenceExt = None

    # Explicit, fully-specified output file: honour the exact name the user
    # gave (allow overwriting a prior file), but disambiguate within a batch.
    if output and output.lower().endswith(OUTPUT_FILE_EXTENSIONS):
        if sequenceExt and "%" not in os.path.basename(output):
            # A single file path cannot hold an image sequence; use its stem
            # as the sequence folder so frames don't spray into the parent.
            outputFolder = _makeUniqueDir(os.path.splitext(output)[0], usedPaths)
            os.makedirs(outputFolder, exist_ok=True)
            return os.path.join(outputFolder, f"frames_%05d{sequenceExt}")
        # Never an input: the encoder truncates the file the decoder is still
        # reading, so the run failed part-way AND the source was lost (or, in
        # a batch, a later video wrote over an earlier one's source). An
        # explicit name is honoured or refused, like a pinned --preview_port:
        # a substitute could land on a file the user never named.
        sources = [video] if inputs is None else list(inputs)
        if _pathMatcher(sources)(output):
            if isSingleImageInput(output):
                # A single still is read whole before anything is written, so
                # writing it in place is the edit asked for, not a race.
                others = [s for s in sources if not _pathMatcher([s])(output)]
                return _makeUniquePath(output, usedPaths, checkFs=False, avoid=others)
            raise ValueError(
                f"--output {output} is one of the input files; writing to it "
                "would destroy the source while it is being read. Choose "
                "another output name."
            )
        # The -1/-2 names a batch hands out still skip every input.
        return _makeUniquePath(output, usedPaths, checkFs=False, avoid=sources)

    baseDir = output if output and os.path.isdir(output) else defaultOutputPath

    if sequenceExt:
        outputName = generateOutputName(args, video)
        outputFolder = _makeUniqueDir(os.path.join(baseDir, outputName), usedPaths)
        os.makedirs(outputFolder, exist_ok=True)
        return os.path.join(outputFolder, f"frames_%05d{sequenceExt}")

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


def validateEncoder(video, encodeMethod):
    """Validates and potentially adjusts the encoder method based on file type."""
    # Case-blind like every other extension test here: `clip.WEBM` kept
    # libx264, which the WebM muxer refuses after the whole run has processed.
    if str(video).lower().endswith(".gif") and encodeMethod != "gif":
        # The GIF muxer takes only the gif encoder: `--output clip.gif` kept
        # libx264 and failed at the first frame, after every model had loaded.
        from src.infra.logAndPrint import logAndPrint

        logAndPrint(
            f"Output {video} is a GIF, so it is encoded with 'gif' instead of "
            f"'{encodeMethod}'.",
            colorFunc="yellow",
        )
        return "gif"
    isWebm = str(video).lower().endswith(".webm")
    if isWebm and encodeMethod not in WEBM_COMPATIBLE_ENCODERS:
        from src.infra.logAndPrint import logAndPrint

        substitute = "vp9"
        logAndPrint(
            f"Video {video} is a Webm file and encode method was not set to {list(WEBM_COMPATIBLE_ENCODERS)}, defaulting to '{substitute}'.",
            colorFunc="yellow",
        )
        return substitute
    return encodeMethod


# Containers that take the ProRes 4444 a segment matte is always encoded as
# (matchNeluxEncoder's prores_segment); nelux names exactly these two.
SEGMENT_CONTAINERS = (".mov", ".mkv")


def validateSegmentOutput(video, outputPath):
    """Refuse a segment output no ProRes 4444 can be written into.

    Before any model loads: the matte is ProRes 4444 with alpha whatever
    --encode_method says, so `--output matte.mp4` (or .png, .webm, ...)
    segmented every frame and then failed ("prores_ks cannot be written into
    a 'mp4' container"). Refused rather than renamed, as an output naming an
    input is: a substitute could be a file the user never named. A single
    still (written as a PNG with alpha) is not affected.
    """
    if isSingleImageInput(video):
        return
    extension = os.path.splitext(str(outputPath))[1].lower()
    if extension not in SEGMENT_CONTAINERS:
        raise ValueError(
            f"--segment writes ProRes 4444 with alpha, which {outputPath} "
            f"cannot hold; name a {' or '.join(SEGMENT_CONTAINERS)} --output."
        )


def _readListFile(path):
    """The text of a ``.txt`` batch list, whatever saved it.

    A BOM names the encoding: UTF-16 is what Windows PowerShell 5's ``>``
    writes, UTF-8 with a BOM what Notepad offers. Read with the locale's
    encoding, the first gave a decode error or NUL-riddled paths and the
    second glued the BOM onto the first path, and either aborted the batch.
    Without a BOM, UTF-8 comes first and the locale's encoding -- the only one
    tried before -- is the fallback.
    """
    import codecs
    import locale

    with open(path, "rb") as f:
        data = f.read()
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return data.decode("utf-16")
    if data.startswith(codecs.BOM_UTF8):
        return data[len(codecs.BOM_UTF8) :].decode("utf-8")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode(locale.getpreferredencoding(False))


def _resolveListEntry(entry, listFolder):
    """One line of a ``.txt`` batch list as an absolute path.

    A relative line is looked up next to the list first: `dir /b > list.txt`
    writes bare names, which only resolved when TAS was started from the video
    folder. A list written relative to the working directory still works.
    """
    if isUrlInput(entry):
        return entry
    if os.path.isabs(entry):
        return os.path.abspath(entry)
    besideList = os.path.abspath(os.path.join(listFolder, entry))
    fromCwd = os.path.abspath(entry)

    def found(path):
        # A `%05d` pattern names no file; its folder is the best test there is.
        return os.path.exists(os.path.dirname(path) if "%" in path else path)

    if found(fromCwd) and not found(besideList):
        return fromCwd
    return besideList


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
    if isUrlInput(videosInput):
        return [videosInput]

    # Handle image sequence pattern (e.g., frames_%05d.png)
    if "%" in str(videosInput):
        return [_readableSequence(str(videosInput))]

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
            return [_readableSequence(sequencePath, firstFrame)]

        # Otherwise, look for video files in the directory (sorted for a stable,
        # reproducible batch order regardless of filesystem listing order).
        videos = []
        skipped = []
        for f in sorted(os.listdir(absPath)):
            extension = os.path.splitext(f)[1].lower()
            if extension in INPUT_EXTENSIONS:
                videos.append(os.path.join(absPath, f))
            elif extension in _SKIPPED_MEDIA_EXTENSIONS:
                skipped.append(f)

        if skipped:
            # Lazy import: src/cli/validator.py imports this module at module
            # scope, and these modules must stay importable without torch.
            from src.infra.logAndPrint import logAndPrint

            preview = ", ".join(skipped[:5])
            if len(skipped) > 5:
                preview += f", ... (+{len(skipped) - 5} more)"
            logAndPrint(
                f"Skipped {len(skipped)} unsupported file(s) in {absPath}: "
                f"{preview}. Pass one directly with --input to try it anyway.",
                colorFunc="yellow",
            )

        return videos

    if os.path.isfile(absPath):
        # Case-blind like every other extension test: `LIST.TXT` was probed
        # as a video and failed with "Invalid data found when processing input".
        if absPath.lower().endswith(".txt"):
            from io import StringIO

            # Split on \r\n, \r and \n only, as reading the file in text mode
            # did: str.splitlines() also cuts at U+2028, U+0085, \f, ..., which
            # file names may contain.
            listFolder = os.path.dirname(absPath)
            entries = [
                _resolveListEntry(line.strip().strip('"'), listFolder)
                for line in StringIO(_readListFile(absPath), newline=None)
                if line.strip()
            ]
            return [
                _readableSequence(entry) if "%" in entry else entry for entry in entries
            ]
        return [absPath]

    # Fallback
    return [absPath]


def processInputOutputPaths(args, defaultOutputPath):
    """Processes input and output paths for video processing.

    Returns a list of per-video dicts (``videoPath``, ``outputPath``,
    ``encodeMethod``), one entry per resolved input.
    """
    os.makedirs(defaultOutputPath, exist_ok=True)

    output = args.output
    if output:
        output = os.path.abspath(output)
        if not output.lower().endswith(OUTPUT_FILE_EXTENSIONS):
            if _looksLikeMediaFile(output):
                # A media extension TAS cannot write. Creating a directory
                # named `clip.wmv` holding a differently-named .mp4 is a
                # defensible fallback, but doing it silently breaks any
                # wrapper script that expects the file it named.
                from src.infra.logAndPrint import logAndPrint

                logAndPrint(
                    f"--output {output} names a container TAS does not write, "
                    "so it is being treated as an output FOLDER, not a file. "
                    f"Supported output containers: "
                    f"{', '.join(EXTENSIONS)}.",
                    colorFunc="yellow",
                )
            os.makedirs(output, exist_ok=True)
        else:
            parent_dir = os.path.dirname(output)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)

    videoFiles = getVideoFiles(args.input)

    usedPaths = set()
    results = []
    # A staged sequence is decoded from its 0-based twin and named, and
    # guarded against being overwritten, as the sequence the user gave.
    sources = [_STAGED_SOURCES.get(video, video) for video in videoFiles]
    localInputs = [video for video in sources if not isUrlInput(video)]
    for decodePath, video in zip(videoFiles, sources, strict=True):
        if not isUrlInput(video):
            # Skip existence check for image sequence patterns (they contain %)
            if "%" not in str(video) and not os.path.exists(video):
                raise FileNotFoundError(f"File {video} does not exist")

        outputPath = generateOutputPath(
            video, output, defaultOutputPath, args, usedPaths, inputs=localInputs
        )
        if getattr(args, "segment", False):
            validateSegmentOutput(video, outputPath)
        results.append(
            {
                "videoPath": decodePath,
                "outputPath": outputPath,
                # Keyed on the OUTPUT, which is what the muxer sees. Passing the
                # input only worked because the output usually copies the input's
                # extension: an explicit `--output out.webm` from an .mp4 source
                # kept libx264 and FFmpeg refused to write the header, leaving a
                # 262-byte file the run still reported as a success.
                "encodeMethod": validateEncoder(outputPath, args.encode_method),
            }
        )

    return results
