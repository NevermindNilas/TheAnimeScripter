import json
import logging
import os
import textwrap

import src.constants as cs
from src.infra.logAndPrint import logAndPrint
from src.io.runOutcome import isSequencePattern

# Codecs nelux's NVDEC path cannot decode. NVDEC's cuvid covers compressed
# streams only (H.264/HEVC/VP9/AV1/MPEG2/MPEG4/VC1/VP8/MJPEG); uncompressed
# and lossless-intermediate codecs have no cuvid decoder and either fail
# opaquely or deadlock the parser. Sourced from the nelux README codec table
# (https://github.com/NevermindNilas/Nelux#supported-codecs--formats).
_NVDEC_UNSUPPORTED_CODECS: frozenset[str] = frozenset(
    {
        "rawvideo",
        "ffv1",
        "ffvhuff",
        "huffyuv",
        "lagarith",
        "utvideo",
        "qtrle",
        "qdraw",
        "8bps",
        "cinepak",
        "msrle",
        "msvideo1",
        "rle",
        "vp6",
        "vp6a",
        "vp6f",
    }
)

# NVDEC outputs YUV-family pix_fmts only (NV12/P010/P016/YUV444 8/10/12/16-bit).
# Packed RGB/BGR/GBR/palette/gray-alpha sources have no NVDEC path.
_NVDEC_UNSUPPORTED_PIXFMT_PREFIXES: tuple[str, ...] = (
    "bgr",
    "rgb",
    "gbr",
    "pal",
    "ya",
)


def isNvdecCompatible(codec: str | None, pixFmt: str | None) -> bool:
    """Return True if the source codec + pix_fmt can be decoded by NVDEC.

    Conservative: only rules out cases the NVDEC path provably cannot handle
    (raw/uncompressed codecs, packed RGB/BGR/GBR pixel formats). Anything
    unclear is allowed through so valid hardware-decodable input is not
    surprise-downgraded to CPU. ``None``/empty inputs are treated as
    "unknown" and allowed through.
    """
    codecNorm = (codec or "").lower()
    pixFmtNorm = (pixFmt or "").lower()
    if codecNorm in _NVDEC_UNSUPPORTED_CODECS:
        return False
    if pixFmtNorm.startswith(_NVDEC_UNSUPPORTED_PIXFMT_PREFIXES):
        return False
    return True


def saveMetadata(metadata, videoDataDump=None):
    # Hand the writers an in-process copy first. The file below stays for
    # external consumers (the .jsx panel lives in another repo), but it is a
    # single fixed install-dir path shared by every TAS process on the machine,
    # so it cannot be the input to this run's colour decision -- a concurrent
    # run overwriting it between this write and the writer's read is what made
    # a BT.2020 source encode as bt709. Copy rather than alias: the caller keeps
    # using this dict.
    cs.PROBED_METADATA = dict(metadata)

    metadataPath = os.path.join(cs.WHEREAMIRUNFROM, "metadata.json")
    with open(metadataPath, "w") as jsonFile:
        data = {
            "metadata": metadata,
            "FFPROBE DUMP": videoDataDump if videoDataDump else None,
        }
        json.dump(data, jsonFile, indent=4)

    cs.METADATAPATH = metadataPath


def resolveSourceFps(props):
    """Pick the fps the pipeline should treat as the source rate.

    Returns ``(fps, warning)`` where ``warning`` is a printable message when
    the pick deserves a console note, else ``None``.

    Exact fps comes from the integer ratio (e.g. 24000/1001 = 23.9760…);
    rounding leaked 23.98 into the encoder -r and drifted timing. On VFR
    sources r_frame_rate is the highest instantaneous rate, not the real one —
    while the decoder (nelux VideoReader in BuildBuffer) indexes frames with
    the AVERAGE rate. Tagging the output with r_frame_rate therefore played it
    several times too fast against full-length audio. Prefer the average rate
    whenever the container says VFR or the two rates genuinely disagree; CFR
    sources keep r_frame_rate so its exactness is unchanged.
    """

    def _prop(key, default=None):
        val = props.get(key, default)
        return default if val in (None, "N/A", "") else val

    def _ratio(numKey, denKey):
        try:
            num = _prop(numKey)
            den = _prop(denKey)
            return float(num) / float(den) if num and den else None
        except TypeError, ValueError, ZeroDivisionError:
            return None

    rFps = _ratio("r_frame_rate_num", "r_frame_rate_den")
    avgFps = _ratio("avg_frame_rate_num", "avg_frame_rate_den")
    isVfr = bool(_prop("is_vfr", False))

    fps = rFps
    warning = None
    ratesDisagree = (
        rFps is not None
        and avgFps is not None
        and abs(rFps - avgFps) > 0.01 * max(rFps, avgFps)
    )
    if avgFps and (isVfr or ratesDisagree):
        if ratesDisagree:
            warning = (
                f"Variable frame rate input: using its average rate "
                f"({avgFps:.3f} fps, r_frame_rate claims {rFps:.3f}); the "
                f"output will be constant-frame-rate."
            )
        fps = avgFps
    if not fps:
        fps = float(_prop("fps", 1.0) or 1.0)
    return fps, warning


def trimFrameRange(fps, inPoint, outPoint):
    """The ``[start, end)`` source frame indices ``--inpoint``/``--outpoint`` select.

    The single definition of the trim arithmetic. It used to be written twice:
    ``BuildBuffer`` rounded each endpoint independently while this module
    floored the product, so for any non-integer ``(outpoint-inpoint)*fps`` --
    which on 23.976/29.97 material is nearly every trim -- the decoder emitted
    one more frame than ``TotalFramesToBeProcessed`` claimed, and the drivers
    that use that number as a hard ``range()`` bound stopped a frame early.

    ``end`` is ``None`` when no ``--outpoint`` was given, meaning "decode to
    EOF". Otherwise it is floored to at least one frame past ``start``: the
    end test used to be gated on the derived index (``endFrame > 0``) rather
    than on whether a trim was requested, so an ``--outpoint`` under half a
    frame (20.9 ms at 23.98 fps, but 0.5 s on a 1 fps timelapse) collapsed to
    0 and read as "no limit", re-encoding the whole file against a fraction of
    a second of audio. With a non-zero ``--inpoint`` the same collapse made
    ``end == start`` and emitted no frames at all.
    """
    startFrame = int(round(float(inPoint) * fps)) if inPoint and inPoint > 0 else 0
    if not outPoint or float(outPoint) <= 0:
        return startFrame, None
    return startFrame, max(startFrame + 1, int(round(float(outPoint) * fps)))


def getVideoMetadata(inputPath, inPoint, outPoint):
    """
    Get metadata from a video file using ffprobe.

    Parameters:
    inputPath (str): The path to the video file
    inPoint (float): Start time of clip
    outPoint (float): End time of clip
    ffprobePath (str): Path to ffprobe executable

    Returns:
    tuple: (width, height, fps, totalFramesToProcess, hasAudio)
    """
    try:
        # An image sequence arrives as an FFmpeg pattern (frames_%05d.png), which
        # is never a path on disk -- every other layer already exempts it
        # (ffmpegSettings.BuildBuffer, inputOutputHandler's existence check,
        # detectImageSequence, which manufactures the pattern in the first
        # place). This one did not, and it runs first, so every sequence input
        # died here with "Video file not found" before anything opened it. Only
        # a real image2 counter earns the exemption -- a literal percent in an
        # ordinary name (50%_off.mp4) would otherwise slip a missing file past
        # this guard, to fail later as an opaque probe error.
        if not isSequencePattern(inputPath) and not os.path.exists(inputPath):
            logging.error("Video file not found")
            raise FileNotFoundError("Video file not found")

        # nelux reads container/stream metadata through the same libavformat that
        # ffprobe used, so it replaces the ffprobe subprocess entirely. The probe
        # is header-based (~8-14ms, size-independent) vs ffprobe's ~30-90ms that
        # grew with file size. torch must be imported before nelux.
        import torch  # noqa: F401,I001
        import nelux

        imageExtensions = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".exr", ".dpx"}
        isImageInput = os.path.splitext(inputPath)[1].lower() in imageExtensions

        # nelux.probe() opens the container and reads stream info only -- no
        # decoder, no resolution-sized frame buffer, no worker threads -- and
        # returns the same dict as VideoReader.properties. That makes it both
        # cheaper than constructing a reader (~0.6ms/360p, ~17ms/4K vs a full
        # reader's ~20-65ms) and immune to the nvdec-construction deadlock on
        # uncompressed/odd codecs, since it never opens a decoder. FFmpeg DLLs
        # are already on the search path (src/cli/startup.py -> getFFMPEG).
        normPath = os.path.normpath(inputPath)
        if hasattr(nelux, "probe"):
            props = dict(nelux.probe(normPath))
        else:
            # nelux < 0.15.1 has no probe(); fall back to a decoder-less-as-
            # possible VideoReader read (still nelux, still no ffprobe).
            reader = nelux.VideoReader(normPath, decode_accelerator="cpu")
            try:
                props = dict(reader.get_properties())
            finally:
                del reader

        def _prop(key, default=None):
            val = props.get(key, default)
            return default if val in (None, "N/A", "") else val

        width = int(_prop("width", 0) or 0)
        height = int(_prop("height", 0) or 0)
        if width <= 0 or height <= 0:
            raise ValueError(f"nelux returned no video dimensions for {inputPath}")

        fps, fpsWarning = resolveSourceFps(props)
        if fpsWarning:
            logAndPrint(fpsWarning, "yellow")

        try:
            duration = float(_prop("duration", 0.0) or 0.0)
        except TypeError, ValueError:
            duration = 0.0

        # Header frame count; duration*fps as a last resort (VFR / odd headers).
        # The progress-bar total self-corrects at stream end (main.py
        # bar.updateTotal), and no frame is ever skipped over this count.
        try:
            totalFrames = int(_prop("nb_frames", 0) or _prop("total_frames", 0) or 0)
        except TypeError, ValueError:
            totalFrames = 0
        if totalFrames < 1 and duration and fps:
            totalFrames = int(duration * fps)
        if isImageInput and totalFrames < 1:
            totalFrames = 1

        # Gate on the run-wide intent, not on cs.AUDIO: this runs once per video
        # in a batch and writes cs.AUDIO back, so reading cs.AUDIO here made the
        # first silent video latch audio off for every video after it.
        hasAudio = bool(cs.AUDIO_REQUESTED and _prop("has_audio", False))
        cs.AUDIO = hasAudio

        codecName = _prop("codec_name") or _prop("codec", "unknown")
        colorFormat = _prop("pixel_format", "unknown")
        pixelFormat = _prop("color_primaries", "unknown")
        colorSpace = _prop("color_space", "unknown")
        ColorTRT = _prop("color_transfer", "unknown")
        ColorRange = _prop("color_range", "unknown")

        if isImageInput:
            totalFramesToProcess = totalFrames
        else:
            # Same helper the decoder uses, so the count and the decode agree.
            startFrame, endFrame = trimFrameRange(fps, inPoint, outPoint)
            if endFrame is not None:
                totalFramesToProcess = endFrame - startFrame
            elif totalFrames:
                totalFramesToProcess = max(totalFrames - startFrame, 0)
            else:
                totalFramesToProcess = totalFrames

        if isImageInput and totalFramesToProcess < 1:
            totalFramesToProcess = 1

        metadata = {
            "Width": width,
            "Height": height,
            "AspectRatio": round(width / height, 2),
            # Store the exact fps (e.g. 24000/1001 = 23.9760…). Rounding to 2
            # decimals here leaked 23.98 into the encoder's -r, drifting timing
            # against a 23.976 comp (frames cut off). Round only for display.
            "FPS": fps,
            "Codec": codecName,
            "ColorRange": ColorRange,
            "ColorFormat": colorFormat,
            "ColorSpace": colorSpace,
            "ColorTRT": ColorTRT,
            "PixelFormat": pixelFormat,
            "Duration": duration,
            "Inpoint": inPoint,
            "Outpoint": outPoint,
            "NumberOfTotalFrames": totalFrames,
            "TotalFramesToBeProcessed": totalFramesToProcess,
            "HasAudio": hasAudio,
        }

        logging.info(
            textwrap.dedent(f"""
        ============== Video Metadata ==============
        Width: {width}
        Height: {height}
        AspectRatio: {metadata["AspectRatio"]}
        FPS: {round(fps, 2)}
        Codec: {metadata["Codec"]}
        ColorRange: {ColorRange}
        ColorFormat: {colorFormat},
        ColorSpace: {colorSpace},
        ColorTRTR: {ColorTRT},
        Duration: {duration} seconds
        Inpoint: {inPoint}
        Outpoint: {outPoint}
        Number of total frames: {totalFrames}
        Total frames to be processed: {totalFramesToProcess}
        Has Audio: {hasAudio}""")
        )

        saveMetadata(metadata, props)
        return metadata

    except Exception as e:
        logging.error(f"Error getting metadata with nelux: {e}")
        raise
