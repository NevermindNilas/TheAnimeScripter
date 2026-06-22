import json
import logging
import os
import subprocess
import textwrap

import src.constants as cs

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
        "huffyuf",
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


def isNvdecCompatible(codec: str, pixFmt: str) -> bool:
    """Return True if the source codec + pix_fmt can be decoded by NVDEC.

    Conservative: only rules out cases the NVDEC path provably cannot handle
    (raw/uncompressed codecs, packed RGB/BGR/GBR pixel formats). Anything
    unclear is allowed through so valid hardware-decodable input is not
    surprise-downgraded to CPU.
    """
    codecNorm = (codec or "").lower()
    pixFmtNorm = (pixFmt or "").lower()
    if codecNorm in _NVDEC_UNSUPPORTED_CODECS:
        return False
    if pixFmtNorm.startswith(_NVDEC_UNSUPPORTED_PIXFMT_PREFIXES):
        return False
    return True


def saveMetadata(metadata, videoDataDump=None):
    metadataPath = os.path.join(cs.WHEREAMIRUNFROM, "metadata.json")
    with open(metadataPath, "w") as jsonFile:
        data = {
            "metadata": metadata,
            "FFPROBE DUMP": videoDataDump if videoDataDump else None,
        }
        json.dump(data, jsonFile, indent=4)

    cs.METADATAPATH = metadataPath


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
        if not os.path.exists(cs.FFMPEGPATH):
            logging.error("ffprobe not found")
            raise FileNotFoundError("ffprobe path not found")
        if not os.path.exists(inputPath):
            logging.error("Video file not found")
            raise FileNotFoundError("Video file not found")

        cmd = [
            cs.FFPROBEPATH,
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            "-count_packets",
            inputPath,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )

        if not result.stdout:
            raise Exception("No output received from ffprobe")

        probeData = json.loads(result.stdout)

        # Get video stream
        videoStream = next(
            stream for stream in probeData["streams"] if stream["codec_type"] == "video"
        )

        # Check for audio streams
        # If the condition is true, then we can check for audio streams, otherwise we can skip this step
        if cs.AUDIO:
            hasAudio = any(
                stream["codec_type"] == "audio" for stream in probeData["streams"]
            )
            # Is there any audio stream in the video to begin with?
            # If yes, then set the global variable accordingly
            cs.AUDIO = hasAudio
        else:
            hasAudio = False

        # Extract metadata
        width = int(videoStream["width"])
        height = int(videoStream["height"])
        fpsValue = videoStream.get("r_frame_rate", "1/1")
        fpsParts = fpsValue.split("/") if isinstance(fpsValue, str) else ["1", "1"]
        try:
            fpsNum = float(fpsParts[0]) if len(fpsParts) > 0 else 1.0
            fpsDen = float(fpsParts[1]) if len(fpsParts) > 1 else 1.0
            fps = fpsNum / fpsDen if fpsDen != 0 else 1.0
        except Exception:
            fps = 1.0

        durationRaw = probeData.get("format", {}).get("duration", 0)
        try:
            duration = float(durationRaw)
        except Exception:
            duration = 0.0

        totalFramesRaw = videoStream.get("nb_read_packets")
        if totalFramesRaw in (None, "N/A", ""):
            totalFramesRaw = videoStream.get("nb_frames", 0)
        try:
            totalFrames = int(totalFramesRaw)
        except Exception:
            totalFrames = 0

        imageExtensions = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".exr", ".dpx"}
        isImageInput = os.path.splitext(inputPath)[1].lower() in imageExtensions
        if isImageInput and totalFrames < 1:
            totalFrames = 1
        colorFormat = videoStream.get("pix_fmt", "unknown")
        pixelFormat = videoStream.get("color_primaries", "unknown")
        colorSpace = videoStream.get("color_space", "unknown")
        ColorTRT = videoStream.get("color_transfer", "unknown")
        ColorRange = videoStream.get("color_range", "unknown")

        if outPoint != 0 and not isImageInput:
            totalFramesToProcess = int((outPoint - inPoint) * fps)
        else:
            totalFramesToProcess = totalFrames

        if isImageInput and totalFramesToProcess < 1:
            totalFramesToProcess = 1

        metadata = {
            "Width": width,
            "Height": height,
            "AspectRatio": round(width / height, 2),
            "FPS": round(fps, 2),
            "Codec": videoStream["codec_name"],
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

        saveMetadata(metadata, videoStream)
        return metadata

    except Exception as e:
        logging.error(f"Error getting metadata with ffprobe: {e}")
        raise
