from src.infra.logAndPrint import logWarning


def matchEncoder(encode_method: str):
    """
    encode_method: str - The method to use for encoding the video. Options include "x264", "x264_animation", "nvenc_h264", etc.
    """
    command = []
    match encode_method:
        case "x264":
            command.extend(
                [
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "15",
                ]
            )
        case "slow_x264":
            command.extend(
                [
                    "-c:v",
                    "libx264",
                    "-preset",
                    "slow",
                    "-crf",
                    "18",
                    "-tune",
                    "animation",
                    "-g",
                    "240",
                ]
            )

        case "x264_10bit":
            command.extend(
                [
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "15",
                    "-profile:v",
                    "high10",
                ]
            )
        case "x264_animation":
            command.extend(
                [
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-tune",
                    "animation",
                    "-crf",
                    "15",
                ]
            )
        case "x264_animation_10bit":
            command.extend(
                [
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-tune",
                    "animation",
                    "-crf",
                    "15",
                    "-profile:v",
                    "high10",
                ]
            )
        case "x265":
            command.extend(
                [
                    "-c:v",
                    "libx265",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "15",
                ]
            )

        case "slow_x265":
            command.extend(
                [
                    "-c:v",
                    "libx265",
                    "-preset",
                    "slow",
                    "-crf",
                    "18",
                    "-profile:v",
                    "main",
                    "-level",
                    "5.1",
                    "-tune",
                    "ssim",
                    "-g",
                    "240",
                ]
            )
        case "x265_10bit":
            command.extend(
                [
                    "-c:v",
                    "libx265",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "15",
                    "-profile:v",
                    "main10",
                    "-x265-params",
                    "log-level=0",
                ]
            )
        case "nvenc_h264":
            command.extend(["-c:v", "h264_nvenc", "-preset", "p1", "-cq", "15"])
        case "slow_nvenc_h264":
            command.extend(
                [
                    "-c:v",
                    "h264_nvenc",
                    "-preset",
                    "p7",
                    "-cq",
                    "15",
                    "-b:v",
                    "0",
                    "-g",
                    "240",
                ]
            )
        case "nvenc_h265":
            command.extend(["-c:v", "hevc_nvenc", "-preset", "p1", "-cq", "15"])

        case "slow_nvenc_h265":
            command.extend(
                [
                    "-c:v",
                    "hevc_nvenc",
                    "-preset",
                    "p7",
                    "-cq",
                    "12",
                    "-b:v",
                    "0",
                    "-g",
                    "240",
                ]
            )
        case "nvenc_h265_10bit":
            command.extend(
                [
                    "-c:v",
                    "hevc_nvenc",
                    "-preset",
                    "p1",
                    "-cq",
                    "15",
                    "-profile:v",
                    "main10",
                ]
            )
        case "qsv_h264":
            command.extend(
                ["-c:v", "h264_qsv", "-preset", "veryfast", "-global_quality", "15"]
            )
        case "qsv_h265":
            command.extend(
                ["-c:v", "hevc_qsv", "-preset", "veryfast", "-global_quality", "15"]
            )
        case "qsv_h265_10bit":
            command.extend(
                [
                    "-c:v",
                    "hevc_qsv",
                    "-preset",
                    "veryfast",
                    "-global_quality",
                    "15",
                    "-profile:v",
                    "main10",
                ]
            )
        case "nvenc_av1":
            command.extend(["-c:v", "av1_nvenc", "-preset", "p1", "-cq", "15"])

        case "slow_nvenc_av1":
            command.extend(
                [
                    "-c:v",
                    "av1_nvenc",
                    "-preset",
                    "p7",
                    "-cq",
                    "15",
                    "-b:v",
                    "0",
                    "-g",
                    "240",
                ]
            )

        case "av1":
            command.extend(
                [
                    "-c:v",
                    "libsvtav1",
                    "-preset",
                    "8",
                    "-crf",
                    "15",
                ]
            )

        case "slow_av1":
            command.extend(
                [
                    "-c:v",
                    "libsvtav1",
                    "-preset",
                    "4",
                    "-crf",
                    "27",
                    "-g",
                    "240",
                    "-b:v",
                    "0",
                    "-row-mt",
                    "1",
                ]
            )
        case "h264_amf":
            command.extend(
                ["-c:v", "h264_amf", "-quality", "speed", "-rc", "cqp", "-qp", "15"]
            )
        case "hevc_amf":
            command.extend(
                ["-c:v", "hevc_amf", "-quality", "speed", "-rc", "cqp", "-qp", "15"]
            )
        case "hevc_amf_10bit":
            command.extend(
                [
                    "-c:v",
                    "hevc_amf",
                    "-quality",
                    "speed",
                    "-rc",
                    "cqp",
                    "-qp",
                    "15",
                    "-profile:v",
                    "main10",
                ]
            )
        case "prores" | "prores_segment":
            command.extend(["-c:v", "prores_ks", "-profile:v", "4", "-qscale:v", "15"])
        case "gif":
            command.extend(["-c:v", "gif", "-qscale:v", "1", "-loop", "0"])
        case "vp9":
            command.extend(["-c:v", "libvpx-vp9", "-crf", "15", "-preset", "veryfast"])
        case "qsv_vp9":
            command.extend(["-c:v", "vp9_qsv", "-preset", "veryfast"])

        case "lossless":
            command.extend(
                [
                    "-c:v",
                    "libx264",
                    "-preset",
                    "ultrafast",
                    "-crf",
                    "0",
                ]
            )
        case "lossless_nvenc" | "lossless_nvenc_h264":
            command.extend(
                [
                    "-c:v",
                    "h264_nvenc",
                    "-preset",
                    "p1",
                    "-qp",
                    "0",
                    "-b:v",
                    "0",
                ]
            )
        case "png":
            command.extend(
                [
                    "-c:v",
                    "png",
                    "-q:v",
                    "0",
                ]
            )
        case _:
            # Without this arm an unhandled name returns [], the command carries
            # no -c:v, and FFmpeg quietly encodes with the container default at
            # its own CRF -- a wrong-looking output file with nothing in the log
            # to explain it. No CLI choice can reach this today (WriteBuffer
            # maps the *_nelux names to their twins first, and every remaining
            # choice is pinned to an arm by tests/test_registryDrift.py); it is
            # here so the next unmapped name is loud instead of silent.
            logWarning(
                f"Unrecognized encode method '{encode_method}'. FFmpeg will "
                "pick the container's default encoder and quality."
            )

    return command


def getPixFMT(encode_method, bitDepth, grayscale, transparent):
    """
    Return (inputPixFormat, outputPixFormat, encode_method) based on settings.
    """
    if bitDepth == "8bit":
        defaultInPixFMT = "rgb24"
        defaultOutPixFMT = "yuv420p"
    else:
        defaultInPixFMT = "rgb48le"
        defaultOutPixFMT = "yuv444p10le"

    inPixFmt = defaultInPixFMT
    outPixFmt = defaultOutPixFMT
    enc = encode_method

    if transparent and encode_method not in ["prores_segment"]:
        enc = "prores_segment"
        inPixFmt = "rgba"
        outPixFmt = "yuva444p10le"
    elif grayscale:
        if bitDepth == "8bit":
            inPixFmt = "gray"
            outPixFmt = "yuv420p"
        else:
            inPixFmt = "gray16le"
            outPixFmt = "yuv444p10le"
    elif encode_method in ["x264_10bit", "x265_10bit", "x264_animation_10bit"]:
        if bitDepth == "8bit":
            inPixFmt = "rgb24"
            outPixFmt = "yuv420p10le"
        else:
            inPixFmt = "rgb48le"
            outPixFmt = "yuv420p10le"
    elif encode_method in ["nvenc_h264"]:
        if bitDepth == "8bit":
            inPixFmt = "rgb24"
            outPixFmt = "yuv420p"
        else:
            logWarning(
                "NVENC H.264 only supports 8-bit encoding. Falling back to 8-bit."
            )

            inPixFmt = "rgb48le"
            outPixFmt = "yuv420p"
    elif encode_method in [
        "nvenc_h265_10bit",
        "hevc_amf_10bit",
        "qsv_h265_10bit",
    ]:
        if bitDepth == "8bit":
            inPixFmt = "rgb24"
            outPixFmt = "p010le"
        else:
            inPixFmt = "rgb48le"
            outPixFmt = "p010le"
    elif encode_method in ["prores"]:
        if bitDepth == "8bit":
            inPixFmt = "rgb24"
            outPixFmt = "yuv444p10le"
        else:
            inPixFmt = "rgb48le"
            outPixFmt = "yuv444p10le"

    elif encode_method == "png":
        if bitDepth == "8bit":
            inPixFmt = "rgb24"
            outPixFmt = "rgb24"
        else:
            inPixFmt = "rgb48le"
            outPixFmt = "rgb48le"

    return inPixFmt, outPixFmt, enc


# Transfer characteristics FFmpeg's setparams accepts, so a probed value can be
# handed straight back. BT.2020 covers three very different curves -- PQ
# (smpte2084), HLG (arib-std-b67) and SDR (bt2020-10/-12) -- and stamping the
# wrong one tells the player to apply the wrong EOTF.
SETPARAMS_TRANSFERS = frozenset(
    {
        "smpte2084",
        "arib-std-b67",
        "bt2020-10",
        "bt2020-12",
        "bt709",
        "smpte428",
        "linear",
        "iec61966-2-1",
    }
)

# libavutil reports a BT.2020 matrix as bt2020nc/bt2020c and BT.2020 primaries
# as the bare "bt2020", so a detector keyed only on "bt2020" never matches a
# source that carries the matrix but no primaries -- it was converted to, and
# tagged as, BT.709.
BT2020_COLOR_VALUES = ("bt2020", "bt2020nc", "bt2020c")

BT709_FILTER = (
    "scale=in_range=pc:out_range=tv:out_color_matrix=bt709,"
    "format=yuv444p16le,"
    "setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709:range=tv"
)


def bt2020Filter(sourceTransfer: str = "unknown") -> str:
    """zscale conversion for a BT.2020 source, tagged with its own transfer.

    Only the matrix is converted here, so the transfer is whatever the source
    already had: copy it rather than guess. An unrecognized or missing value
    leaves `color_trc` off -- an untagged stream is recoverable, a mislabelled
    one is not.

    The working format is 16-bit like the BT.709 arm. It used to be `yuv420p`,
    which crushed a 16-bit HDR frame to 8-bit 4:2:0 *inside* the graph and then
    re-expanded it for the encoder: a 16-bit ramp came out as 220 distinct
    10-bit codes, every one a multiple of 4. Let `-pix_fmt` do the reduction,
    which is also where the dithering belongs.
    """
    setparams = "setparams=colorspace=bt2020nc:color_primaries=bt2020:range=tv"
    if sourceTransfer in SETPARAMS_TRANSFERS:
        setparams += f":color_trc={sourceTransfer}"
    return (
        f"zscale=matrix=bt2020nc:dither=error_diffusion,format=yuv444p16le,{setparams}"
    )


def colorSpaceFilter(probedMetadata: dict) -> str:
    """The colour-conversion filter for a source, from its probed metadata.

    Kept out of ffmpegSettings so it can be tested without torch/nelux/cv2 --
    that module's import chain is why CI skipped every colourspace test while a
    filter string that FFmpeg could not even parse shipped.
    """
    fields = ("ColorSpace", "PixelFormat", "ColorTRT")
    for field in fields:
        if probedMetadata.get(field, "unknown") in BT2020_COLOR_VALUES:
            return bt2020Filter(probedMetadata.get("ColorTRT", "unknown"))
    return BT709_FILTER
