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
        case "jpeg":
            # mjpeg's default qmin is 2, which silently clamps -q:v 1 back to
            # 2; lower qmin first so the best-quality setting actually applies.
            command.extend(
                [
                    "-c:v",
                    "mjpeg",
                    "-qmin",
                    "1",
                    "-q:v",
                    "1",
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


def matchNeluxEncoder(encode_method: str) -> dict | None:
    """
    nelux.VideoEncoder kwargs mirroring matchEncoder's FFmpeg arm one-for-one,
    or None for a method with no Nelux mapping.

    nelux (>= 0.18.0) maps the knobs per codec: an INTEGER preset translates
    per family (x264/x265 1=ultrafast..9=veryslow so 3=veryfast; libsvtav1
    preset = 13-n so 5 -> svt 8; NVENC -> p<n>), a STRING preset is forwarded
    to FFmpeg verbatim; cq maps to `-crf` (libx264/libx265/libsvtav1/libaom)
    or NVENC constqp `-qp`; `options` is an AVOption dict applied last, so it
    reaches anything the wrappers don't cover (tune, g, profile, and the
    libvpx-vp9 knobs, which nelux's cq does not map). Every mapping below is
    verified against its FFmpeg twin on real frames: PSNR within 0.1 dB,
    size within the encoder-invocation noise (prores within 0.03%).

    pixel_format mirrors getPixFMT's arm for the twin at 8-bit input --
    *_nelux methods only reach the Nelux writer at --bit_depth 8bit
    (src/cli/validator.py:_resolveNeluxEncoder downgrades 16bit), and uint8
    input into a 10-bit format is upconverted by nelux's libswscale pass just
    as FFmpeg upconverts rgb24.
    """
    match encode_method:
        case "x264_nelux":
            # ffmpeg `x264`: -preset veryfast -crf 15
            return dict(codec="libx264", preset=3, cq=15, pixel_format="yuv420p")
        case "slow_x264_nelux":
            # ffmpeg `slow_x264`: -preset slow -crf 18 -tune animation -g 240
            return dict(
                codec="libx264",
                preset="slow",
                cq=18,
                pixel_format="yuv420p",
                options={"tune": "animation", "g": "240"},
            )
        case "x264_10bit_nelux":
            # ffmpeg `x264_10bit`: -preset veryfast -crf 15 -profile:v high10
            return dict(
                codec="libx264",
                preset=3,
                cq=15,
                pixel_format="yuv420p10le",
                options={"profile": "high10"},
            )
        case "x264_animation_nelux":
            # ffmpeg `x264_animation`: -preset veryfast -tune animation -crf 15
            return dict(
                codec="libx264",
                preset=3,
                cq=15,
                pixel_format="yuv420p",
                options={"tune": "animation"},
            )
        case "x264_animation_10bit_nelux":
            return dict(
                codec="libx264",
                preset=3,
                cq=15,
                pixel_format="yuv420p10le",
                options={"profile": "high10", "tune": "animation"},
            )
        case "x265_nelux":
            # ffmpeg `x265`: -preset veryfast -crf 15 (nelux itself passes
            # x265-params log-level=none, covering the 10bit arm's log-level=0)
            return dict(codec="libx265", preset=3, cq=15, pixel_format="yuv420p")
        case "slow_x265_nelux":
            # ffmpeg `slow_x265`: -preset slow -crf 18 -profile:v main
            # -level 5.1 -tune ssim -g 240. `-level` is dropped: FFmpeg's
            # libx265 wrapper never reads the generic level field, so the twin's
            # flag is already a silent no-op.
            return dict(
                codec="libx265",
                preset="slow",
                cq=18,
                pixel_format="yuv420p",
                options={"tune": "ssim", "profile": "main", "g": "240"},
            )
        case "x265_10bit_nelux":
            # ffmpeg `x265_10bit`: -preset veryfast -crf 15 -profile:v main10
            return dict(
                codec="libx265",
                preset=3,
                cq=15,
                pixel_format="yuv420p10le",
                options={"profile": "main10"},
            )
        case "nvenc_h264_nelux":
            # ffmpeg `nvenc_h264`: -preset p1 -cq 15 (nelux cq -> constqp qp)
            return dict(codec="h264_nvenc", preset=1, cq=15, pixel_format="yuv420p")
        case "slow_nvenc_h264_nelux":
            # ffmpeg `slow_nvenc_h264`: -preset p7 -cq 15 -b:v 0 -g 240
            return dict(
                codec="h264_nvenc",
                preset=7,
                cq=15,
                pixel_format="yuv420p",
                options={"g": "240"},
            )
        case "nvenc_h265_nelux":
            return dict(codec="hevc_nvenc", preset=1, cq=15, pixel_format="yuv420p")
        case "slow_nvenc_h265_nelux":
            # ffmpeg `slow_nvenc_h265`: -preset p7 -cq 12 -b:v 0 -g 240
            return dict(
                codec="hevc_nvenc",
                preset=7,
                cq=12,
                pixel_format="yuv420p",
                options={"g": "240"},
            )
        case "nvenc_h265_10bit_nelux":
            # ffmpeg `nvenc_h265_10bit`: -preset p1 -cq 15 -profile:v main10
            return dict(
                codec="hevc_nvenc",
                preset=1,
                cq=15,
                pixel_format="p010le",
                options={"profile": "main10"},
            )
        case "nvenc_av1_nelux":
            return dict(codec="av1_nvenc", preset=1, cq=15, pixel_format="yuv420p")
        case "slow_nvenc_av1_nelux":
            # ffmpeg `slow_nvenc_av1`: -preset p7 -cq 15 -b:v 0 -g 240
            return dict(
                codec="av1_nvenc",
                preset=7,
                cq=15,
                pixel_format="yuv420p",
                options={"g": "240"},
            )
        case "av1_nelux":
            # ffmpeg `av1`: libsvtav1 -preset 8 -crf 15 (int 5 -> svt 13-5=8)
            return dict(codec="libsvtav1", preset=5, cq=15, pixel_format="yuv420p")
        case "slow_av1_nelux":
            # ffmpeg `slow_av1`: libsvtav1 -preset 4 -crf 27 -g 240 -b:v 0
            # (-row-mt is a libvpx knob the twin carries as a no-op; nelux
            # clears the bitrate itself whenever crf is set). The STRING "4"
            # skips nelux's 13-n int mapping and lands svt preset 4 verbatim.
            return dict(
                codec="libsvtav1",
                preset="4",
                cq=27,
                pixel_format="yuv420p",
                options={"g": "240"},
            )
        case "vp9_nelux":
            # ffmpeg `vp9`: libvpx-vp9 -crf 15 (-preset is not a libvpx option
            # -- the twin's veryfast is a no-op). nelux's cq/preset do not map
            # to libvpx, so quality goes through AVOptions; b=0 selects
            # CRF-only mode like the FFmpeg wrapper does when no bitrate is set.
            return dict(
                codec="libvpx-vp9",
                pixel_format="yuv420p",
                options={"crf": "15", "b": "0"},
            )
        case "prores_nelux":
            # ffmpeg `prores`: prores_ks -profile:v 4 -qscale:v 15. The CLI's
            # -qscale:v is flags=+qscale with global_quality = 15 *
            # FF_QP2LAMBDA (118) = 1770; prores_ks divides it back out.
            # Verified within 0.03% of the twin's size at equal PSNR.
            return dict(
                codec="prores_ks",
                pixel_format="yuv444p10le",
                options={"profile": "4", "flags": "+qscale", "global_quality": "1770"},
            )
        case "gif_nelux":
            # ffmpeg `gif`: -qscale:v 1 -loop 0, and both halves are inert --
            # the gif encoder is palette+LZW and reads no qscale, and the gif
            # muxer's loop already defaults to 0 (infinite). Byte-comparable
            # output to the twin.
            return dict(codec="gif")
        case "lossless_nelux":
            # ffmpeg `lossless`: libx264 -preset ultrafast -crf 0
            return dict(
                codec="libx264", preset="ultrafast", cq=0, pixel_format="yuv420p"
            )
        case "lossless_nvenc_nelux":
            # ffmpeg `lossless_nvenc`: h264_nvenc -preset p1 -qp 0 -b:v 0
            # (nelux cq=0 -> rc constqp qp 0, the same lossless mode)
            return dict(codec="h264_nvenc", preset=1, cq=0, pixel_format="yuv420p")
        case _:
            # png/jpeg (image sequences -- nelux writes one container),
            # qsv_*/*_amf (no hardware here to verify a mapping against), and
            # prores_segment (segment builds WriteBuffer directly) stay
            # FFmpeg-only.
            return None


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

    elif encode_method == "jpeg":
        # JPEG is 8-bit only; keep full-range 4:4:4 chroma so the sequence
        # stays as close to the source frames as the format allows.
        if bitDepth == "8bit":
            inPixFmt = "rgb24"
        else:
            logWarning("JPEG only supports 8-bit encoding. Falling back to 8-bit.")
            inPixFmt = "rgb48le"
        outPixFmt = "yuvj444p"

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


def colorSpaceOptions(probedMetadata: dict) -> dict:
    """The Nelux encoder's colour AVOptions for a source, from probed metadata.

    colorSpaceFilter's twin for NeluxWriteBuffer, detecting BT.2020 off the
    same fields. nelux (>= 0.18.0) reads these options back after
    avcodec_open2, so they drive the actual libswscale conversion matrix AND
    the container/bitstream tags together -- verified: the bt709 arm converts
    byte-identically to FFmpeg's `scale=out_color_matrix=bt709`, the bt2020nc
    arm within +-1 of zscale's float pipeline. Left unset, nelux converts and
    tags swscale's default bt470bg/601 -- self-consistent, but not what the
    FFmpeg writer produces for the same frames.

    Like bt2020Filter, the BT.2020 arm converts only the matrix and copies the
    source's own transfer -- and only when it is a value FFmpeg itself accepts;
    an unrecognized transfer leaves `color_trc` unset, because an untagged
    stream is recoverable and a mislabelled one is not.

    Every key/value is a generic AVCodecContext AVOption, valid for any codec;
    RGB-destination codecs (gif) ignore the matrix and keep only the tags.
    """
    fields = ("ColorSpace", "PixelFormat", "ColorTRT")
    for field in fields:
        if probedMetadata.get(field, "unknown") in BT2020_COLOR_VALUES:
            options = {
                "colorspace": "bt2020nc",
                "color_primaries": "bt2020",
                "color_range": "tv",
            }
            sourceTransfer = probedMetadata.get("ColorTRT", "unknown")
            if sourceTransfer in SETPARAMS_TRANSFERS:
                options["color_trc"] = sourceTransfer
            return options
    return {
        "colorspace": "bt709",
        "color_primaries": "bt709",
        "color_trc": "bt709",
        "color_range": "tv",
    }
