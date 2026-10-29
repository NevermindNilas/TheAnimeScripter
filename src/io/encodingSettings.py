from src.infra.logAndPrint import logWarning


def matchNeluxEncoder(
    encode_method: str,
    bitDepth: str = "8bit",
    grayscale: bool = False,
    transparent: bool = False,
) -> dict | None:
    """
    nelux.VideoEncoder kwargs for an --encode_method, or None for an unknown
    one.

    Each method keeps the settings and pixel format it had when it ran on the
    FFmpeg-subprocess writer (removed): the per-arm `ffmpeg ...` comments
    record that command line, and the parity tests that held each mapping to
    it ran until the writer went. transparent=True turns any method into
    prores_segment (ProRes 4444 with alpha), and grayscale=True picks the
    pixel_format a gray pipe was encoded into -- see _grayscalePixelFormat.

    nelux (>= 0.18.0) maps the knobs per codec: an INTEGER preset translates
    per family (x264/x265 1=ultrafast..9=veryslow so 3=veryfast; libsvtav1
    preset = 13-n so 5 -> svt 8; NVENC -> p<n>), a STRING preset is forwarded
    to FFmpeg verbatim; cq maps to `-crf` (libx264/libx265/libsvtav1/libaom)
    or NVENC constqp `-qp` -- which is not NVENC's `-cq`, so those arms pass
    cq as an AVOption; `options` is an AVOption dict applied last, so it
    reaches anything the wrappers don't cover (tune, g, profile, and the
    libvpx-vp9 knobs: nelux >= 0.21.0 maps cq/preset there too, but the twin's
    preset is a libvpx no-op, so vp9 keeps its AVOptions). nelux >= 0.20.0 leaves
    GOP length and B-frames at each codec's defaults like the ffmpeg CLI
    (0.19.0 forced g=60/bf=2: -2.6 dB on x264). Measured against the FFmpeg
    twins through both writers (1080p, 600 frames, Y-PSNR vs the prores
    twin): within 0.2 dB and up to 18% smaller (x264/x265/vp9/av1/nvenc/qsv),
    prores within 0.03% in size. Scored against the input frames instead,
    nelux is level or ahead (vp9 +0.13 dB, lossless +0.3 dB): the twin's
    16-bit swscale pass dithers on the way down to 8 bits, and the prores
    reference carries that same dither.

    pixel_format mirrors getPixFMT's arm for the twin: at 8-bit input it is
    the 8-bit format (uint8 input into a 10-bit format is upconverted by
    nelux's libswscale pass just as FFmpeg upconverts rgb24). At
    --bit_depth 16bit (nelux >= 0.17.0 carries uint16 input at full precision
    into any >8-bit destination) an 8-bit pixel_format is promoted to its
    10-bit counterpart -- yuv444p10le like the FFmpeg twin's default 16-bit
    output, except libsvtav1 (yuv420p10le: it has no 444 10-bit mode and
    nelux would fall back to 8-bit yuv420p) and hevc/av1_nvenc (p010le, the
    CUDA 10-bit surface). h264_nvenc has no 10-bit path at all and stays
    yuv420p with a warning, mirroring getPixFMT, as does mjpeg (8-bit only);
    png and the QSV and AMF encoders take what the ffmpeg CLI swaps the twin's
    format for (_FFMPEG_AUTOSELECTED).
    """
    if transparent:
        encode_method = "prores_segment"

    mapping: dict | None = None
    match encode_method:
        case "x264":
            # ffmpeg `x264`: -preset veryfast -crf 15
            mapping = dict(codec="libx264", preset=3, cq=15, pixel_format="yuv420p")
        case "slow_x264":
            # ffmpeg `slow_x264`: -preset slow -crf 18 -tune animation -g 240
            mapping = dict(
                codec="libx264",
                preset="slow",
                cq=18,
                pixel_format="yuv420p",
                options={"tune": "animation", "g": "240"},
            )
        case "x264_10bit":
            # ffmpeg `x264_10bit`: -preset veryfast -crf 15 -profile:v high10
            mapping = dict(
                codec="libx264",
                preset=3,
                cq=15,
                pixel_format="yuv420p10le",
                options={"profile": "high10"},
            )
        case "x264_animation":
            # ffmpeg `x264_animation`: -preset veryfast -tune animation -crf 15
            mapping = dict(
                codec="libx264",
                preset=3,
                cq=15,
                pixel_format="yuv420p",
                options={"tune": "animation"},
            )
        case "x264_animation_10bit":
            mapping = dict(
                codec="libx264",
                preset=3,
                cq=15,
                pixel_format="yuv420p10le",
                options={"profile": "high10", "tune": "animation"},
            )
        case "x265":
            # ffmpeg `x265`: -preset veryfast -crf 15 (nelux itself passes
            # x265-params log-level=none, covering the 10bit arm's log-level=0)
            mapping = dict(codec="libx265", preset=3, cq=15, pixel_format="yuv420p")
        case "slow_x265":
            # ffmpeg `slow_x265`: -preset slow -crf 18
            # -level 5.1 -tune ssim -g 240. `-level` is dropped: FFmpeg's
            # libx265 wrapper never reads the generic level field, so the twin's
            # flag is already a silent no-op.
            mapping = dict(
                codec="libx265",
                preset="slow",
                cq=18,
                pixel_format="yuv420p",
                options={"tune": "ssim", "g": "240"},
            )
        case "x265_10bit":
            # ffmpeg `x265_10bit`: -preset veryfast -crf 15 -profile:v main10
            mapping = dict(
                codec="libx265",
                preset=3,
                cq=15,
                pixel_format="yuv420p10le",
                options={"profile": "main10"},
            )
        # NVENC's `-cq N` is VBR with a target quality, not nelux's cq=
        # (rc constqp, qp N): that wrote half the twin's size at 1-1.4 dB
        # lower PSNR. So cq goes through as an AVOption, with the twin's
        # bitrate -- `-b:v 0`, or, where the twin passes none, the NVENC
        # encoders' own 2M default, which nelux would otherwise replace with
        # its 4M one. 1080p, 300 frames: h264 p1 9.27 vs 9.45 MB at 53.28 vs
        # 53.48 dB; hevc p7 10.96 vs 11.15 MB at 54.43 vs 54.71 dB.
        case "nvenc_h264":
            # ffmpeg `nvenc_h264`: -preset p1 -cq 15
            mapping = dict(
                codec="h264_nvenc",
                preset=1,
                pixel_format="yuv420p",
                options={"cq": "15", "b": "2M"},
            )
        case "slow_nvenc_h264":
            # ffmpeg `slow_nvenc_h264`: -preset p7 -cq 15 -b:v 0 -g 240
            mapping = dict(
                codec="h264_nvenc",
                preset=7,
                pixel_format="yuv420p",
                options={"cq": "15", "b": "0", "g": "240"},
            )
        case "nvenc_h265":
            # ffmpeg `nvenc_h265`: -preset p1 -cq 15
            mapping = dict(
                codec="hevc_nvenc",
                preset=1,
                pixel_format="yuv420p",
                options={"cq": "15", "b": "2M"},
            )
        case "slow_nvenc_h265":
            # ffmpeg `slow_nvenc_h265`: -preset p7 -cq 12 -b:v 0 -g 240
            mapping = dict(
                codec="hevc_nvenc",
                preset=7,
                pixel_format="yuv420p",
                options={"cq": "12", "b": "0", "g": "240"},
            )
        case "nvenc_h265_10bit":
            # ffmpeg `nvenc_h265_10bit`: -preset p1 -cq 15 -profile:v main10
            mapping = dict(
                codec="hevc_nvenc",
                preset=1,
                pixel_format="p010le",
                options={"cq": "15", "b": "2M", "profile": "main10"},
            )
        case "nvenc_av1":
            # ffmpeg `nvenc_av1`: -preset p1 -cq 15
            mapping = dict(
                codec="av1_nvenc",
                preset=1,
                pixel_format="yuv420p",
                options={"cq": "15", "b": "2M"},
            )
        case "slow_nvenc_av1":
            # ffmpeg `slow_nvenc_av1`: -preset p7 -cq 15 -b:v 0 -g 240
            mapping = dict(
                codec="av1_nvenc",
                preset=7,
                pixel_format="yuv420p",
                options={"cq": "15", "b": "0", "g": "240"},
            )
        case "av1":
            # ffmpeg `av1`: libsvtav1 -preset 8 -crf 15 (int 5 -> svt 13-5=8)
            mapping = dict(codec="libsvtav1", preset=5, cq=15, pixel_format="yuv420p")
        case "slow_av1":
            # ffmpeg `slow_av1`: libsvtav1 -preset 4 -crf 27 -g 240 -b:v 0
            # (-row-mt is a libvpx knob the twin carries as a no-op; nelux
            # clears the bitrate itself whenever crf is set). The STRING "4"
            # skips nelux's 13-n int mapping and lands svt preset 4 verbatim.
            mapping = dict(
                codec="libsvtav1",
                preset="4",
                cq=27,
                pixel_format="yuv420p",
                options={"g": "240"},
            )
        case "vp9":
            # ffmpeg `vp9`: libvpx-vp9 -crf 15 (-preset is not a libvpx option
            # -- the twin's veryfast is a no-op). nelux >= 0.21.0 maps an int
            # preset to libvpx's cpu-used, which the twin never sets, so quality
            # goes through AVOptions. b=0 keeps libvpx in true CRF
            # (rc_end_usage VPX_Q), as the twin runs; nelux < 0.21.0 set its 4M
            # default here and turned -crf into constrained quality at 4 Mb/s.
            mapping = dict(
                codec="libvpx-vp9",
                pixel_format="yuv420p",
                options={"crf": "15", "b": "0"},
            )
        case "prores":
            # ffmpeg `prores`: prores_ks -profile:v 4 -qscale:v 15. The CLI's
            # -qscale:v is flags=+qscale with global_quality = 15 *
            # FF_QP2LAMBDA (118) = 1770; prores_ks divides it back out.
            # Verified within 0.03% of the twin's size at equal PSNR.
            mapping = dict(
                codec="prores_ks",
                pixel_format="yuv444p10le",
                options={"profile": "4", "flags": "+qscale", "global_quality": "1770"},
            )
        case "prores_segment":
            # ffmpeg `prores_segment`: the prores flags above. Fed RGBA
            # (--segment) it encodes yuva444p10le, ProRes 4444 with its alpha
            # plane, like getPixFMT's transparent arm. Picked as a plain method
            # it takes getPixFMT's default arm instead and has no alpha:
            # yuv420p, which the CLI swaps for yuv422p10le, or yuv444p10le at
            # 16-bit.
            if transparent:
                proresFormat = "yuva444p10le"
            elif bitDepth == "16bit":
                proresFormat = "yuv444p10le"
            else:
                proresFormat = _FFMPEG_AUTOSELECTED[("prores_ks", "yuv420p")]
            mapping = dict(
                codec="prores_ks",
                pixel_format=proresFormat,
                options={"profile": "4", "flags": "+qscale", "global_quality": "1770"},
            )
        case "gif":
            # ffmpeg `gif`: -qscale:v 1 -loop 0, and both halves are inert --
            # the gif encoder is palette+LZW and reads no qscale, and the gif
            # muxer's loop already defaults to 0 (infinite). Byte-comparable
            # output to the twin.
            mapping = dict(codec="gif")
        case "lossless":
            # ffmpeg `lossless`: libx264 -preset ultrafast -crf 0
            mapping = dict(
                codec="libx264", preset="ultrafast", cq=0, pixel_format="yuv420p"
            )
        case "lossless_nvenc":
            # ffmpeg `lossless_nvenc`: h264_nvenc -preset p1 -qp 0 -b:v 0
            # (nelux cq=0 -> rc constqp qp 0, the same lossless mode)
            mapping = dict(codec="h264_nvenc", preset=1, cq=0, pixel_format="yuv420p")
        case "qsv_h264":
            # ffmpeg `qsv_h264`: -preset veryfast -global_quality 15. nelux
            # wraps no QSV knob, so both go through verbatim (string preset,
            # AVOption global_quality). nv12 is what the ffmpeg CLI swaps the
            # twin's yuv420p for -- the QSV encoders take no planar 4:2:0.
            mapping = dict(
                codec="h264_qsv",
                preset="veryfast",
                pixel_format="nv12",
                options={"global_quality": "15"},
            )
        case "qsv_h265":
            # ffmpeg `qsv_h265`: -preset veryfast -global_quality 15
            mapping = dict(
                codec="hevc_qsv",
                preset="veryfast",
                pixel_format="nv12",
                options={"global_quality": "15"},
            )
        case "qsv_h265_10bit":
            # ffmpeg `qsv_h265_10bit`: -preset veryfast -global_quality 15
            # -profile:v main10, into getPixFMT's p010le
            mapping = dict(
                codec="hevc_qsv",
                preset="veryfast",
                pixel_format="p010le",
                options={"global_quality": "15", "profile": "main10"},
            )
        case "qsv_vp9":
            # ffmpeg `qsv_vp9`: -preset veryfast and nothing else, so the
            # encoder's own rate control. b=0 keeps it: nelux would otherwise
            # set its 4M default and cap the twin's ~7 Mb/s VBR at 4 Mb/s.
            mapping = dict(
                codec="vp9_qsv",
                preset="veryfast",
                pixel_format="nv12",
                options={"b": "0"},
            )
        # The AMF twins pass `-qp 15`, which no AMF encoder has (QP is per
        # frame type: qp_i/qp_p/qp_b), so the ffmpeg CLI only warns that it
        # was unused and the twin runs cqp at the driver's default QP. That
        # no-op is dropped here, as slow_x265's -level is. b=0 is the AMF
        # encoders' own default; nelux would otherwise set its 4M one.
        # Mapped from FFmpeg's amfenc sources and CLI negotiation only: no
        # AMD GPU here to run them on.
        case "h264_amf":
            # ffmpeg `h264_amf`: -quality speed -rc cqp -qp 15
            mapping = dict(
                codec="h264_amf",
                pixel_format="yuv420p",
                options={"quality": "speed", "rc": "cqp", "b": "0"},
            )
        case "hevc_amf":
            # ffmpeg `hevc_amf`: -quality speed -rc cqp -qp 15
            mapping = dict(
                codec="hevc_amf",
                pixel_format="yuv420p",
                options={"quality": "speed", "rc": "cqp", "b": "0"},
            )
        case "hevc_amf_10bit":
            # ffmpeg `hevc_amf_10bit`: -quality speed -rc cqp -qp 15
            # -profile:v main10, into getPixFMT's p010le
            mapping = dict(
                codec="hevc_amf",
                pixel_format="p010le",
                options={
                    "quality": "speed",
                    "rc": "cqp",
                    "b": "0",
                    "profile": "main10",
                },
            )
        case "png":
            # ffmpeg `png`: -q:v 0, which the png encoder never reads (it is
            # lossless and has no q-scale), into getPixFMT's rgb24. A numbered
            # output path keeps nelux on FFmpeg's own png defaults; only a
            # single still gets nelux's faster compression (still lossless).
            # color_range=pc rides on top of the BT.709 colour options, which
            # ask for tv: the twin's cICP chunk marks its RGB full range, and a
            # tv tag makes browsers that honour cICP stretch the levels.
            mapping = dict(
                codec="png", pixel_format="rgb24", options={"color_range": "pc"}
            )
        case "jpeg":
            # ffmpeg `jpeg`: mjpeg -qmin 1 -q:v 1 into yuvj444p. -q:v is
            # flags=+qscale with global_quality = 1 * FF_QP2LAMBDA (118), as
            # for prores; qmin goes first for the same reason as the twin's.
            # color_range=pc rides on top of the BT.709 colour options, which
            # ask for tv: yuvj444p is full range, as in the twin. nelux 0.19.0
            # already keeps yuvj formats full range; this keeps the colour
            # options from ever overriding it.
            mapping = dict(
                codec="mjpeg",
                pixel_format="yuvj444p",
                options={
                    "qmin": "1",
                    "flags": "+qscale",
                    "global_quality": "118",
                    "color_range": "pc",
                },
            )
        case _:
            mapping = None

    if mapping is not None and grayscale:
        # Replaces the 16-bit promotion below rather than following it, as
        # getPixFMT's grayscale arm comes before its per-method arms.
        mapping["pixel_format"] = _grayscalePixelFormat(str(mapping["codec"]), bitDepth)
    elif mapping is not None and bitDepth == "16bit":
        # Promote an 8-bit destination to 10-bit so a uint16 frame keeps its
        # precision instead of being narrowed back to 8-bit before swscale
        # ever sees it. Already-deep formats (the *_10bit methods, prores)
        # and pixfmt-less ones (gif, whose palette is 8-bit by nature) pass
        # through untouched.
        pixFmt = str(mapping.get("pixel_format", "") or "")
        isDeep = (
            "10le" in pixFmt
            or "12le" in pixFmt
            or "16le" in pixFmt
            or "p010" in pixFmt
            or "p016" in pixFmt
        )
        if pixFmt and not isDeep:
            codec = str(mapping.get("codec", "") or "")
            if codec == "h264_nvenc":
                # No 10-bit H.264 NVENC path; the uint16 input is narrowed
                # encoder-side, exactly as the FFmpeg twin's forced yuv420p.
                logWarning(
                    "NVENC H.264 only supports 8-bit encoding. Falling back to 8-bit."
                )
            elif codec == "mjpeg":
                # getPixFMT keeps jpeg at yuvj444p with the same warning.
                logWarning("JPEG only supports 8-bit encoding. Falling back to 8-bit.")
            elif codec == "png":
                # getPixFMT's 16-bit png output is rgb48le, which the png
                # encoder takes as rgb48be.
                mapping["pixel_format"] = _FFMPEG_AUTOSELECTED[("png", "rgb48le")]
            elif codec.endswith(("_qsv", "_amf")):
                # The twin's default yuv444p10le, swapped for what the QSV or
                # AMF encoder takes.
                mapping["pixel_format"] = _FFMPEG_AUTOSELECTED[(codec, "yuv444p10le")]
            elif codec in ("hevc_nvenc", "av1_nvenc"):
                mapping["pixel_format"] = "p010le"
            elif codec == "libsvtav1":
                # No yuv444p10le mode; asking for it falls back to 8-bit
                # yuv420p with a warning, which is worse than 420 10-bit.
                mapping["pixel_format"] = "yuv420p10le"
            else:
                # libx264 / libx265 / libvpx-vp9 (+ lossless libx264): the
                # FFmpeg twin's default 16-bit output is yuv444p10le.
                mapping["pixel_format"] = "yuv444p10le"

    return mapping


# What the ffmpeg CLI auto-selected when the FFmpeg writer asked an encoder for
# a format it does not take ("Incompatible pixel format ... auto-selecting
# format ..."), read off its log. nelux has no such fallback (or falls back to
# a different format), so the mapping names it up front.
_FFMPEG_AUTOSELECTED = {
    ("prores_ks", "yuv420p"): "yuv422p10le",
    ("libsvtav1", "yuv444p10le"): "yuv420p10le",
    ("gif", "yuv420p"): "rgb8",
    ("gif", "yuv444p10le"): "rgb8",
    # h264_qsv has no 10-bit or 4:4:4 input at all; hevc/vp9_qsv take the
    # twin's 16-bit 4:4:4 as packed xv30le (HEVC Range Extensions, VP9
    # profile 3).
    ("h264_qsv", "yuv420p"): "nv12",
    ("h264_qsv", "yuv444p10le"): "nv12",
    ("hevc_qsv", "yuv420p"): "nv12",
    ("hevc_qsv", "yuv444p10le"): "xv30le",
    ("vp9_qsv", "yuv420p"): "nv12",
    ("vp9_qsv", "yuv444p10le"): "xv30le",
    # AMF takes no planar 4:4:4: the CLI picks packed RGB, which the AMF
    # driver converts to YUV itself.
    ("h264_amf", "yuv444p10le"): "x2bgr10le",
    ("hevc_amf", "yuv444p10le"): "x2bgr10le",
    ("png", "rgb48le"): "rgb48be",
    ("png", "yuv420p"): "rgb24",
    ("png", "yuv444p10le"): "rgb48be",
    ("mjpeg", "yuv444p10le"): "yuvj444p",
}


def _grayscalePixelFormat(codec: str, bitDepth: str) -> str:
    """The pixel_format a gray (depth map) pipe is encoded into for `codec`.

    yuv420p (8-bit) or yuv444p10le (16-bit) whatever the method, as the FFmpeg
    writer encoded it. NVENC takes yuv420p (its nv12), or p010le for a 16-bit
    run on the HEVC/AV1 encoders.
    """
    eightBit = bitDepth == "8bit"
    if codec.endswith("_nvenc"):
        if not eightBit and codec in ("hevc_nvenc", "av1_nvenc"):
            return "p010le"
        return "yuv420p"
    pixFmt = "yuv420p" if eightBit else "yuv444p10le"
    return _FFMPEG_AUTOSELECTED.get((codec, pixFmt), pixFmt)


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


def colorSpaceOptions(probedMetadata: dict) -> dict:
    """The Nelux encoder's colour AVOptions for a source, from probed metadata.

    nelux (>= 0.18.0) reads these options back after avcodec_open2, so they
    drive the actual libswscale conversion matrix AND the container/bitstream
    tags together -- verified: the bt709 arm converts byte-identically to
    FFmpeg's `scale=out_color_matrix=bt709`, the bt2020nc arm within +-1 of
    zscale's float pipeline. Left unset, nelux converts and tags swscale's
    default bt470bg/601.

    The BT.2020 arm converts only the matrix and copies the source's own
    transfer -- and only when it is a value FFmpeg itself accepts;
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
