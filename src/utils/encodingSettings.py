from src.utils.logAndPrint import logAndPrint


def matchEncoder(encode_method: str):
    """
    encode_method: str - The method to use for encoding the video. Options include "x264", "x264_animation", "nvenc_h264", etc.
    """
    command = []
    match encode_method:
        case "x264":
            command.extend(["-c:v", "libx264", "-preset", "veryfast", "-crf", "15"])
        case "slow_x264":
            command.extend(
                [
                    "-c:v",
                    "libx264",
                    "-preset",
                    "slow",
                    "-crf",
                    "18",
                    "-profile:v",
                    "high",
                    "-level",
                    "4.1",
                    "-tune",
                    "animation",
                    "-x264-params",
                    "ref=4:bframes=8:b-adapt=2:direct=auto:me=umh:subme=10:merange=24:trellis=2:deblock=-1,-1:psy-rd=1.00,0.15:aq-strength=1.0:rc-lookahead=60",
                    "-bf",
                    "3",
                    "-g",
                    "250",
                    "-keyint_min",
                    "25",
                    "-sc_threshold",
                    "40",
                    "-qcomp",
                    "0.6",
                    "-qmin",
                    "10",
                    "-qmax",
                    "51",
                    "-maxrate",
                    "5000k",
                    "-bufsize",
                    "10000k",
                    "-movflags",
                    "+faststart",
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
                    "-x265-params",
                    "log-level=0",
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
                    "-x265-params",
                    "ref=6:bframes=8:b-adapt=2:direct=auto:me=umh:subme=7:merange=57:rd=6:psy-rd=2.0:aq-mode=3:aq-strength=0.8:rc-lookahead=60",
                    "-bf",
                    "4",
                    "-g",
                    "250",
                    "-keyint_min",
                    "25",
                    "-sc_threshold",
                    "40",
                    "-qcomp",
                    "0.7",
                    "-qmin",
                    "10",
                    "-qmax",
                    "51",
                    "-maxrate",
                    "5000k",
                    "-bufsize",
                    "10000k",
                    "-movflags",
                    "+faststart",
                    "-x265-params",
                    "log-level=0",
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
                    "-rc",
                    "vbr_hq",
                    "-b:v",
                    "0",
                    "-maxrate",
                    "5000k",
                    "-bufsize",
                    "10000k",
                    "-g",
                    "240",
                    "-keyint_min",
                    "23",
                    "-sc_threshold",
                    "40",
                    "-spatial_aq",
                    "1",
                    "-temporal_aq",
                    "1",
                    "-aq-strength",
                    "15",
                    "-rc-lookahead",
                    "60",
                    "-surfaces",
                    "64",
                    "-gpu",
                    "all",
                    "-movflags",
                    "+faststart",
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
                    "-rc",
                    "vbr_hq",
                    "-b:v",
                    "0",
                    "-maxrate",
                    "8000k",
                    "-bufsize",
                    "16000k",
                    "-g",
                    "240",
                    "-keyint_min",
                    "23",
                    "-sc_threshold",
                    "40",
                    "-spatial_aq",
                    "1",
                    "-temporal_aq",
                    "1",
                    "-aq-strength",
                    "15",
                    "-rc-lookahead",
                    "60",
                    "-surfaces",
                    "64",
                    "-gpu",
                    "all",
                    "-movflags",
                    "+faststart",
                    "-bf",
                    "2",
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
                    "-rc",
                    "vbr_hq",
                    "-b:v",
                    "0",
                    "-maxrate",
                    "5000k",
                    "-bufsize",
                    "10000k",
                    "-g",
                    "240",
                    "-keyint_min",
                    "23",
                    "-sc_threshold",
                    "40",
                    "-spatial_aq",
                    "1",
                    "-temporal_aq",
                    "1",
                    "-aq-strength",
                    "15",
                    "-rc-lookahead",
                    "60",
                    "-surfaces",
                    "64",
                    "-gpu",
                    "all",
                    "-movflags",
                    "+faststart",
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
                    "-svtav1-params",
                    "log-level=0:stat-report=0:verbosity=0:progress=0",
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
                    "30",
                    "-pix_fmt",
                    "yuv420p",
                    "-g",
                    "240",
                    "-keyint_min",
                    "23",
                    "-sc_threshold",
                    "40",
                    "-rc",
                    "vbr",
                    "-b:v",
                    "0",
                    "-maxrate",
                    "5000k",
                    "-bufsize",
                    "10000k",
                    "-tile-columns",
                    "2",
                    "-tile-rows",
                    "2",
                    "-row-mt",
                    "1",
                    "-movflags",
                    "+faststart",
                    "-svtav1-params",
                    "log-level=0:stat-report=0:verbosity=0",
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
        case "lossless_nvenc_h264":
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

    return command


def getPixFMT(encode_method, bitDepth, grayscale, transparent):
    """
    Return (inputPixFormat, outputPixFormat, encode_method) based on settings.
    """
    if bitDepth == "8bit":
        defaultInPixFMT = "yuv420p"
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
            inPixFmt = "yuv420p"
            outPixFmt = "yuv420p10le"
        else:
            inPixFmt = "rgb48le"
            outPixFmt = "yuv420p10le"
    elif encode_method in ["nvenc_h264"]:
        if bitDepth == "8bit":
            inPixFmt = "yuv420p"
            outPixFmt = "yuv420p"
        else:
            logAndPrint(
                "Warning: NVENC H.264 only supports 8-bit encoding. Falling back to 8-bit.",
                "yellow",
            )

            inPixFmt = "rgb48le"
            outPixFmt = "yuv420p"
    elif encode_method in [
        "nvenc_h265_10bit",
        "hevc_amf_10bit",
        "qsv_h265_10bit",
    ]:
        if bitDepth == "8bit":
            inPixFmt = "yuv420p"
            outPixFmt = "p010le"
        else:
            inPixFmt = "rgb48le"
            outPixFmt = "p010le"
    elif encode_method in ["prores"]:
        if bitDepth == "8bit":
            inPixFmt = "yuv420p"
            outPixFmt = "yuv444p10le"
        else:
            inPixFmt = "rgb48le"
            outPixFmt = "yuv444p10le"

    return inPixFmt, outPixFmt, enc
