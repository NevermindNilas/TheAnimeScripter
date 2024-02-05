import logging


def encodeSettings(encode_method: str, new_width: int, new_height: int, fps: float, output: str, ffmpeg_path: str, sharpen: bool, sharpen_sens: float, custom_encoder, grayscale: bool = False):
    """
    encode_method: str - The method to use for encoding the video. Options include "x264", "x264_animation", "nvenc_h264", etc.
    new_width: int - The width of the output video in pixels.
    new_height: int - The height of the output video in pixels.
    fps: float - The frames per second of the output video.
    output: str - The path to the output file.
    ffmpeg_path: str - The path to the FFmpeg executable.
    sharpen: bool - Whether to apply a sharpening filter to the video.
    sharpen_sens: float - The sensitivity of the sharpening filter.
    grayscale: bool - Whether to encode the video in grayscale.
    """
    if grayscale:
        pix_fmt = 'gray'
        output_pix_fmt = "yuv420p16le"
        if encode_method not in ["x264", "x264_animation", "av1"]:
            logging.info(
                "The selected encoder does not support yuv420p16le, switching to yuv420p10le.")
            output_pix_fmt = 'yuv420p10le'

    else:
        pix_fmt = 'rgb24'
        output_pix_fmt = 'yuv420p'

    command = [ffmpeg_path,
               '-y',
               '-f', 'rawvideo',
               '-vcodec', 'rawvideo',
               '-s', f'{new_width}x{new_height}',
               '-pix_fmt', f'{pix_fmt}',
               '-r', str(fps),
               '-i', '-',
               '-an',
               "-fps_mode", 'vfr'
               ]


    if custom_encoder == "":
        command.extend(match_encoder(encode_method))

        filters = []
        if sharpen:
            filters.append('cas={}'.format(sharpen_sens))
        if grayscale:
            filters.append('format=gray')

        if filters:
            custom_encoder_list.extend(['-vf', ','.join(filters)])

    else:
        custom_encoder_list = custom_encoder.split()

        if '-vf' in custom_encoder_list:
            vf_index = custom_encoder_list.index('-vf')

            if sharpen:
                custom_encoder_list[vf_index +
                                    1] += ',cas={}'.format(sharpen_sens)

            if grayscale:
                custom_encoder_list[vf_index + 1] += ',format=gray'
        else:
            filters = []
            if sharpen:
                filters.append('cas={}'.format(sharpen_sens))
            if grayscale:
                filters.append('format=gray')

            if filters:
                custom_encoder_list.extend(['-vf', ','.join(filters)])

        command.extend(custom_encoder_list)

    command.extend(['-pix_fmt', output_pix_fmt, output])

    logging.info(f"Encoding options: {' '.join(map(str, command))}")
    return command


def decodeSettings(input: str, inpoint: float, outpoint: float, dedup: bool, dedup_strenght: str, ffmpeg_path: str, resize: bool, width: int, height: int, resize_method: str):
    """
    input: str - The path to the input video file.
    inpoint: float - The start time of the segment to decode, in seconds.
    outpoint: float - The end time of the segment to decode, in seconds.
    dedup: bool - Whether to apply a deduplication filter to the video.
    dedup_strenght: float - The strength of the deduplication filter.
    ffmpeg_path: str - The path to the FFmpeg executable.
    resize: bool - Whether to resize the video.
    resize_method: str - The method to use for resizing the video. Options include: "fast_bilinear", "bilinear", "bicubic", "experimental", "neighbor", "area", "bicublin", "gauss", "sinc", "lanczos",
    "spline",
    """
    command = [
        ffmpeg_path,
    ]
    if outpoint != 0:
        command.extend(
            ["-ss", str(inpoint), "-to", str(outpoint)])

    command.extend([
        "-i", input,
    ])

    filters = []
    if dedup:
        filters.append(dedup_strenght)

    if resize:
        filters.append(
            f'scale={width}:{height}:flags={resize_method}')

    if filters:
        command.extend(["-vf", ','.join(filters)])

    command.extend([
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-v", "quiet",
        "-stats",
        "-",
    ])

    logging.info(f"Decoding options: {' '.join(map(str, command))}")
    return command


def get_dedup_strength(dedup_sens):
    hi = interpolate(dedup_sens, 0, 100, 64*2, 64*150)
    lo = interpolate(dedup_sens, 0, 100, 64*2, 64*30)
    frac = interpolate(dedup_sens, 0, 100, 0.1, 0.3)
    return f"mpdecimate=hi={hi}:lo={lo}:frac={frac},setpts=N/FRAME_RATE/TB"


def interpolate(x, x1, x2, y1, y2):
    return y1 + (x - x1) * (y2 - y1) / (x2 - x1)


def encodeYTDLP(input, output, ffmpeg_path, encode_method, custom_encoder):
    # This is for non rawvideo bytestreams, it's simpler to keep track this way
    # And have everything FFMPEG related organized in one file

    command = [ffmpeg_path, '-i', input]

    if custom_encoder == "":
        command.extend(match_encoder(encode_method))
    else:
        command.extend(custom_encoder.split())

    command.append(output)

    logging.info(f"Encoding options: {' '.join(command)}")

    return command


def match_encoder(encode_method: str):
    """
    encode_method: str - The method to use for encoding the video. Options include "x264", "x264_animation", "nvenc_h264", etc.
    """
    command = []
    # Settings inspiration from: https://www.xaymar.com/guides/obs/high-quality-recording/
    match encode_method:
        case "x264":
            command.extend(['-c:v', 'libx264', '-preset', 'veryfast', '-crf', '14'])
        case "x264_animation":
            command.extend(['-c:v', 'libx264', '-preset', 'veryfast', '-tune', 'animation', '-crf', '14'])
        case "x265":
            command.extend(['-c:v', 'libx265', '-preset', 'veryfast', '-crf', '14'])
        
        # Experimental, not tested    
        #case "x265_animation":
        #    command.extend(['-c:v', 'libx265', '-preset', 'veryfast', '-crf', '14', '-psy-rd', '1.0', '-psy-rdoq', '10.0'])
            
        case "nvenc_h264":
            command.extend(['-c:v', 'h264_nvenc', '-preset', 'p1', '-cq', '14'])
        case "nvenc_h265":
            command.extend(['-c:v', 'hevc_nvenc', '-preset', 'p1', '-cq', '14'])
        case "qsv_h264":
            command.extend(['-c:v', 'h264_qsv', '-preset', 'veryfast', '-global_quality', '14'])
        case "qsv_h265":
            command.extend(['-c:v', 'hevc_qsv', '-preset', 'veryfast', '-global_quality', '14'])
        case "nvenc_av1":
            command.extend(['-c:v', 'av1_nvenc', '-preset', 'p1', '-cq', '14'])
        case "av1":
            command.extend(['-c:v', 'libsvtav1', "-preset", "8", '-crf', '14'])
        case "h264_amf":
            command.extend(['-c:v', 'h264_amf', '-quality', 'speed', '-rc', 'cqp', '-qp', '14'])
        case "hevc_amf":
            command.extend(['-c:v', 'hevc_amf', '-quality', 'speed', '-rc', 'cqp', '-qp', '14'])
            
    return command
