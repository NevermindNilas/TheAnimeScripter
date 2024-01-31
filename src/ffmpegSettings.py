import picologging as logging


def encodeSettings(encode_method: str, new_width: int, new_height: int, fps: float, output: str, ffmpeg_path: str, sharpen: bool, sharpen_sens: float, grayscale: bool = False):
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
               "-vsync", 'vfr'
               ]

    # Settings from: https://www.xaymar.com/guides/obs/high-quality-recording/
    # Looking into adding a "Max compression" or a more custom way of setting the FFMPEG Params
    # My only concern is that it might be too complicated for the average user
    # Time will tell
    match encode_method:
        case "x264":
            command.extend(['-c:v', 'libx264',
                            '-preset', 'veryfast',
                            '-crf', '14',
                            ])

        case "x264_animation":
            command.extend(['-c:v', 'libx264',
                            '-preset', 'veryfast',
                            '-tune', 'animation',
                            '-crf', '14',
                            ])

        case "nvenc_h264":
            command.extend(['-c:v', 'h264_nvenc',
                            '-preset', 'p1',
                            '-cq', '14',
                            ])

        case "nvenc_h265":
            command.extend(['-c:v', 'hevc_nvenc',
                            '-preset', 'p1',
                            '-cq', '14',
                            ])

        case "qsv_h264":
            command.extend(['-c:v', 'h264_qsv',
                            '-preset', 'veryfast',
                            '-global_quality', '14',
                            ])

        case "qsv_h265":
            command.extend(['-c:v', 'hevc_qsv',
                            '-preset', 'veryfast',
                            '-global_quality', '14',
                            ])

        case "nvenc_av1":
            command.extend(['-c:v', 'av1_nvenc',
                            '-preset', 'p1',
                            '-cq', '14',
                            ])

        case "av1":
            command.extend(['-c:v', 'libsvtav1',
                            "-preset", "8",
                            '-crf', '14',
                            ])

        # I can't quite test these out since I do not have an AMD GPU but they are there just in case
        case "h264_amf":
            command.extend(['-c:v', 'h264_amf',
                            '-quality', 'speed',
                            '-rc', 'cqp',
                            '-qp', '14',
                            ])

        case "hevc_amf":
            command.extend(['-c:v', 'hevc_amf',
                            '-quality', 'speed',
                            '-rc', 'cqp',
                            '-qp', '14',
                            ])

    filters = []
    if sharpen:
        filters.append(f'cas={sharpen_sens}')
    if grayscale:
        # Will need to look into Dithering and see if it's worth adding
        filters.append('format=gray')

    if filters:
        command.extend(['-vf', ','.join(filters)])

    command.extend(['-pix_fmt', f'{output_pix_fmt}',
                    output])

    logging.info(f"Encoding options: {' '.join(command)}")
    return command


def decodeSettings(input: str, inpoint: float, outpoint: float, dedup: bool, dedup_strenght: str, ffmpeg_path: str, resize: bool, resize_factor: int, resize_method: str):
    """
    input: str - The path to the input video file.
    inpoint: float - The start time of the segment to decode, in seconds.
    outpoint: float - The end time of the segment to decode, in seconds.
    dedup: bool - Whether to apply a deduplication filter to the video.
    dedup_strenght: float - The strength of the deduplication filter.
    ffmpeg_path: str - The path to the FFmpeg executable.
    resize: bool - Whether to resize the video.
    resize_factor: int - The factor to resize the video by.
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
        if resize_factor < 0:
            resize_factor = 1 / abs(resize_factor)
        filters.append(
            f'scale=iw*{resize_factor}:ih*{resize_factor}:flags={resize_method}')

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


def encodeYTDLP(input, output, ffmpeg_path, encode_method):
    # This is for non rawvideo bytestreams, it's simpler to keep track this way
    # And have everything FFMPEG related organized in one file

    command = [ffmpeg_path, '-i', input]

    match encode_method:
        # I know that superfast isn't exactly the best preset for x264, but I fear that someone will try to convert a 4k 24 min video
        # On a i7 4770k and it will take 3 business days to finish
        case "x264":
            command.extend(
                ['-c:v', 'libx264', '-preset', 'superfast', '-crf', '14'])
        case "x264_animation":
            command.extend(
                ['-c:v', 'libx264', '-preset', 'superfast', '-tune', 'animation', '-crf', '14'])
        case "nvenc_h264":
            command.extend(
                ['-c:v', 'h264_nvenc', '-preset', 'p1', '-cq', '14'])
        case "nvenc_h265":
            command.extend(
                ['-c:v', 'hevc_nvenc', '-preset', 'p1', '-cq', '14'])
        case "qsv_h264":
            command.extend(
                ['-c:v', 'h264_qsv', '-preset', 'veryfast', '-global_quality', '14'])
        case "qsv_h265":
            command.extend(
                ['-c:v', 'hevc_qsv', '-preset', 'veryfast', '-global_quality', '14'])
        case "nvenc_av1":
            command.extend(
                ['-c:v', 'av1_nvenc', '-preset', 'p1', '-cq', '14'])
        case "av1":
            command.extend(
                ['-c:v', 'libsvtav1', '-preset', '8', '-crf', '14'])
        case "h264_amf":
            command.extend(
                ['-c:v', 'h264_amf', '-quality', 'speed', '-rc', 'cqp', '-qp', '14'])
        case "hevc_amf":
            command.extend(
                ['-c:v', 'hevc_amf', '-quality', 'speed', '-rc', 'cqp', '-qp', '14'])

    command.append(output)

    logging.info(f"Encoding options: {' '.join(command)}")

    return command
