import logging

def encodeSettings(encode_method: str, new_width: int, new_height: int, fps: float, output: str, ffmpeg_path: str, sharpen: bool, sharpen_sens: float):
    """
    encode_method: str - The method to use for encoding the video. Options include "x264", "x264_animation", "nvenc_h264", etc.
    new_width: int - The width of the output video in pixels.
    new_height: int - The height of the output video in pixels.
    fps: float - The frames per second of the output video.
    output: str - The path to the output file.
    ffmpeg_path: str - The path to the FFmpeg executable.
    sharpen: bool - Whether to apply a sharpening filter to the video.
    sharpen_sens: float - The sensitivity of the sharpening filter.
    """
    
    command = [ffmpeg_path,
               '-y',
               '-f', 'rawvideo',
               '-vcodec', 'rawvideo',
               '-s', f'{new_width}x{new_height}',
               '-pix_fmt', 'rgb24',
               '-r', str(fps),
               '-i', '-',
               '-an',
               ]

    # Settings from: https://www.xaymar.com/guides/obs/high-quality-recording/
    # TO-DO: Add AMF Support
    match encode_method:
        case "x264":
            command.extend(['-c:v', 'libx264',
                            '-preset', 'veryfast',
                            '-crf', '14',
                            output])

        case "x264_animation":
            command.extend(['-c:v', 'libx264',
                            '-preset', 'veryfast',
                            '-tune', 'animation',
                            '-crf', '14',
                            output])

        case "nvenc_h264":
            command.extend(['-c:v', 'h264_nvenc',
                            '-preset', 'p1',
                            '-cq', '14',
                            output])

        case "nvenc_h265":
            command.extend(['-c:v', 'hevc_nvenc',
                            '-preset', 'p1',
                            '-cq', '14',
                            output])

        case "qsv_h264":
            command.extend(['-c:v', 'h264_qsv',
                            '-preset', 'veryfast',
                            '-global_quality', '14',
                            output])

        case "qsv_h265":
            command.extend(['-c:v', 'hevc_qsv',
                            '-preset', 'veryfast',
                            '-global_quality', '14',
                            output])

        case "nvenc_av1":
            command.extend(['-c:v', 'av1_nvenc',
                            '-preset', 'p1',
                            '-cq', '14',
                            output])

        case "av1":
            command.extend(['-c:v', 'libsvtav1',
                            "-preset", "8",
                            '-crf', '14',
                            output])

    if sharpen:
        command.insert(-1, '-vf')
        command.insert(-1, f'cas={sharpen_sens}')

    logging.info(f"Encoding options: {' '.join(command)}")
    return command


def decodeSettings(input: str, inpoint: float, outpoint: float, dedup: bool, dedup_strenght: str, ffmpeg_path: str):
    """
    input: str - The path to the input video file.
    inpoint: float - The start time of the segment to decode, in seconds.
    outpoint: float - The end time of the segment to decode, in seconds.
    dedup: bool - Whether to apply a deduplication filter to the video.
    dedup_strenght: float - The strength of the deduplication filter.
    ffmpeg_path: str - The path to the FFmpeg executable.
    """
    command = [
        ffmpeg_path,
        "-i", str(input),
    ]
    if outpoint != 0:
        command.extend(
            ["-ss", str(inpoint), "-to", str(outpoint)])

    if dedup:
        command.extend(
            ["-vf", dedup_strenght])

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
