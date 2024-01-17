import subprocess

def dedup_ffmpeg(input, output, mpdecimate_params, ffmpeg_path, encode_method="libx264"):
    encode_options = handle_encoder(encode_method)
    ffmpeg_command = [ffmpeg_path, "-i", input, "-vf", mpdecimate_params, "-an"] + flatten_dict(encode_options) + ["-y", output]
    subprocess.Popen(ffmpeg_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def trim_input_dedup(input, output, inpoint, outpoint, mpdecimate_params, ffmpeg_path, encode_method="libx264"):
    encode_options = handle_encoder(encode_method)
    command = [ffmpeg_path, "-ss", str(inpoint), "-to", str(outpoint), "-i", input, "-vf", mpdecimate_params, "-an"] + flatten_dict(encode_options) + ["-y", output, "-v", "quiet", "-stats"]
    subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def handle_encoder(encode_method):
    if encode_method == "x264":
        return {'-c:v': 'libx264', '-preset': 'veryfast', '-crf': '14'}
    elif encode_method == "x264_animation":
        return {'-c:v': 'libx264', '-preset': 'veryfast', '-tune': 'animation', '-crf': '14'}
    elif encode_method == "nvenc_h264":
        return {'-c:v': 'h264_nvenc', '-preset': 'p1', '-cq': '14'}
    elif encode_method == "nvenc_h265":
        return {'-c:v': 'hevc_nvenc', '-preset': 'p1', '-cq': '14'}
    elif encode_method == "qsv_h264":
        return {'-c:v': 'h264_qsv', '-preset': 'veryfast', '-global_quality': '14'}
    elif encode_method == "qsv_h265":
        return {'-c:v': 'hevc_qsv', '-preset': 'veryfast', '-global_quality': '14'}
    elif encode_method == "nvenc_av1":
        return {'-c:v': 'av1_nvenc', '-preset': 'p1', '-cq': '14'}
    elif encode_method == "av1":
        return {'-c:v': 'libsvtav1', "-preset": "8", '-crf': '14'}
    else:
        return {'-c:v': 'libx264', '-preset': 'veryfast', '-crf': '14'}

def flatten_dict(d):
    return [item for sublist in d.items() for item in sublist]