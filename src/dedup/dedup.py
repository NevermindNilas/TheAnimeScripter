import subprocess
import logging

from src.ffmpegSettings import match_encoder

def dedup_ffmpeg(input, output, filters, ffmpeg_path, encode_method="libx264"):
    encode_options = match_encoder(encode_method)
    ffmpeg_command = [ffmpeg_path, "-hide_banner", "-i", input, "-vf", filters,
                      "-an"] + encode_options + [output]

    logging.info(f"Encoding options: {' '.join(ffmpeg_command)}")
    
    logging.info(
        "\n============== FFMPEG Output Log ============")
    
    process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    for line in iter(process.stdout.readline, b''):
        logging.info(line.decode().strip())

def trim_input_dedup(input, output, inpoint, outpoint, filters, ffmpeg_path, encode_method="libx264"):
    encode_options = match_encoder(encode_method)
    ffmpeg_command = [ffmpeg_path, "-hide_banner", "-ss", str(inpoint), "-to", str(outpoint), "-i", input, "-vf", filters,
               "-an"] + encode_options + [output, "-v", "quiet", "-stats"]

    logging.info(f"Encoding options: {' '.join(ffmpeg_command)}")
    
    logging.info(
        "\n============== FFMPEG Output Log ============")
    
    process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    for line in iter(process.stdout.readline, b''):
        logging.info(line.decode().strip())