import subprocess
import logging

from src.ffmpegSettings import matchEncoder


def dedupFFMPEG(
    input,
    output,
    filters,
    ffmpeg_path,
    encode_method="libx264",
    inpoint=None,
    outpoint=None,
):
    encode_options = matchEncoder(encode_method)

    filters = " ".join(filters)

    ffmpeg_command = [ffmpeg_path, "-hide_banner"]

    if outpoint != 0:
        ffmpeg_command += ["-ss", str(inpoint), "-to", str(outpoint)]

    ffmpeg_command += (
        ["-i", input, "-vf", filters, "-an"] + encode_options + [output, "-y"]
    )

    logging.info(f"Encoding options: {' '.join(ffmpeg_command)}")

    logging.info("\n============== FFMPEG Output Log ============")

    process = subprocess.Popen(
        ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )

    for line in iter(process.stdout.readline, b""):
        logging.info(line.decode().strip())
