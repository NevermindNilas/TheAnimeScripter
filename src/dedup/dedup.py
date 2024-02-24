import subprocess
import logging
import sys

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
        ["-i", input, "-vf", filters, "-an"]
        + encode_options
        + [output, "-y", "-v", "error", "-stats"]
    )
    logging.info(f"Encoding options: {' '.join(ffmpeg_command)}")

    subprocess.run(
        ffmpeg_command, stdout=sys.stdout, stderr=sys.stderr, universal_newlines=True
    )

