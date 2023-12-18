import argparse

"""
AE doesn't really like cooperating with me and systemCall commands, so I offload as much of the logic off of it as possible.
"""

def main(args):
    import os
    import subprocess
    abs_path = os.path.abspath(__file__)
    dir_path = os.path.dirname(abs_path)

    ffmpeg_path = os.path.join(dir_path, "ffmpeg", "ffmpeg.exe")
    command = f'"{ffmpeg_path}" -i "{args.input}" -ss {args.start} -to {args.to} "{args.output}" -v quiet -stats -y'
    subprocess.run(command, shell=True)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-i", "--input", help="Input file")
    argparser.add_argument("-o", "--output", help="Output file")
    argparser.add_argument("-ss", "--start", help="Start frame")
    argparser.add_argument("-to", "--to", help="End frame")

    args = argparser.parse_args()
    main(args)
