import os
import random


def outputNameGenerator(args):
    parts = [os.path.splitext(os.path.basename(args.input))[0] if args.input else "TAS"]

    if args.resize:
        parts.append(f"-Re{args.resize_factor}")

    if args.dedup:
        parts.append("-De")

    if args.interpolate:
        parts.append(f"-Int{args.interpolate_factor}")

    if args.upscale:
        parts.append(f"-Up{args.upscale_factor}")

    if args.sharpen:
        parts.append(f"-Sh{args.sharpen_sens}")

    if args.segment:
        parts.append("-Segment")

    if args.depth:
        parts.append("-Depth")

    if args.ytdlp:
        parts.append("-YTDLP")

    parts.append(f"-{random.randint(0, 1000)}")

    if args.ytdlp:
        extension = ".mp4"
    elif args.input:
        extension = os.path.splitext(args.input)[1]
    else:
        extension = ".mp4"
        
    outputName = "".join(parts) + extension
    
    return outputName
