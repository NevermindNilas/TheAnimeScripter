import os
import random

def outputNameGenerator(args):
    arg_map = {
        'resize': f"-Re{args.resize_factor}",
        'dedup': "-De",
        'interpolate': f"-Int{args.interpolate_factor}",
        'upscale': f"-Up{args.upscale_factor}",
        'sharpen': f"-Sh{args.sharpen_sens}",
        'segment': "-Segment",
        'depth': "-Depth",
        'ytdlp': "-YTDLP"
    }

    parts = [os.path.splitext(os.path.basename(args.input))[0] if args.input else "TAS"]

    for arg, format_str in arg_map.items():
        if getattr(args, arg, None):
            parts.append(format_str)

    parts.append(f"-{random.randint(0, 1000)}")

    if args.ytdlp:
        extension = ".mp4"
    elif args.input:
        extension = os.path.splitext(args.input)[1]
    else:
        extension = ".mp4"
        
    outputName = "".join(parts) + extension
    
    return outputName