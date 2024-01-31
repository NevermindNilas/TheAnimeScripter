import os
import random

def outputNameGenerator(args, main_path):
    
    if not os.path.exists(os.path.join(main_path, "output")):
        os.makedirs(os.path.join(main_path, "output"), exist_ok=True)
    
    output_name = os.path.splitext(os.path.basename(args.input))[0]
    if args.dedup:
        output_name += "-De"

    if args.interpolate:
        output_name += "-Int" + str(args.interpolate_factor)

    if args.upscale:
        output_name += "-Up" + str(args.upscale_factor)

    if args.sharpen:
        output_name += "-Sh" + str(args.sharpen_sens)

    if args.segment:
        output_name += "-Segment"

    if args.depth:
        output_name += "-Depth"

    if args.motion_blur:
        output_name += "-MotionBlur"
    
    if not args.ytdlp == "":
        output_name += "-YTDLP"

    random_number = random.randint(0, 1000)
    output_name = os.path.join(main_path, "output", output_name + "-" + str(random_number) + ".mp4")
    
    return output_name