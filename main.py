import argparse
import os
import sys


def generate_output_filename(output, filename_without_ext):
    return os.path.join(output, f"{filename_without_ext}_output.mp4")

def main(video_file, model_type, half, multi, kind_model, pro, nt, output):
    from moviepy.editor import VideoFileClip
    video_file = os.path.normpath(video_file)
    
    # Maybe making a dictionary of the metada would be better
    clip = VideoFileClip(video_file)
    metadata = {
        "fps": clip.fps,
        "duration": clip.duration,
        "width": clip.w,
        "height": clip.h,
        "nframes": clip.reader.nframes,
    }
    clip.close()

    # Following Xaymar's guide: https://www.xaymar.com/guides/obs/high-quality-recording/avc/
    # These should be relatively good settings for most cases, feel free to change them as you see fit.
    if model_type == "swinir":
        # Swinir is for general purpose upscaling so we don't need animation tune
        ffmpeg_params = ["-c:v", "libx264", "-preset", "fast", "-crf", "15", "-movflags", "+faststart", "-y"]
    else:
        ffmpeg_params = ["-c:v", "libx264", "-preset", "fast", "-crf", "15", "-tune", "animation", "-movflags", "+faststart", "-y"]

    
    if output == "":
        basename = os.path.basename(video_file)
        filename_without_ext = os.path.splitext(basename)[0]
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, 'output')

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        if "m4v" in ffmpeg_params:
            m4v_index = ffmpeg_params.index("m4v")
            ffmpeg_params[m4v_index] = "mp4"

        output = generate_output_filename(output_path, filename_without_ext)
        
    elif os.path.isdir(output):
        basename = os.path.basename(video_file)
        filename_without_ext = os.path.splitext(basename)[0]
        output = generate_output_filename(output, filename_without_ext)
    
    if model_type == "rife":
        from src.rife.rife import Rife
                # UHD mode is auto decided by the script in order to avoid user errors.
        UHD = True if metadata["width"] >= 3840 or metadata["height"] >= 2160 else False
            
        Rife(video_file, output, UHD, 1, multi, half, metadata, kind_model, ffmpeg_params)

    elif model_type in ["cugan", "shufflecugan"]:
        from src.cugan.cugan import Cugan

        if model_type == "shufflecugan " and metadata["width"] < 1280 and metadata["height"] < 720:
            print("For resolutions under 1280x720p, please use cugan or compact model instead")
            sys.exit()
            
        if model_type == "shufflecugan" and multi != 2:
            print("The only scale that Shufflecugan works with is 2x, auto setting scale to 2")
            multi = 2
        
        if multi > 4:
            print("Cugan only supports up to 4x scaling, auto setting scale to 4")
            multi = 4
        
        Cugan(video_file, output, multi, half, kind_model, pro, metadata, nt, model_type, ffmpeg_params)
        
    elif model_type == "swinir":
        from src.swinir.swinir import Swin

        if multi != 2 and multi != 4:
            print("Swinir only supports 2x and 4x scaling, auto setting scale to 2")
            multi = 2
            
        Swin(video_file, output, model_type, multi, half, nt, metadata, kind_model, ffmpeg_params)
        
    elif model_type in ["compact", "ultracompact"]:
        from src.compact.compact import Compact
        if multi > 2:
            print(f"{model_type.upper()} only supports up to 2x scaling, auto setting scale to 2")
            multi = 2
        
        Compact(video_file, output, multi, half, nt, metadata, model_type, ffmpeg_params)
        
    elif model_type == "dedup":
        from src.dedup.dedup import Dedup
        
        Dedup(video_file, output, kind_model)
    
    elif model_type == "depth":
        from src.midas.depth import Depth

        Depth(video_file, output, half, metadata, nt, kind_model, ffmpeg_params)
            
    elif model_type == "segment":
        from src.segment.segment import Segment
        #Segment(video_file, output, nt, half, w, h, fps, tot_frame, kind_model, ffmpeg_params)
    else:
        sys.exit("Please select a valid model type ", model_type, " was not found")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Contact Sheet Generator")
    parser.add_argument("-video", type=str, help="", default="", action="store")
    parser.add_argument("-model_type", type=str, help="rife, cugan, shufflecugan, swinir, dedup", default="rife", action="store")
    parser.add_argument("-half", type=bool, help="", default=True, action="store")
    parser.add_argument("-multi", type=int, help="", default=2, action="store")
    parser.add_argument("-kind_model", type=str, help="", default="", action="store")
    parser.add_argument("-pro", type=bool, help="", default=False, action="store")
    parser.add_argument("-nt", type=int, help="", default=1, action="store")
    parser.add_argument("-output", type=str, help="can be path to folder or filename only", default="", action="store")
    #parser.add_argument("-chain", type=str, help="For chaining models", default=False, action="store")
    args = parser.parse_args()
    
    if args.video is None:
        sys.exit("Please select a video file")
        
    main(args.video, args.model_type, args.half, args.multi, args.kind_model, args.pro, args.nt, args.output)
