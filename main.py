import argparse
import os
#os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import sys
from src.rife.rife import Rife
from src.cugan.cugan import Cugan
from src.dedup.dedup import Dedup
from src.swinir.swinir import Swin
#from src.segment.segment import Segment
from src.compact.compact import Compact
import cv2

#os.environ["CUDA_MODULE_LOADING"] = "LAZY"

def generate_output_filename(input_dir, filename_without_ext, extension):
    return os.path.join(input_dir, f"{filename_without_ext}_{extension}.mp4")

def main(video_file, model_type, half, multi, kind_model, pro, nt):
    video_file = os.path.normpath(video_file)

    cap = cv2.VideoCapture(video_file)
    w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    tot_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    basename = os.path.basename(video_file)
    filename_without_ext = os.path.splitext(basename)[0]
    input_dir = os.path.dirname(video_file)
    
    output = generate_output_filename(input_dir, filename_without_ext, model_type)
    model_type = model_type.lower()
    
    if h > 1080:
        print("If you are using CUDA, you might run into CUDA out of memory errors, if so, try to lower the multithread size, autotiling will be added in the future to avoid this issue")
    
    if model_type == "rife":

        UHD = True if w >= 2160 or h >= 2160 else False
        # UHD mode is auto decided by the script in order to avoid user errors.

        Rife(video_file, output, UHD, 1, multi, half, w, h, nt, fps, tot_frame, kind_model)
        
    elif model_type == "cugan" or model_type == "shufflecugan":
        
        if model_type == "cugan" and h > 1080:
            print("This may take a while, preferably use shufflecugan for over 1080p videos")
            
        if model_type == "shufflecugan" and multi != 2:
            print("The only scale that Shufflecugan works with is 2x, auto setting scale to 2")
            multi = 2
        
        if multi > 4:
            print("Cugan only supports up to 4x scaling, auto setting scale to 4")
            multi = 4
        
        Cugan(video_file, output, multi, half, kind_model, pro, w, h, nt, fps, tot_frame, model_type)
        
    elif model_type == "swinir":
        
        if multi != 2 and multi != 4:
            print("Swinir only supports 2x and 4x scaling, auto setting scale to 2")
            multi = 2
            
        Swin(video_file, output, model_type, multi, half, nt, w, h, fps, kind_model, tot_frame)
        
    elif model_type == "compact" or model_type == "ultracompact":
        
        if multi > 2:
            print("Compact only supports up to 2x scaling, auto setting scale to 2")
            multi = 2
        
        Compact(video_file, output, multi, half, w, h, nt, tot_frame, fps, model_type)
        
    elif model_type == "dedup":
        
        Dedup(video_file, output, kind_model)
    
    elif model_type == "segment":
        pass
        #Segment(video_file, output, kind_model, nt, half, w, h, tot_frame)
    else:
        sys.exit("Please select a valid model type", model_type, "was not found")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Contact Sheet Generator")
    parser.add_argument("-video", type=str, help="", default="", action="store")
    parser.add_argument("-model_type",required=False, type=str, help="rife, cugan, shufflecugan, swinir, dedup", default="rife", action="store")
    parser.add_argument("-half", type=bool, help="", default=True, action="store")
    parser.add_argument("-multi", type=int, help="", default=2, action="store")
    parser.add_argument("-kind_model", type=str, help="", default="", action="store")
    parser.add_argument("-pro", type=bool, help="", default=False, action="store")
    parser.add_argument("-nt", type=int, help="", default=1, action="store")
    args = parser.parse_args()
    
    if args.video is None:
        sys.exit("Please select a video file")
    main(args.video, args.model_type, args.half, args.multi, args.kind_model, args.pro, args.nt)