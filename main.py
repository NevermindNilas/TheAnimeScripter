import argparse
import os
import sys
from src.rife.rife import Rife
from src.cugan.cugan import Cugan
from src.dedup.dedup import Dedup
from src.swinir.swinir import Swin
from src.segment.segment import Segment
import cv2   

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
    
    model_type = model_type.lower()
    
    if model_type == "rife":
        output = f"{filename_without_ext}_{int(multi * fps)}fps.mp4"
        output = os.path.join(input_dir, output)
        
        UHD = True if w >= 2160 or h >= 2160 else False
        # UHD mode is auto decided by the script in order to avoid user errors.

        Rife(video_file, output, UHD, 1, multi, half, w, h, nt, fps, tot_frame)
        
    elif model_type == "cugan" or model_type == "shufflecugan":
        
        if model_type == "shufflecugan" and multi != 2:
            print("The only scale that Shufflecugan works with is 2x, setting scale to 2")
            multi = 2
        
        if multi > 4:
            print("Cugan only supports up to 4x scaling, setting scale to 4")
            multi = 4
            
        output = f"{filename_without_ext}_{str(multi)}.mp4"
        output = os.path.join(input_dir, output)
        
        Cugan(video_file, output, multi, half, kind_model, pro, w, h, nt, tot_frame, model_type)
        
    elif model_type == "dedup":
        output = f"{filename_without_ext}_dedduped.mp4"
        output = os.path.join(input_dir, output)
        
        Dedup(video_file, output, kind_model)
    
    elif model_type == "swinir":
        output = f"{filename_without_ext}_{str(multi)}.mp4"
        output = os.path.join(input_dir, output)
        
        Swin(video_file, output, model_type, multi, half, nt, kind_model, tot_frame)
        
    elif model_type == "segment":
        output = f"{filename_without_ext}_segmented.mp4"
        output = os.path.join(input_dir, output)
        
        Segment(video_file, output, kind_model, nt, half, w, h, tot_frame)
    else:
        sys.exit("Please specify a valid model type", model_type, "was not found")

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
        sys.exit("Please specify a video file")
    main(args.video, args.model_type, args.half, args.multi, args.kind_model, args.pro, args.nt)