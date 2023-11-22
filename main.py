import argparse
import os
import sys
from src.rife.rife import Rife
from src.cugan.cugan import Cugan
from src.dedup.dedup import Dedup
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
    
    if "rife" in model_type:
        output = f"{filename_without_ext}_{int(multi * fps)}fps.mp4"
        output = os.path.join(input_dir, output)
        
        UHD = True if w >= 2160 or h >= 2160 else False

        Rife(video_file, output, UHD, 1, multi, half, w, h, nt, fps, tot_frame)
        
    elif "cugan" in model_type:
        output = f"{filename_without_ext}_{str(multi)}.mp4"
        output = os.path.join(input_dir, output)
        
        Cugan(video_file, output, multi, half, kind_model, pro, w, h, nt, tot_frame)
        
    elif "dedup" in model_type:
        output = f"{filename_without_ext}_dedduped.mp4"
        output = os.path.join(input_dir, output)
        
        Dedup(video_file, output, kind_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Contact Sheet Generator")
    parser.add_argument("-video", type=str, help="", default="", action="store")
    parser.add_argument("-model_type",required=False, type=str, help="rife, cugan, dedup", default="rife", action="store")
    parser.add_argument("-half", type=bool, help="", default=True, action="store")
    parser.add_argument("-multi", type=int, help="", default=2, action="store")
    parser.add_argument("-kind_model", type=str, help="", default="shufflecugan", action="store")
    parser.add_argument("-pro", type=bool, help="", default=False, action="store")
    parser.add_argument("-nt", type=int, help="", default=1, action="store")
    args = parser.parse_args()
    
    if args.video is None:
        sys.exit("Please specify a video file")
    main(args.video, args.model_type, args.half, args.multi, args.kind_model, args.pro, args.nt)