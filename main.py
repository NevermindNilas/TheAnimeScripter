import argparse
import os
import sys
import warnings
from src.rife.rife import Rife
from src.cugan.cugan import Cugan
import cv2
import random

warnings.filterwarnings("ignore")

def main(video_file, model_type, half, multi, kind_model, pro, nt):
    random_number = str(random.randint(0, 10000000))

    cap = cv2.VideoCapture(video_file)
    w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()    
    
    if "rife" in model_type:
        basename = os.path.basename(video_file)
        filename_without_ext = os.path.splitext(basename)[0]
        output = filename_without_ext + "_" + str(int(multi*fps)) + "_" + random_number + ".mp4"
        
        UHD = False
        if w >= 2160 or h >= 2160:
            UHD = True
        img = None, None
        Rife(video_file, output, UHD, 1, multi, half, w, h)
        
    elif "cugan" in model_type:
        basename = os.path.basename(video_file)
        filename_without_ext = os.path.splitext(basename)[0]
        output = filename_without_ext + "_" + str(multi) + "_" + random_number + ".mp4"
        
        Cugan(video_file, output, multi, half, kind_model, pro, w, h, nt)
        
    elif "dedup" in model_type:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Contact Sheet Generator")
    parser.add_argument("-video", type=str, help="", default="", action="store")
    parser.add_argument("-model_type",required=False,type=str,help="rife",default="rife",action="store")
    parser.add_argument("-half", type=bool, help="", default=True, action="store")
    parser.add_argument("-multi", type=int, help="", default=2, action="store")
    parser.add_argument("-kind_model", type=str, help="", default="no-denoise", action="store")
    parser.add_argument("-pro", type=bool, help="", default=False, action="store")
    parser.add_argument("-nt", type=int, help="", default=1, action="store")
    args = parser.parse_args()
    
    if args.video is None:
        sys.exit("Please specify a video file")
    main(args.video, args.model_type, args.half, args.multi, args.kind_model, args.pro, args.nt)
