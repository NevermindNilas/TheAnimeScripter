import argparse
import os
import sys
import warnings
from src.rife.rife import Rife
import cv2
import random

warnings.filterwarnings("ignore")

def main(video_file, multi, half, model_type):
    random_number = str(random.randint(0, 10000000))
    output_path = os.path.dirname(video_file)

    if "rife" in model_type:
        cap = cv2.VideoCapture(video_file)
        w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()    

        basename = os.path.basename(video_file)
        filename_without_ext = os.path.splitext(basename)[0]
        output = filename_without_ext + "_" + str(int(multi*fps)) + "_" + random_number + ".mp4"
        
        UHD = False
        if h >= 3840 or w >= 3840:
            UHD = True
        png, img = None, None
        Rife(video_file, output, img, UHD, 1, png, multi, half, w, h)
    elif "cugan" in model_type:
        pass
    elif "dedup" in model_type:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Contact Sheet Generator")
    parser.add_argument(
        "-model_type",
        required=False,
        type=str,
        help="rife",
        default="rife",
        action="store",
    )
    parser.add_argument("-video", type=str, help="", default="", action="store")
    parser.add_argument("-half", type=str, help="", default="True", action="store")
    parser.add_argument("-multi", type=int, help="", default=2, action="store")
    args = parser.parse_args()
    
    if args.video is None:
        sys.exit("Please specify a video file")
    main(args.video, args.multi, args.half, args.model_type)
