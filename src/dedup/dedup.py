from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

import cv2
import os
import subprocess
import random

"""
Glorified deduplication method, for the best performance use FFMPEG.

SSIM is good but it generally takes into account 'human' perception of the image, so it might not be the best for this use case.

MSE is probably a good middlegorund, FFMPEG should still be used since it is a lot faster and doesn't run into race conditions, I hope.

TO:DO 
    - Only import the modules if they are needed, basically lazy loading
    
"""

class DedupSSIM():
    def __init__(self):
        """
        Processes two frames and returns True if they are similar enough to be considered duplicates
        """
        pass

    def run(self, I0, I1, dedup_sens):
        I0_gray = cv2.cvtColor(I0, cv2.COLOR_BGR2GRAY)
        I1_gray = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)

        ssim_val = ssim(I0_gray, I1_gray)

        dedup_sens = 1 - (dedup_sens / 100.0)

        if ssim_val > dedup_sens:
            return True
        else:
            return False


class DedupMSE():
    def __init__(self):
        """
        Processes two frames and returns True if they are similar enough to be considered duplicates
        """
        pass

    def run(self, I0, I1, dedup_sens):
        I0_gray = cv2.cvtColor(I0, cv2.COLOR_BGR2GRAY)
        I1_gray = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)

        mse_val = mean_squared_error(I0_gray, I1_gray)

        dedup_sens = 1 - (dedup_sens / 100.0)

        if mse_val < dedup_sens:
            return True
        else:
            return False


class DedupFFMPEG():
    def __init__(self, input, output, no_process):
        """
        Return full path to deduplicated video
        """
        self.input = input
        
        if no_process:
            self.output = output
        else:
            self.output = os.path.dirname(
                output) + os.path.basename(output).split(".")[0] + "_dedup" + ".mp4"

        dir_path = os.path.dirname(os.path.realpath(__file__))
        dir_path = os.path.dirname(dir_path)
        
        self.ffmpeg_path = os.path.join(dir_path, "ffmpeg", "ffmpeg.exe")
        print(self.ffmpeg_path)
        # Check if FFMPEG exists at that path
        if not os.path.exists(self.ffmpeg_path):
            print("Couldn't find FFMPEG, downloading it now")
            ffmpeg_bat_location = os.path.join(dir_path, "get_ffmpeg.bat")
            subprocess.call(ffmpeg_bat_location, shell=True)
            
    def run(self):
        ffmpeg_command = [self.ffmpeg_path, "-i", self.input, "-vf",
                          "mpdecimate=hi=64*24:lo=64*12:frac=0.1,setpts=N/FRAME_RATE/TB", "-an", "-y", self.output]
        subprocess.Popen(
            ffmpeg_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        return self.output

    
