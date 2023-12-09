import os
import subprocess
import numpy as np
import _thread
from torch.nn import functional as F
from skimage.metrics import structural_similarity as ssim
from queue import Queue

class Dedup():
    def __init__(self, video, output, kind_model):
        self.video = video
        self.output = output
        self.kind_model = kind_model
        
        if self.kind_model == "ffmpeg":
            self.ffmpeg()
        elif self.kind_model == "ssim":
            self.ssim()
        elif self.kind_model == "hash":
            self.hash()
        elif self.kind_model == "vmaf":
            self.vmaf()
        else:
            self.kind_model == "ffmpeg"
    
    def ffmpeg(self):
        print("The output is 1 second long only to avoid Variable Framerate issues if further processing is wanted.")
        subprocess.call(["static_ffmpeg", "-i", self.video, "-vf", "mpdecimate,setpts=N/FRAME_RATE/TB", "-v", "quiet", "-stats", "-y", self.output])

    def hash(self):
        pass

    def vmaf(self):
        pass 

    def ssim(self):
        pass


    