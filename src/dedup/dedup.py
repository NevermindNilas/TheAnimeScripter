import os
import subprocess
import static_ffmpeg
import skvideo.io
import numpy as np
import _thread
from torch.nn import functional as F
from skimage.metrics import structural_similarity as ssim
from queue import Queue

class Dedup():
    def __init__(self, video, output, kind_model, inputdict, outputdict):
        self.video = video
        self.output = output
        self.kind_model = kind_model
        self.inputdict = inputdict
        self.outputdict = outputdict
        
        if self.kind_model == "mpdecimate":
            self.mpdecimate()
        elif self.kind_model == "ssim":
            pass
            #self.ssim()
        elif self.kind_model == "hash":
            self.hash()
        elif self.kind_model == "vmaf":
            self.vmaf()
    
    def mpdecimate(self):
        subprocess.call(["static_ffmpeg", "-i", self.video, "-vf", "mpdecimate", "-loglevel", "error", "-stats", "-y", self.output])
    def hash(self):
        pass
    def vmaf(self):
        pass 
    '''def ssim(self):
        self.read_buffer = Queue(maxsize=500)
        _thread.start_new_thread(self._build_read_buffer, ())
        videogen = skvideo.io.vreader(self.video)
        video_out = skvideo.io.FFmpegWriter(self.output)
        prev_frame = None
        for frame in videogen:
            small = F.interpolate(frame, (32, 32), mode='bilinear', align_corners=False)
            if prev_frame is not None:
                ssim_value = ssim(prev_frame, small)
                if ssim_value < 0.9:
                    skvideo.io.vwrite(self.output, frame)
        
    def _build_read_buffer(self):
        try:
            for frame in self.videogen:
                self.read_buffer.put(frame)
        except:
            pass
        self.read_buffer.put(None)'''

    