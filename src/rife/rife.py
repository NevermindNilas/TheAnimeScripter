import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import warnings
import _thread
import skvideo.io
from queue import Queue
from .pytorch_msssim import ssim_matlab
import time

warnings.filterwarnings("ignore")
# Turning rife into a python class
'''
Credit: https://github.com/hzwer/Practical-RIFE/blob/main/inference_video.py
'''

@torch.inference_mode()

class Rife():
    def __init__(self, video, output, UHD, scale, multi, half, w, h, nt, inputdict, outputdict, fps, tot_frame):
        self.video = video
        self.output = output
        self.half = half
        self.UHD = UHD
        self.scale = scale
        self.multi = multi
        self.modelDir = 'src/rife'
        self.w = w
        self.h = h
        self.nt = nt
        self.inputdict = inputdict
        self.outputdict = outputdict
        self.fps = fps
        self.tot_frame = tot_frame
        
        self._initialize()
    
    def _initialize(self):
        if self.UHD == True and self.scale == 1.0:
            self.scale = 0.5
        assert self.scale in [0.25, 0.5, 1.0, 2.0, 4.0]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            if self.half:
                torch.set_default_tensor_type(torch.cuda.HalfTensor)
        try:
            from src.rife.RIFE_HDv3 import Model
        except:
            print("Please download our model from the model list")
        self.model = Model()
        if not hasattr(self.model, 'version'):
            self.model.version = 0
        self.model.load_model(self.modelDir, -1)
        self.model.eval()
        self.model.device()

        self.videogen = skvideo.io.vreader(self.video)
        self.lastframe = next(self.videogen)
        self.vid_out = skvideo.io.FFmpegWriter(self.output, self.inputdict, self.outputdict)
        self.h, self.w, _ = self.lastframe.shape
        self.padding = (0, ((self.w - 1) // 128 + 1) * 128 - self.w, 0, ((self.h - 1) // 128 + 1) * 128 - self.h)
        
        self.pbar = tqdm(total=self.tot_frame)
        self.write_buffer = Queue(maxsize=500)
        self.read_buffer = Queue(maxsize=500)
        
        _thread.start_new_thread(self._build_read_buffer, ())
        _thread.start_new_thread(self._clear_write_buffer, ())

        self.process_video()
        
    def _clear_write_buffer(self):
        while True:
            frame = self.write_buffer.get()
            if frame is None:
                break
            self.pbar.update(1)
            self.vid_out.writeFrame(frame)
            
        self.vid_out.close()
        self.pbar.close()
        
    def _build_read_buffer(self):
        try:
            for frame in self.videogen:
                self.read_buffer.put(frame)
        except:
            pass
        for _ in range(self.nt):
            self.read_buffer.put(None)
    
    def make_inference(self, I0, I1, n):
        if self.model.version >= 3.9:
            res = []
            for i in range(n):
                res.append(self.model.inference(I0, I1, (i + 1) * 1. / (n + 1), self.scale))
            return res
        else:
            middle = self.model.inference(I0, I1, self.scale)
            if n == 1:
                return [middle]
            first_half = self.make_inference(I0, middle, n=n // 2)
            second_half = self.make_inference(middle, I1, n=n // 2)
            if n % 2:
                return [*first_half, middle, *second_half]
            else:
                return [*first_half, *second_half]
    
    def _pad_image(self, img):
        if self.half:
            return F.pad(img, self.padding).half()
        else:
            return F.pad(img, self.padding)
        
    def process_video(self):
        I1 = torch.from_numpy(np.transpose(self.lastframe, (2, 0, 1))).to(self.device, non_blocking=True).unsqueeze(
            0).float() / 255.
        I1 = self._pad_image(I1)
        self.temp = None  # save lastframe when processing static frame
        while True:
            if self.temp is not None:
                frame = self.temp
                self.temp = None
            else:
                frame = self.read_buffer.get()

            if frame is None:
                break

            I0 = I1
            I1 = torch.from_numpy(np.transpose(frame, (2, 0, 1))).to(self.device, non_blocking=True).unsqueeze(
                0).float() / 255.
            I1 = self._pad_image(I1)
            I0_small = F.interpolate(I0, (32, 32), mode='bilinear', align_corners=False)
            I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
            ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])

            break_flag = False
            if ssim > 0.996:
                frame = self.read_buffer.get()  # read a new frame
                if frame is None:
                    break_flag = True
                    frame = self.lastframe
                else:
                    self.temp = frame

                I1 = torch.from_numpy(np.transpose(frame, (2, 0, 1))).to(self.device, non_blocking=True).unsqueeze(
                    0).float() / 255.
                I1 = self._pad_image(I1)
                I1 = self.model.inference(I0, I1, self.scale)
                I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
                ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
                frame = (I1[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:self.h, :self.w]

            if ssim < 0.2:
                output = []
                for i in range(self.multi - 1):
                    output.append(I0)
            else:
                output = self.make_inference(I0, I1, self.multi - 1)

            self.write_buffer.put(self.lastframe)
            for mid in output:
                mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
                self.write_buffer.put(mid[:self.h, :self.w])

            self.pbar.update(1)
            self.lastframe = frame
            if break_flag:
                break

            self.write_buffer.put(self.lastframe)

        while not self.write_buffer.empty():
            time.sleep(0.1)
        self.pbar.update(1)
        self.pbar.close()
        if not self.vid_out is None:
            self.vid_out.release()