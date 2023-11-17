import cv2
import os
from threading import Thread
import torch
from torch.nn import functional as F
import time
from collections import deque
import numpy as np
from src.rife.pytorch_msssim import ssim_matlab
from tqdm import tqdm
import _thread
from queue import Queue, Empty

def process_video_rife(video_file, output_path, model, scale, device, half):
    video_stream = VideoDecodeStream(video_file)
    video_stream.start()
    prev_frame = video_stream.read()
    write_buffer = Queue(maxsize=500)
    _thread.start_new_thread(clear_write_buffer, (output_path, write_buffer))
    lastframe = prev_frame
    h,w,_= video_stream.get_shape()
    tmp = max(128, int(128 / scale))
    ph = ((h - 1) // tmp + 1) * tmp
    pw = ((w - 1) // tmp + 1) * tmp
    padding = (0, pw - w, 0, ph - h)
    multi = scale
    progessbar = tqdm(total=video_stream.get_frame_count(), unit='frames')
    I1 = torch.from_numpy(np.transpose(lastframe, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
    I1 = pad_image(I1, half, padding)
    temp = None
    while True:
        if temp is not None:
                frame = temp
                temp = None
        else:
            frame = video_stream.read()
        if frame is None:
            break
        I0 = I1
        I1 = torch.from_numpy(np.transpose(frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
        I1 = pad_image(I1, half, padding)
        break_flag = False
        output = make_inference(I0, I1, scale-1, scale, model)
        
        write_buffer.put(lastframe)
        for mid in output:
            mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
            write_buffer.put(mid[:h, :w])
                
        progessbar.update(1)
        lastframe = frame
        if break_flag:
            video_stream.stop()
            break

def make_inference(I0, I1, n, scale, model):
    if model.version >= 3.9:
        res = []
        for i in range(n):
            res.append(model.inference(I0, I1, (i+1) * 1. / (n+1), scale))
        return res
    else:
        middle = model.inference(I0, I1, scale)
        if n == 1:
            return [middle]
        first_half = make_inference(I0, middle, n=n//2)
        second_half = make_inference(middle, I1, n=n//2)
        if n%2:
            return [*first_half, middle, *second_half]
        else:
            return [*first_half, *second_half]

def pad_image(img, half, padding):
    if(half):
        return F.pad(img, padding).half()
    else:
        return F.pad(img, padding)

def clear_write_buffer(output_path, write_buffer):
    cnt = 0
    time.sleep(5)
    while True:
        item = write_buffer.get()
        if item is None:
            break
        cv2.imwrite(os.path.join(output_path, f"{cnt:0>5d}.png"), item)
        cnt += 1
            
class VideoDecodeStream:
    def __init__(self, video_file):
        self.vcap = cv2.VideoCapture(video_file)
        self.decode_buffer = []
        self.grabbed , self.frame = self.vcap.read()
        self.decode_buffer.append(self.frame)
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True 

    def start(self):
        self.stopped = False
        self.thread.start()

    def update(self):
        while not self.stopped:
            grabbed, frame = self.vcap.read()
            if not grabbed:
                self.stopped = True
                break
            self.decode_buffer.append(frame)
        self.vcap.release()

    def read(self):
        try:
            return self.decode_buffer.pop(0)
        except:
            return None
    def stop(self):
        self.stopped = True 

    def get_fps(self):
        return self.vcap.get(cv2.CAP_PROP_FPS)

    def get_frame_count(self):
        return self.vcap.get(cv2.CAP_PROP_FRAME_COUNT)

    def get_fourcc(self):
        return cv2.VideoWriter_fourcc(*'mp4v')
    
    def get_shape(self):
        return self.frame.shape
    