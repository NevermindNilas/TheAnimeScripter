import cv2
import os
import torch
import _thread
import time
import numpy as np
import skvideo.io

from collections import deque
from torch.nn import functional as F
from src.rife.pytorch_msssim import ssim_matlab
from tqdm import tqdm
from threading import Thread
from queue import Queue, Empty

@torch.inference_mode()

def clear_write_buffer(write_buffer, writer):
    while True:
        frame = write_buffer.get()
        if frame is None:
            break
        writer.write(frame[:, :, ::-1])

def build_read_buffer(video_file, read_buffer):
    frames = skvideo.io.vread(video_file)
    try:
        for frame in frames:
            read_buffer.put(frame)
    except:
        pass
    read_buffer.put(None)

def process_video_rife(video_file, output_path, model, scale, device, half):
    
    vcap = cv2.VideoCapture(video_file)
    total_frames = vcap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = vcap.get(cv2.CAP_PROP_FPS)
    fps *= scale
    h,w,_ = vcap.read()[1].shape
    vcap.release()
    
    read_buffer = Queue(maxsize=500)
    write_buffer = Queue(maxsize=500)

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (w, h))
    _thread.start_new_thread(build_read_buffer, (video_file, read_buffer))
    _thread.start_new_thread(clear_write_buffer, (write_buffer, writer))
    
    prev_frame = read_buffer.get()
    lastframe = prev_frame
    
    tmp = max(128, int(128 / scale))
    ph = ((h - 1) // tmp + 1) * tmp
    pw = ((w - 1) // tmp + 1) * tmp
    padding = (0, pw - w, 0, ph - h)
    
    progressbar = tqdm(total=total_frames, unit="frames")
    time.sleep(0.5)
    I1 = torch.from_numpy(np.transpose(lastframe, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
    I1 = pad_image(I1, half, padding)
    temp = None
    
    while True:
        if temp is not None:
            frame = temp
            temp = None
        else:
            frame = read_buffer.get()
        if frame is None:
            break
        I0 = I1
        I1 = torch.from_numpy(np.transpose(frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
        I1 = pad_image(I1, half, padding)
        I0_small = F.interpolate(I0, (32, 32), mode='bilinear', align_corners=False)
        I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
        ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])

        break_flag = False
        if ssim > 0.996:
            frame = read_buffer.get() # read a new frame
            if frame is None:
                break_flag = True
                frame = lastframe
            else:
                temp = frame
            I1 = torch.from_numpy(np.transpose(frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
            I1 = pad_image(I1, half, padding)
            I1 = model.inference(I0, I1, scale)
            I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
            ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
            frame = (I1[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]
            
        if ssim < 0.2:
            output = []
            for i in range(scale - 1):
                output.append(I0)
        else:
            output = make_inference(I0, I1, scale-1, scale, model)
            
        write_buffer.put(lastframe)
        for mid in output:
            mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
            write_buffer.put(mid[:h, :w])
        progressbar.update(1)
        lastframe = frame
        if break_flag:
            break
    
    while(not write_buffer.empty()):
        time.sleep(0.1)
    
    progressbar.close()
    writer.release()
        
def make_inference(I0, I1, n, scale, model):
    if model.version >= 3.9:
        res = []
        for i in range(n):
            res.append(model.inference(I0, I1, (i + 1) * 1.0 / (n + 1), scale))
        return res
    else:
        middle = model.inference(I0, I1, scale)
        if n == 1:
            return [middle]
        first_half = make_inference(I0, middle, n=n // 2)
        second_half = make_inference(middle, I1, n=n // 2)
        if n % 2:
            return [*first_half, middle, *second_half]
        else:
            return [*first_half, *second_half]


def pad_image(img, half, padding):
    if half:
        return F.pad(img, padding).half()
    else:
        return F.pad(img, padding)



