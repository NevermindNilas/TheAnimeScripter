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

def clear_write_buffer(output_path, write_buffer):
    writer = skvideo.io.FFmpegWriter(output_path)
    while True:
        frame = write_buffer.get()
        if frame is None:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert color space back to BGR
        writer.writeFrame(frame)
    writer.close()

def build_read_buffer(video_file, read_buffer):
    frames = skvideo.io.vread(video_file)
    frame_idx = 0
    while frame_idx < len(frames):
        frame = frames[frame_idx]
        frame_idx += 1
        read_buffer.put(frame)

def process_video_rife(video_file, output_path, model, scale, device, half):
    
    vcap = cv2.VideoCapture(video_file)
    total_frames = vcap.get(cv2.CAP_PROP_FRAME_COUNT)
    vcap.release()
    
    read_buffer = Queue(maxsize=500)
    write_buffer = Queue(maxsize=500)
    _thread.start_new_thread(build_read_buffer, (video_file, read_buffer))
    _thread.start_new_thread(clear_write_buffer, (output_path, write_buffer))
    
    prev_frame = read_buffer.get()
    lastframe = prev_frame
    h, w, _ = prev_frame.shape
    tmp = max(128, int(128 / scale))
    ph = ((h - 1) // tmp + 1) * tmp
    pw = ((w - 1) // tmp + 1) * tmp
    padding = (0, pw - w, 0, ph - h)
    progressbar = tqdm(total=total_frames, unit="frames")
    I1 = (
        torch.from_numpy(np.transpose(lastframe, (2, 0, 1)))
        .to(device, non_blocking=True)
        .unsqueeze(0)
        .float()
        / 255.0
    )
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
        I1 = (
            torch.from_numpy(np.transpose(frame, (2, 0, 1)))
            .to(device, non_blocking=True)
            .unsqueeze(0)
            .float()
            / 255.0
        )
        I1 = pad_image(I1, half, padding)
        break_flag = False
        output = make_inference(I0, I1, scale - 1, scale, model)

        write_buffer.put(lastframe)
        for mid in output:
            mid = (mid[0] * 255.0).byte().cpu().numpy().transpose(1, 2, 0)
            write_buffer.put(mid[:h, :w])

        progressbar.update(1)
        lastframe = frame
        if break_flag:
            break
    
    while(not write_buffer.empty()):
        time.sleep(0.1)
    progressbar.close()

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



