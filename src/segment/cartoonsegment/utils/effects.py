from numba import jit, njit
import numpy as np
import time
import cv2
import math
from pathlib import Path
import os.path as osp
import torch
from .cupy_utils import launch_kernel, preprocess_kernel
import cupy

def bokeh_filter_cupy(img, depth, dx, dy, im_h, im_w, num_samples=32):
    blurred = img.clone()
    n = im_h * im_w

    str_kernel = '''
        extern "C" __global__ void kernel_bokeh(
            const int n,
            const int h,
            const int w,
            const int nsamples,
            const float dx,
            const float dy,
            const float* img,
            const float* depth,
            float* blurred
        ) { 

            const int im_size = min(h, w);
            const int sample_offset = nsamples / 2;
            for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n * 3; intIndex += blockDim.x * gridDim.x) {

                const int intSample = intIndex / 3;

                const int c = intIndex % 3;
                const int y = ( intSample / w) % h;
                const int x = intSample % w;    

                const int flatten_xy = y * w + x;
                const int fid = flatten_xy * 3 + c;
                const float d = depth[flatten_xy];
                
                const float _dx = dx * d;
                const float _dy = dy * d;
                float weight = 0;
                float color = 0;
                for (int s = 0; s < nsamples; s += 1) {

                    const int sp = (s - sample_offset) * im_size;
                    const int x_ = x + int(round(_dx * sp));
                    const int y_ = y + int(round(_dy * sp));
                
                    if ((x_ >= w) | (y_ >= h) | (x_ < 0) | (y_ < 0))
                        continue;
                    
                    const int flatten_xy_ = y_ * w + x_;
                    const float w_ = depth[flatten_xy_];
                    weight += w_;
                    const int fid_ = flatten_xy_ * 3 + c;
                    color += img[fid_] * w_;
                }

                if (weight != 0) {
                    color /= weight;
                }
                else {
                    color = img[fid];
                }

                blurred[fid] = color;
                
            }

        }
    '''
    launch_kernel('kernel_bokeh', str_kernel)(
        grid=tuple([ int((n + 512 - 1) / 512), 1, 1 ]),
        block=tuple([ 512, 1, 1 ]),
        args=[ cupy.int32(n), cupy.int32(im_h), cupy.int32(im_w), \
              cupy.int32(num_samples), cupy.float32(dx), cupy.float32(dy),
              img.data_ptr(), depth.data_ptr(), blurred.data_ptr() ]
    )

    return blurred


def np2flatten_tensor(arr: np.ndarray, to_cuda: bool = True) -> torch.Tensor:
    c = 1
    if len(arr.shape) == 3:
        c = arr.shape[2]
    else:
        arr = arr[..., None]
    arr = arr.transpose((2, 0, 1))[None, ...]
    t = torch.from_numpy(arr).view(1, c, -1)
    
    if to_cuda:
        t = t.cuda()
    return t

def ftensor2img(t: torch.Tensor, im_h, im_w):
    t = t.detach().cpu().numpy().squeeze()
    c = t.shape[0]
    t = t.transpose((1, 0)).reshape((im_h, im_w, c))
    return t


@njit
def bokeh_filter(img, depth, dx, dy, num_samples=32):

    sample_offset = num_samples // 2
    # _scale = 0.0005
    # depth = depth * _scale

    im_h, im_w = img.shape[0], img.shape[1]
    im_size = min(im_h, im_w)
    blured = np.zeros_like(img)
    for x in range(im_w):
        for y in range(im_h):
            d = depth[y, x]
            _color = np.array([0, 0, 0], dtype=np.float32)
            _dx = dx * d
            _dy = dy * d
            weight = 0
            for s in range(num_samples):
                s = (s - sample_offset) * im_size
                x_ = x + int(round(_dx * s))
                y_ = y + int(round(_dy * s))
                if x_ >= im_w or y_ >= im_h or x_ < 0 or y_ < 0:
                    continue
                _w = depth[y_, x_]
                weight += _w
                _color += img[y_, x_] * _w
            if weight == 0:
                blured[y, x] = img[y, x]
            else:
                blured[y, x] = _color / np.array([weight, weight, weight], dtype=np.float32)
    
    return blured




def bokeh_blur(img, depth, num_samples=32, lightness_factor=10, depth_factor=2, use_cuda=False, focal_plane=None):
    img = np.ascontiguousarray(img)
    
    if depth is not None:
        depth = depth.astype(np.float32)
        if focal_plane is not None:
            depth = depth.max() - np.abs(depth - focal_plane)
        if depth_factor != 1:
            depth = np.power(depth, depth_factor)
        depth = depth - depth.min()
        depth = depth.astype(np.float32) / depth.max()
        depth = 1 - depth

    img = img.astype(np.float32) / 255
    img_hightlighted = np.power(img, lightness_factor)
    
    # img = 
    im_h, im_w = img.shape[:2]
    PI = math.pi

    _scale = 0.0005
    depth = depth * _scale

    if use_cuda:
        img_hightlighted = np2flatten_tensor(img_hightlighted, True)
        depth = np2flatten_tensor(depth, True)
        vertical_blured = bokeh_filter_cupy(img_hightlighted, depth, 0, 1, im_h, im_w, num_samples)
        diag_blured = bokeh_filter_cupy(vertical_blured, depth, math.cos(-PI/6), math.sin(-PI/6), im_h, im_w, num_samples)
        rhom_blur = bokeh_filter_cupy(diag_blured, depth, math.cos(-PI * 5 /6), math.sin(-PI * 5 /6), im_h, im_w, num_samples)
        blured = (diag_blured + rhom_blur) / 2
        blured = ftensor2img(blured, im_h, im_w)
    else:
        vertical_blured = bokeh_filter(img_hightlighted, depth, 0, 1, num_samples)
        diag_blured = bokeh_filter(vertical_blured, depth, math.cos(-PI/6), math.sin(-PI/6), num_samples)
        rhom_blur = bokeh_filter(diag_blured, depth, math.cos(-PI * 5 /6), math.sin(-PI * 5 /6), num_samples)
        blured = (diag_blured + rhom_blur) / 2
    blured = np.power(blured, 1 / lightness_factor)
    blured = (blured * 255).astype(np.uint8)

    return blured