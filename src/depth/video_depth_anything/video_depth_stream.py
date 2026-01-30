# Copyright (2025) Bytedance Ltd. and/or its affiliates 

# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 

#     http://www.apache.org/licenses/LICENSE-2.0 

# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.transforms import Compose
import cv2
import numpy as np

from src.depth.video_depth_anything.dinov2 import DINOv2
from src.depth.video_depth_anything.dpt_temporal import DPTHeadTemporal
from src.depth.video_depth_anything.transform import Resize, NormalizeImage, PrepareForNet


# infer settings, do not change
INFER_LEN = 32
OVERLAP = 10
INTERP_LEN = 8

class VideoDepthAnything(nn.Module):
    def __init__(
        self,
        encoder='vitl',
        features=256, 
        out_channels=[256, 512, 1024, 1024], 
        use_bn=False, 
        use_clstoken=False,
        num_frames=32,
        pe='ape'
    ):
        super(VideoDepthAnything, self).__init__()

        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            "vitb": [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23]
        }
        
        self.encoder = encoder
        self.pretrained = DINOv2(model_name=encoder)

        self.head = DPTHeadTemporal(self.pretrained.embed_dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken, num_frames=num_frames, pe=pe)
        self.transform = None
        self.frame_id_list = []
        self.frame_cache_list = []
        self.id = -1

    def forward(self, x):
        return self.forward_depth(self.forward_features(x), x.shape)[0]
    
    def forward_features(self, x):
        features = self.pretrained.get_intermediate_layers(x.flatten(0,1), self.intermediate_layer_idx[self.encoder], return_class_token=True)
        return features

    def forward_depth(self, features, x_shape, cached_hidden_state_list=None):
        B, T, C, H, W = x_shape
        patch_h, patch_w = H // 14, W // 14
        depth, cur_cached_hidden_state_list = self.head(features, patch_h, patch_w, T, cached_hidden_state_list=cached_hidden_state_list)
        depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=True)
        depth = F.relu(depth)
        return depth.squeeze(1).unflatten(0, (B, T)), cur_cached_hidden_state_list # return shape [B, T, H, W]
    
    def infer_video_depth_one(self, frame, input_size=518, device='cuda', fp32=False):
        self.id += 1

        if self.transform is None:  # first frame
            # Initialize the transform
            frame_height, frame_width = frame.shape[:2]
            self.frame_height = frame_height
            self.frame_width = frame_width
            ratio = max(frame_height, frame_width) / min(frame_height, frame_width)
            if ratio > 1.78:  # we recommend to process video with ratio smaller than 16:9 due to memory limitation
                input_size = int(input_size * 1.777 / ratio)
                input_size = round(input_size / 14) * 14

            self.transform = Compose([
                Resize(
                    width=input_size,
                    height=input_size,
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method='lower_bound',
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ])

            # Inference the first frame
            cur_list = [torch.from_numpy(self.transform({'image': frame.astype(np.float32) / 255.0})['image']).unsqueeze(0).unsqueeze(0)]
            cur_input = torch.cat(cur_list, dim=1).to(device)
            
            with torch.no_grad():
                with torch.autocast(device_type=device, enabled=(not fp32)):
                    cur_feature = self.forward_features(cur_input)
                    x_shape = cur_input.shape
                    depth, cached_hidden_state_list = self.forward_depth(cur_feature, x_shape)

            depth = depth.to(cur_input.dtype)
            depth = F.interpolate(depth.flatten(0,1).unsqueeze(1), size=(frame_height, frame_width), mode='bilinear', align_corners=True)

            # Initialize cache with the first frame only. We'll pad with the most recent
            # cache during warm-up instead of repeating the first frame across the window.
            self.frame_cache_list = [cached_hidden_state_list]
            self.frame_id_list.append(0)

            new_depth = depth[0][0]
        else:
            frame_height, frame_width = frame.shape[:2]
            assert frame_height == self.frame_height
            assert frame_width == self.frame_width

            # infer feature
            cur_input = torch.from_numpy(self.transform({'image': frame.astype(np.float32) / 255.0})['image']).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                with torch.autocast(device_type=device, enabled=(not fp32)):
                    cur_feature = self.forward_features(cur_input)
                    x_shape = cur_input.shape

            cache_list = self.frame_cache_list
            if len(cache_list) >= INFER_LEN:
                cur_list = cache_list[0:2] + cache_list[-INFER_LEN+3:]
            else:
                cur_list = cache_list
                if len(cur_list) < INFER_LEN - 1:
                    pad_count = (INFER_LEN - 1) - len(cur_list)
                    cur_list = cur_list + [cur_list[-1]] * pad_count
            '''
            cur_id = self.frame_id_list[0:2] + self.frame_id_list[-INFER_LEN+3:]
            print(f"cur_id: {cur_id}")
            '''
            assert len(cur_list) == INFER_LEN - 1
            cur_cache = [torch.cat([h[i] for h in cur_list], dim=1) for i in range(len(cur_list[0]))]

            # infer depth
            with torch.no_grad():
                with torch.autocast(device_type=device, enabled=(not fp32)):
                    depth, new_cache = self.forward_depth(cur_feature, x_shape, cached_hidden_state_list=cur_cache)

            depth = depth.to(cur_input.dtype)
            depth = F.interpolate(depth.flatten(0,1).unsqueeze(1), size=(frame_height, frame_width), mode='bilinear', align_corners=True)
            depth_list = [depth[i][0] for i in range(depth.shape[0])]

            new_depth = depth_list[-1]

            self.frame_cache_list.append(new_cache)

        # adjust the sliding window to keep the most recent caches
        self.frame_id_list.append(self.id)
        max_cache = INFER_LEN - 1
        if len(self.frame_cache_list) > max_cache:
            self.frame_cache_list.pop(0)
            if len(self.frame_id_list) > max_cache:
                self.frame_id_list.pop(0)

        return new_depth
