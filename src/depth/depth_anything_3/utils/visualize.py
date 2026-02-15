# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
from einops import rearrange

from depth_anything_3.utils.logger import logger


def visualize_depth(
    depth: np.ndarray,
    depth_min=None,
    depth_max=None,
    percentile=2,
    ret_minmax=False,
    ret_type=np.uint8,
    cmap="Spectral",
):
    """
    Visualize a depth map using a colormap.

    Args:
        depth: Input depth map array
        depth_min: Minimum depth value for normalization. If None, uses percentile
        depth_max: Maximum depth value for normalization. If None, uses percentile
        percentile: Percentile for min/max computation if not provided
        ret_minmax: Whether to return min/max depth values
        ret_type: Return array type (uint8 or float)

    Returns:
        Colored depth visualization as numpy array
        If ret_minmax=True, also returns depth_min and depth_max
    """
    depth = depth.copy()
    depth.copy()
    valid_mask = depth > 0
    depth[valid_mask] = 1 / depth[valid_mask]
    if depth_min is None:
        if valid_mask.sum() <= 10:
            depth_min = 0
        else:
            depth_min = np.percentile(depth[valid_mask], percentile)
    if depth_max is None:
        if valid_mask.sum() <= 10:
            depth_max = 0
        else:
            depth_max = np.percentile(depth[valid_mask], 100 - percentile)
    if depth_min == depth_max:
        depth_min = depth_min - 1e-6
        depth_max = depth_max + 1e-6
    depth = ((depth - depth_min) / (depth_max - depth_min)).clip(0, 1)
    depth = 1 - depth



# GS video rendering visulization function, since it operates in Tensor space...


def vis_depth_map_tensor(
    result: torch.Tensor,  # "*batch height width"
    color_map: str = "Spectral",
) -> torch.Tensor:  # "*batch 3 height with"
    """
    Color-map the depth map.
    """
    far = result.reshape(-1)[:16_000_000].float().quantile(0.99).log().to(result)
    try:
        near = result[result > 0][:16_000_000].float().quantile(0.01).log().to(result)
    except (RuntimeError, ValueError) as e:
        logger.error(f"No valid depth values found. Reason: {e}")
        near = torch.zeros_like(far)
    result = result.log()
    result = (result - near) / (far - near)
    return apply_color_map_to_image(result, color_map)



def apply_color_map_to_image(
    image: torch.Tensor,  # "*batch height width"
    color_map: str = "inferno",
) -> torch.Tensor:  # "*batch 3 height with"
    return rearrange(image, "... h w c -> ... c h w")
