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

from __future__ import annotations

import torch
import torch.nn as nn

from depth_anything_3.utils.alignment import compute_sky_mask, set_sky_regions_to_max_depth
from src.depth.attr_dict import AttrDict as Dict


class DepthAnything3Net(nn.Module):
    PATCH_SIZE = 14

    def __init__(self, net: nn.Module, head: nn.Module):
        super().__init__()
        if not isinstance(net, nn.Module) or not isinstance(head, nn.Module):
            raise TypeError("DepthAnything3Net expects instantiated backbone and head modules")
        self.backbone = net
        self.head = head

    def forward(
        self,
        x: torch.Tensor,
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
        export_feat_layers: list[int] | None = None,
        infer_gs: bool = False,
        use_ray_pose: bool = False,
        ref_view_strategy: str = "first",
    ) -> Dict[str, torch.Tensor]:
        if extrinsics is not None or intrinsics is not None:
            raise ValueError("The stripped DA3 runtime only supports monocular depth inference")
        if infer_gs or use_ray_pose:
            raise ValueError("The stripped DA3 runtime does not support GS or pose branches")

        feat_layers = export_feat_layers or []
        feats, aux_feats = self.backbone(
            x,
            cam_token=None,
            export_feat_layers=feat_layers,
            ref_view_strategy=ref_view_strategy,
        )
        height, width = x.shape[-2], x.shape[-1]

        with torch.autocast(device_type=x.device.type, enabled=False):
            output = self.head(feats, height, width, patch_start_idx=0)

        output = self._process_mono_sky_estimation(output)
        output.aux = self._extract_auxiliary_features(aux_feats, feat_layers, height, width)
        return output

    def _process_mono_sky_estimation(self, output: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if "sky" not in output:
            return output
        non_sky_mask = compute_sky_mask(output.sky, threshold=0.3)
        if non_sky_mask.sum() <= 10 or (~non_sky_mask).sum() <= 10:
            return output

        non_sky_depth = output.depth[non_sky_mask]
        if non_sky_depth.numel() > 100000:
            idx = torch.randint(0, non_sky_depth.numel(), (100000,), device=non_sky_depth.device)
            non_sky_depth = non_sky_depth[idx]

        non_sky_max = torch.quantile(non_sky_depth, 0.99)
        output.depth, _ = set_sky_regions_to_max_depth(
            output.depth,
            None,
            non_sky_mask,
            max_depth=non_sky_max,
        )
        return output

    def _extract_auxiliary_features(
        self,
        feats: list[torch.Tensor],
        feat_layers: list[int],
        height: int,
        width: int,
    ) -> Dict[str, torch.Tensor]:
        aux_features = Dict()
        for feat, feat_layer in zip(feats, feat_layers):
            aux_features[f"feat_layer_{feat_layer}"] = feat.reshape(
                feat.shape[0],
                feat.shape[1],
                height // self.PATCH_SIZE,
                width // self.PATCH_SIZE,
                feat.shape[-1],
            )
        return aux_features
