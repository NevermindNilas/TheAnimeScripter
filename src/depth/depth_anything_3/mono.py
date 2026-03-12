from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn

try:
    from safetensors.torch import load_file as load_safetensors_file
except ImportError:
    load_safetensors_file = None

from depth_anything_3.model.da3 import DepthAnything3Net
from depth_anything_3.model.dinov2.dinov2 import DinoV2
from depth_anything_3.model.dpt import DPT
from depth_anything_3.model.dualdpt import DualDPT
from depth_anything_3.utils.alignment import compute_sky_mask, set_sky_regions_to_max_depth
from depth_anything_3.utils.io.input_processor import InputProcessor
from depth_anything_3.utils.io.output_processor import OutputProcessor
from depth_anything_3.utils.model_loading import convert_general_state_dict, convert_metric_state_dict


MONO_PRESETS: dict[str, dict[str, Any]] = {
    "da3-small": {
        "backbone": {
            "name": "vits",
            "out_layers": [5, 7, 9, 11],
            "alt_start": 4,
            "qknorm_start": 4,
            "rope_start": 4,
            "cat_token": True,
        },
        "head": {
            "kind": "dualdpt",
            "dim_in": 768,
            "output_dim": 2,
            "features": 64,
            "out_channels": [48, 96, 192, 384],
        },
    },
    "da3-base": {
        "backbone": {
            "name": "vitb",
            "out_layers": [5, 7, 9, 11],
            "alt_start": 4,
            "qknorm_start": 4,
            "rope_start": 4,
            "cat_token": True,
        },
        "head": {
            "kind": "dualdpt",
            "dim_in": 1536,
            "output_dim": 2,
            "features": 128,
            "out_channels": [96, 192, 384, 768],
        },
    },
    "da3metric-large": {
        "backbone": {
            "name": "vitl",
            "out_layers": [4, 11, 17, 23],
            "alt_start": -1,
            "qknorm_start": -1,
            "rope_start": -1,
            "cat_token": False,
        },
        "head": {
            "kind": "dpt",
            "dim_in": 1024,
            "output_dim": 1,
            "features": 256,
            "out_channels": [256, 512, 1024, 1024],
        },
    },
}

MODEL_ALIASES = {
    "small": "da3-small",
    "vits": "da3-small",
    "base": "da3-base",
    "vitb": "da3-base",
    "metric-large": "da3metric-large",
    "da3-metric-large": "da3metric-large",
    "metric-vitl": "da3metric-large",
    "large": "da3metric-large",
    "vitl": "da3metric-large",
}


def _normalize_model_name(model_name: str) -> str:
    key = model_name.strip().lower().replace("_", "-")
    if key in MONO_PRESETS:
        return key
    if key in MODEL_ALIASES:
        return MODEL_ALIASES[key]
    if ("large" in key) or ("vitl" in key):
        return "da3metric-large"
    if ("base" in key) or ("vitb" in key):
        return "da3-base"
    if ("small" in key) or ("vits" in key):
        return "da3-small"
    raise ValueError(f"Unsupported monocular Depth Anything 3 preset: {model_name}")


def _build_model(model_name: str) -> DepthAnything3Net:
    preset = MONO_PRESETS[_normalize_model_name(model_name)]
    backbone = DinoV2(**preset["backbone"])
    head_cfg = preset["head"]

    if head_cfg["kind"] == "dualdpt":
        head = DualDPT(
            dim_in=head_cfg["dim_in"],
            output_dim=head_cfg["output_dim"],
            features=head_cfg["features"],
            out_channels=head_cfg["out_channels"],
        )
    else:
        head = DPT(
            dim_in=head_cfg["dim_in"],
            output_dim=head_cfg["output_dim"],
            features=head_cfg["features"],
            out_channels=head_cfg["out_channels"],
        )

    return DepthAnything3Net(net=backbone, head=head)


def _load_checkpoint_payload(checkpoint_path: Path) -> dict[str, Any]:
    if checkpoint_path.suffix.lower() == ".safetensors":
        if load_safetensors_file is None:
            raise ImportError("safetensors is required to load .safetensors checkpoints")
        return load_safetensors_file(str(checkpoint_path), device="cpu")
    return torch.load(checkpoint_path, map_location="cpu")


def _unwrap_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "model", "weights"):
            value = payload.get(key)
            if isinstance(value, dict) and value:
                return value
        if payload and all(isinstance(key, str) for key in payload):
            return payload
    raise TypeError("Checkpoint does not contain a state dict")


def _prefix_state_dict(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {f"{prefix}{key}": value for key, value in state_dict.items()}


def _normalize_state_dict(state_dict: dict[str, torch.Tensor], model_name: str) -> dict[str, torch.Tensor]:
    keys = tuple(state_dict.keys())
    if not keys:
        raise ValueError("Checkpoint state dict is empty")
    if any(key.startswith("model.") for key in keys):
        return state_dict
    if any(key.startswith(("backbone.", "head.")) for key in keys):
        return _prefix_state_dict(state_dict, "model.")
    if _normalize_model_name(model_name) == "da3metric-large":
        return convert_metric_state_dict(state_dict)
    return convert_general_state_dict(state_dict)


def _resize_depth_map(depth: np.ndarray, height: int, width: int) -> np.ndarray:
    if depth.shape[0] == height and depth.shape[1] == width:
        return depth

    interpolation = cv2.INTER_LINEAR
    if height < depth.shape[0] or width < depth.shape[1]:
        interpolation = cv2.INTER_LINEAR

    return cv2.resize(depth, (width, height), interpolation=interpolation)


class MonocularDepthAnything3(nn.Module):
    def __init__(self, model_name: str = "da3metric-large"):
        super().__init__()
        self.model_name = _normalize_model_name(model_name)
        self.model = _build_model(self.model_name)
        self.model.eval()
        self.input_processor = InputProcessor()
        self.output_processor = OutputProcessor()
        self.device = None

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        model_name: str | None = None,
        *,
        strict: bool = False,
    ) -> MonocularDepthAnything3:
        source = Path(model_path)
        if not source.exists():
            raise FileNotFoundError(f"Monocular Depth Anything 3 checkpoint not found: {source}")

        resolved_model_name = _normalize_model_name(model_name or source.stem)
        instance = cls(resolved_model_name)
        payload = _load_checkpoint_payload(source)
        state_dict = _normalize_state_dict(_unwrap_state_dict(payload), resolved_model_name)
        instance.load_state_dict(state_dict, strict=strict)
        instance.eval()
        return instance

    @torch.inference_mode()
    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        use_autocast = image.device.type == "cuda"
        autocast_dtype = (
            torch.bfloat16 if use_autocast and torch.cuda.is_bf16_supported() else torch.float16
        )
        with torch.no_grad():
            with torch.autocast(
                device_type=image.device.type,
                dtype=autocast_dtype,
                enabled=use_autocast,
            ):
                output = self.model(image, None, None, [], False, False, "first")

        if "sky" in output:
            non_sky_mask = compute_sky_mask(output.sky, threshold=0.3)
            if non_sky_mask.sum() > 10 and (~non_sky_mask).sum() > 10:
                non_sky_depth = output.depth[non_sky_mask]
                max_depth = torch.quantile(non_sky_depth, 0.99)
                output.depth, _ = set_sky_regions_to_max_depth(
                    output.depth,
                    None,
                    non_sky_mask,
                    max_depth=max_depth,
                )
        return output

    def infer_image(
        self,
        image: np.ndarray,
        process_res: int = 504,
        process_res_method: str = "upper_bound_resize",
    ) -> np.ndarray:
        original_height, original_width = image.shape[:2]
        imgs_cpu, _, _ = self.input_processor(
            [image],
            None,
            None,
            process_res,
            process_res_method,
        )
        device = self._get_model_device()
        imgs = imgs_cpu.to(device, non_blocking=True)[None].float()
        output = self.forward(imgs)
        prediction = self.output_processor(output)
        depth = prediction.depth[0]
        depth = _resize_depth_map(depth, original_height, original_width)
        return np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def _get_model_device(self) -> torch.device:
        if self.device is not None:
            return self.device
        for param in self.parameters():
            self.device = param.device
            return param.device
        for buffer in self.buffers():
            self.device = buffer.device
            return buffer.device
        raise ValueError("No tensor found in model")