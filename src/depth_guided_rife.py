"""
Depth-Guided RIFE 4.25 Interpolation with Occlusion-Aware Blending

This module implements a depth-guided post-processing technique for RIFE 4.25
that uses Depth Anything V2 to compute visibility masks and improve blending
in large-parallax scenes.

Algorithm:
1. Run RIFE 4.25 to get warped frames, flows, and learned mask
2. Run Depth Anything V2 on both input frames
3. Warp depth maps to target time using the same flows
4. Compute visibility masks based on relative depth
5. Combine depth visibility with RIFE's mask for improved blending

Usage:
    interpolator = DepthGuidedRifeCuda(
        width=1920,
        height=1080,
        half=True,
        depth_method="small_v2",
    )
    
    # Interpolate between two frames
    output = interpolator.interpolate(img0, img1, timestep=0.5)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from .utils.downloadModels import downloadModels, weightsDir, modelsMap
from .utils.isCudaInit import CudaChecker
from .utils.logAndPrint import logAndPrint

checker = CudaChecker()

# Mean and std tensors for depth normalization
DEPTH_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
DEPTH_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def warp(tenInput, tenFlow):
    """
    Backward warping using optical flow.
    
    Args:
        tenInput: Input tensor [B, C, H, W]
        tenFlow: Optical flow [B, 2, H, W]
    
    Returns:
        Warped tensor [B, C, H, W]
    """
    B, _, H, W = tenFlow.shape
    
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, H, device=tenFlow.device, dtype=tenFlow.dtype),
        torch.linspace(-1.0, 1.0, W, device=tenFlow.device, dtype=tenFlow.dtype),
        indexing='ij'
    )
    grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
    
    flow_normalized = torch.cat([
        tenFlow[:, 0:1] / ((W - 1.0) / 2.0),
        tenFlow[:, 1:2] / ((H - 1.0) / 2.0),
    ], dim=1)
    
    sample_grid = (grid + flow_normalized).permute(0, 2, 3, 1)
    
    return F.grid_sample(
        tenInput,
        sample_grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )


class DepthEstimator:
    """
    Depth Anything V2 wrapper for depth estimation.
    """
    
    def __init__(
        self,
        depth_method: str = "small_v2",
        half: bool = True,
        width: int = 1920,
        height: int = 1080,
        depth_quality: str = "medium",
    ):
        """
        Initialize the depth estimator.
        
        Args:
            depth_method: Depth model variant ("small_v2", "base_v2", "large_v2")
            half: Use FP16 precision
            width: Input image width
            height: Input image height
            depth_quality: Depth estimation quality ("low", "medium", "high")
        """
        self.depth_method = depth_method
        self.half = half
        self.width = width
        self.height = height
        self.depth_quality = depth_quality
        self.device = checker.device
        self.dtype = torch.float16 if half else torch.float32
        
        self._load_model()
        self._setup_preprocessing()
    
    def _load_model(self):
        """Load the Depth Anything V2 model."""
        from .depth.dpt_v2 import DepthAnythingV2
        
        method_to_encoder = {
            "small_v2": "vits",
            "base_v2": "vitb",
            "large_v2": "vitl",
            "giant_v2": "vitg",
        }
        
        encoder = method_to_encoder.get(self.depth_method, "vits")
        
        model_configs = {
            "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
            "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
            "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
            "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
        }
        
        filename = modelsMap(model=self.depth_method, modelType="pth", half=self.half)
        model_path = os.path.join(weightsDir, self.depth_method, filename)
        
        if not os.path.exists(model_path):
            model_path = downloadModels(
                model=self.depth_method,
                half=self.half,
                modelType="pth",
            )
        
        self.model = DepthAnythingV2(**model_configs[encoder])
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model = self.model.to(self.device).eval()
        
        if self.half:
            self.model = self.model.half()
    
    def _setup_preprocessing(self):
        """Setup preprocessing tensors and dimensions."""
        if self.depth_quality == "high":
            self.depth_height = ((self.height + 13) // 14) * 14
            self.depth_width = ((self.width + 13) // 14) * 14
        elif self.depth_quality == "medium":
            self.depth_height = 518
            self.depth_width = 518
        else:  # low
            self.depth_height = 364
            self.depth_width = 364
        
        self.mean = DEPTH_MEAN.to(device=self.device, dtype=self.dtype)
        self.std = DEPTH_STD.to(device=self.device, dtype=self.dtype)
    
    @torch.inference_mode()
    def estimate(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Estimate depth for a single frame.
        
        Args:
            frame: Input frame [B, 3, H, W] in range [0, 1]
        
        Returns:
            Depth map [B, 1, H, W] normalized to [0, 1]
        """
        frame_resized = F.interpolate(
            frame,
            (self.depth_height, self.depth_width),
            mode="bilinear",
            align_corners=True,
        )
        
        frame_normalized = (frame_resized - self.mean) / self.std
        
        if self.half:
            frame_normalized = frame_normalized.half()
        
        depth = self.model(frame_normalized)
        
        if depth.ndim == 3:
            depth = depth.unsqueeze(1)
        
        depth = F.interpolate(
            depth,
            (self.height, self.width),
            mode="bilinear",
            align_corners=True,
        )
        
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        
        return depth


class DepthGuidedRifeCuda:
    """
    Depth-Guided RIFE 4.25 interpolation with occlusion-aware blending.
    
    This class wraps RIFE 4.25 and adds depth-guided post-processing to
    improve blending quality in large-parallax scenes.
    """
    
    def __init__(
        self,
        width: int,
        height: int,
        half: bool = True,
        interpolate_method: str = "rife4.25",
        depth_method: str = "small_v2",
        depth_quality: str = "medium",
        alpha: float = 10.0,
        blend_mode: str = "multiply",
        ensemble: bool = False,
    ):
        """
        Initialize the depth-guided RIFE interpolator.
        
        Args:
            width: Frame width
            height: Frame height
            half: Use FP16 precision
            interpolate_method: RIFE variant to use
            depth_method: Depth model variant
            depth_quality: Depth estimation quality
            alpha: Sharpness of depth visibility transition
            blend_mode: How to combine depth visibility with mask
                        ("multiply", "replace", "weighted")
            ensemble: Use ensemble mode for RIFE
        """
        self.width = width
        self.height = height
        self.half = half
        self.interpolate_method = interpolate_method
        self.depth_method = depth_method
        self.depth_quality = depth_quality
        self.alpha = alpha
        self.blend_mode = blend_mode
        self.ensemble = ensemble
        
        self.device = checker.device
        self.dtype = torch.float16 if half else torch.float32
        
        self._load_rife_model()
        self._load_depth_model()
        self._setup_buffers()
    
    def _load_rife_model(self):
        """Load the RIFE 4.25 model."""
        from .rifearches.IFNet_rife425 import IFNet
        
        filename = modelsMap(self.interpolate_method)
        model_path = os.path.join(weightsDir, "rife", filename)
        
        if not os.path.exists(model_path):
            model_path = downloadModels(model=self.interpolate_method)
        
        self.scale = 0.5 if (self.width > 1920 and self.height > 1080) else 1.0
        
        if self.scale == 0.5 and self.half:
            logAndPrint(
                "UHD and fp16 are not compatible with RIFE, defaulting to fp32",
                "yellow",
            )
            self.half = False
            self.dtype = torch.float32
        
        self.rife = IFNet(
            ensemble=self.ensemble,
            dynamicScale=False,
            scale=self.scale,
            interpolateFactor=2,
        )
        
        if self.half:
            self.rife.half()
        else:
            self.rife.float()
        
        self.rife.load_state_dict(torch.load(model_path, map_location=self.device))
        self.rife.eval().to(self.device)
        self.rife = self.rife.to(memory_format=torch.channels_last)
    
    def _load_depth_model(self):
        """Load the depth estimation model."""
        self.depth_estimator = DepthEstimator(
            depth_method=self.depth_method,
            half=self.half,
            width=self.width,
            height=self.height,
            depth_quality=self.depth_quality,
        )
    
    def _setup_buffers(self):
        """Setup padding and buffer tensors."""
        ph = ((self.height - 1) // 128 + 1) * 128
        pw = ((self.width - 1) // 128 + 1) * 128
        self.padding = (0, pw - self.width, 0, ph - self.height)
        self.padded_height = ph
        self.padded_width = pw
        
        self.I0 = torch.zeros(
            1, 3, ph, pw,
            dtype=self.dtype,
            device=self.device,
        ).to(memory_format=torch.channels_last)
        
        self.I1 = torch.zeros(
            1, 3, ph, pw,
            dtype=self.dtype,
            device=self.device,
        ).to(memory_format=torch.channels_last)
        
        self.D0 = torch.zeros(
            1, 1, ph, pw,
            dtype=self.dtype,
            device=self.device,
        )
        
        self.D1 = torch.zeros(
            1, 1, ph, pw,
            dtype=self.dtype,
            device=self.device,
        )
        
        self.stream = torch.cuda.Stream()
        self.firstRun = True
        
        logging.info(f"DepthGuidedRifeCuda initialized: {self.width}x{self.height}, "
                     f"padded to {pw}x{ph}, half={self.half}")
    
    def _pad_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """Pad frame to required size."""
        if self.padding != (0, 0, 0, 0):
            return F.pad(frame, self.padding)
        return frame
    
    def _compute_depth_visibility(
        self,
        D0: torch.Tensor,
        D1: torch.Tensor,
        flow_0_to_t: torch.Tensor,
        flow_1_to_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute depth-based visibility masks.
        
        Args:
            D0: Depth map from frame 0 [B, 1, H, W]
            D1: Depth map from frame 1 [B, 1, H, W]
            flow_0_to_t: Flow from frame 0 to target [B, 2, H, W]
            flow_1_to_t: Flow from frame 1 to target [B, 2, H, W]
        
        Returns:
            visibility_0, visibility_1: Visibility masks [B, 1, H, W]
        """
        D0_warped = warp(D0, flow_0_to_t)
        D1_warped = warp(D1, flow_1_to_t)
        
        depth_diff = D1_warped - D0_warped
        
        visibility_0 = torch.sigmoid(self.alpha * depth_diff)
        visibility_1 = 1.0 - visibility_0
        
        kernel_size = 3
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=self.device, dtype=self.dtype)
        kernel = kernel / (kernel_size * kernel_size)
        
        visibility_0 = F.conv2d(visibility_0, kernel, padding=kernel_size // 2)
        visibility_1 = F.conv2d(visibility_1, kernel, padding=kernel_size // 2)
        
        return visibility_0, visibility_1
    
    def _depth_guided_blend(
        self,
        warped_img0: torch.Tensor,
        warped_img1: torch.Tensor,
        rife_mask: torch.Tensor,
        vis_0: torch.Tensor,
        vis_1: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform depth-guided blending.
        
        Args:
            warped_img0: Warped frame 0 [B, 3, H, W]
            warped_img1: Warped frame 1 [B, 3, H, W]
            rife_mask: RIFE's learned blending mask [B, 1, H, W]
            vis_0: Visibility mask for frame 0 [B, 1, H, W]
            vis_1: Visibility mask for frame 1 [B, 1, H, W]
        
        Returns:
            Blended output [B, 3, H, W]
        """
        if self.blend_mode == "replace":
            W0 = vis_0
            W1 = vis_1
        elif self.blend_mode == "weighted":
            W0 = 0.5 * rife_mask + 0.5 * vis_0
            W1 = 0.5 * (1 - rife_mask) + 0.5 * vis_1
        else:
            W0 = rife_mask * vis_0
            W1 = (1 - rife_mask) * vis_1
        
        W_sum = W0 + W1 + 1e-8
        W0 = W0 / W_sum
        W1 = W1 / W_sum
        
        return warped_img0 * W0 + warped_img1 * W1
    
    @torch.inference_mode()
    def cacheFrameReset(self, frame: torch.Tensor):
        """
        Reset the temporal state with a new frame.
        Called on scene changes.
        """
        with torch.cuda.stream(self.stream):
            frame = frame.to(device=self.device, dtype=self.dtype)
            self.I0.copy_(self._pad_frame(frame))
            D0 = self.depth_estimator.estimate(frame)
            self.D0.copy_(self._pad_frame(D0))
            self.rife.cacheReset(self.I0)
            self.firstRun = False
    
    @torch.inference_mode()
    def __call__(
        self,
        frame: torch.Tensor,
        interpQueue,
        framesToInsert: int = 1,
        timesteps=None,
    ):
        """
        Stateful interpolation call for integration with main.py.
        """
        with torch.cuda.stream(self.stream):
            frame = frame.to(device=self.device, dtype=self.dtype)
            
            if self.firstRun:
                self.I0.copy_(self._pad_frame(frame))
                D0 = self.depth_estimator.estimate(frame)
                self.D0.copy_(self._pad_frame(D0))
                self.rife.f0 = self.rife.encode(self.I0)
                self.firstRun = False
                return

            self.I1.copy_(self._pad_frame(frame))
            D1 = self.depth_estimator.estimate(frame)
            self.D1.copy_(self._pad_frame(D1))
            self.rife.f1 = self.rife.encode(self.I1)

            for i in range(framesToInsert):
                if timesteps is not None and i < len(timesteps):
                    t = timesteps[i]
                else:
                    t = (i + 1) * 1.0 / (framesToInsert + 1)
                
                res = self._interpolate_step(t)
                interpQueue.put(res)

            self.I0.copy_(self.I1)
            self.D0.copy_(self.D1)
            self.rife.cache()
        
        self.stream.synchronize()

    def _interpolate_step(self, timestep: float) -> torch.Tensor:
        """Internal interpolation step using stateful buffers."""
        output, flow, mask, warped_img0, warped_img1 = self._rife_forward_step(timestep)
        
        vis_0, vis_1 = self._compute_depth_visibility(
            self.D0, self.D1,
            flow[:, :2], flow[:, 2:4]
        )
        
        result = self._depth_guided_blend(
            warped_img0, warped_img1, mask, vis_0, vis_1
        )
        
        return result[:, :, :self.height, :self.width].clone()

    def _rife_forward_step(
        self,
        timestep: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """RIFE forward step using self.rife.f0 and self.rife.f1."""
        timestep_tensor = torch.full(
            (1, 1, self.padded_height, self.padded_width),
            timestep,
            dtype=self.dtype,
            device=self.device,
        )
        
        f0 = self.rife.f0
        f1 = self.rife.f1
        img0 = self.I0
        img1 = self.I1
        
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        mask = None
        
        for i, block in enumerate(self.rife.blocks):
            if flow is None:
                flow, mask, feat = block(
                    torch.cat((img0[:, :3], img1[:, :3], f0, f1, timestep_tensor), 1),
                    None,
                    scale=self.rife.scale_list[i],
                )
            else:
                wf0 = warp(f0, flow[:, :2])
                wf1 = warp(f1, flow[:, 2:4])
                fd, m0, feat = block(
                    torch.cat((
                        warped_img0[:, :3],
                        warped_img1[:, :3],
                        wf0,
                        wf1,
                        timestep_tensor,
                        mask,
                        feat,
                    ), 1),
                    flow,
                    scale=self.rife.scale_list[i],
                )
                mask = m0
                flow = flow + fd
            
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
        
        mask = torch.sigmoid(mask)
        output = warped_img0 * mask + warped_img1 * (1 - mask)
        
        return output, flow, mask, warped_img0, warped_img1

    @torch.inference_mode()
    def interpolate(
        self,
        img0: torch.Tensor,
        img1: torch.Tensor,
        timestep: float = 0.5,
    ) -> torch.Tensor:
        """
        Original interpolate method for manual use (kept for compatibility).
        """
        with torch.cuda.stream(self.stream):
            img0 = img0.to(device=self.device, dtype=self.dtype)
            img1 = img1.to(device=self.device, dtype=self.dtype)
            
            self.I0.copy_(self._pad_frame(img0))
            self.I1.copy_(self._pad_frame(img1))
            
            self.D0.copy_(self._pad_frame(self.depth_estimator.estimate(img0)))
            self.D1.copy_(self._pad_frame(self.depth_estimator.estimate(img1)))
            
            self.rife.f0 = self.rife.encode(self.I0)
            self.rife.f1 = self.rife.encode(self.I1)
            
            result = self._interpolate_step(timestep)
        
        self.stream.synchronize()
        return result

    
    @torch.inference_mode()
    def interpolate_simple(
        self,
        img0: torch.Tensor,
        img1: torch.Tensor,
        timestep: float = 0.5,
    ) -> torch.Tensor:
        """
        Simple interpolation without depth guidance (standard RIFE).
        Useful for comparison.
        
        Args:
            img0: First frame [B, 3, H, W] in range [0, 1]
            img1: Second frame [B, 3, H, W] in range [0, 1]
            timestep: Interpolation timestep
        
        Returns:
            Interpolated frame [B, 3, H, W]
        """
        with torch.cuda.stream(self.stream):
            img0 = img0.to(device=self.device, dtype=self.dtype)
            img1 = img1.to(device=self.device, dtype=self.dtype)
            
            img0_padded = self._pad_frame(img0)
            img1_padded = self._pad_frame(img1)
            
            timestep_tensor = torch.full(
                (1, 1, self.padded_height, self.padded_width),
                timestep,
                dtype=self.dtype,
                device=self.device,
            )
            
            self.rife.f0 = self.rife.encode(img0_padded)
            self.rife.f1 = self.rife.encode(img1_padded)
            
            result = self.rife(img0_padded, img1_padded, timestep_tensor)
            result = result[:, :, :self.height, :self.width]
        
        self.stream.synchronize()
        return result


def test_depth_guided_rife():
    """Test function for DepthGuidedRifeCuda."""
    import numpy as np
    from PIL import Image
    
    print("Testing DepthGuidedRifeCuda...")
    
    H, W = 1080, 1920
    
    interpolator = DepthGuidedRifeCuda(
        width=W,
        height=H,
        half=True,
        depth_method="small_v2",
        depth_quality="medium",
        alpha=10.0,
    )
    
    img0 = torch.rand(1, 3, H, W, dtype=torch.float32)
    img1 = torch.rand(1, 3, H, W, dtype=torch.float32)
    
    print("Running depth-guided interpolation...")
    result = interpolator.interpolate(img0, img1, timestep=0.5)
    print(f"Output shape: {result.shape}")
    print(f"Output dtype: {result.dtype}")
    print(f"Output range: [{result.min():.3f}, {result.max():.3f}]")
    
    # Run simple interpolation for comparison
    print("\nRunning simple interpolation for comparison...")
    result_simple = interpolator.interpolate_simple(img0, img1, timestep=0.5)
    print(f"Simple output shape: {result_simple.shape}")
    
    print("\nTest completed successfully!")


if __name__ == "__main__":
    test_depth_guided_rife()
