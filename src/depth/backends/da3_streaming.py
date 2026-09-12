"""Depth-video streaming with DA3 Small/Base and Limbo checkpoints."""

import logging

import numpy as np
import torch

from src.depth.backends.cuda import LimboCuda, OGDepthV3Cuda
from src.depth.chunk_stream import streamDepthChunks
from src.infra.progressBarLogic import ProgressBarLogic


class DA3StreamingCuda(OGDepthV3Cuda):
    def __init__(self, *args, depth_window=32, **kwargs):
        if depth_window not in (4, 8, 16, 32):
            raise ValueError("DA3 streaming window must be 4, 8, 16 or 32 frames")
        self.depthWindow = depth_window
        super().__init__(*args, **kwargs)

    def handleModels(self):
        if self.depth_method not in ("video_small_v3", "video_base_v3"):
            raise ValueError(f"Unsupported DA3 streaming model: {self.depth_method}")
        super().handleModels()
        logging.info(
            "DA3 depth streaming: %s, chunk=%d, overlap=%d",
            self.depth_method,
            self.depthWindow,
            self.depthWindow // 2,
        )

    @torch.inference_mode()
    def _inferChunk(self, frames):
        images, _, _ = self.model.input_processor(
            frames, None, None, self.processRes, self.processResMethod
        )
        # Views of ONE scene [1,N,3,H,W], rather than independent images
        # [N,1,3,H,W]. This enables the checkpoint's cross-view attention.
        images = images.unsqueeze(0).to(self.model._get_model_device()).float()
        result = self.model.output_processor(self.model.forward(images))
        return result.depth

    def _writeDepth(self, depth):
        valid = np.isfinite(depth) & (depth > 0)
        if valid.sum() <= 10:
            self._writeGray(np.zeros_like(depth))
            return
        disparity = np.full_like(depth, np.nan)
        disparity[valid] = 1.0 / depth[valid]
        low, high = np.percentile(disparity[valid], (2, 98))
        gray = ((disparity - low) / max(high - low, 1e-6)).clip(0, 1)
        gray[~valid] = 0
        self._writeGray(gray)

    def process(self):

        def frames():
            # Consume the decoder's EOF sentinel exactly once.
            while (frame := self.readBuffer.read()) is not None:
                yield frame

        count = 0
        with ProgressBarLogic(self.totalFrames) as bar:
            for depth in streamDepthChunks(
                frames(), self._inferChunk, self.depthWindow
            ):
                self._writeDepth(depth)
                count += 1
                bar(1)
        logging.info("Processed %d frames with DA3 streaming", count)
        self.writeBuffer.close()


class LimboStreamingCuda(DA3StreamingCuda):
    def handleModels(self):
        if self.depth_method not in ("video_limbo", "video_limbo_v2"):
            raise ValueError(f"Unsupported Limbo streaming model: {self.depth_method}")
        LimboCuda.handleModels(self)
        self._decodeWidth = self.newWidth
        self._decodeHeight = self.newHeight
        self._decodeResize = True
        logging.info(
            "Limbo depth streaming: %s, chunk=%d, overlap=%d, input=%dx%d",
            self.depth_method,
            self.depthWindow,
            self.depthWindow // 2,
            self.newWidth,
            self.newHeight,
        )

    @torch.inference_mode()
    def _inferChunk(self, frames):
        # Decoder supplies RGB at Limbo's fixed resolution. Preserve the
        # image backend's ImageNet normalization without the DA3 PIL resize.
        images = torch.from_numpy(np.stack(frames)).to(self.model._get_model_device())
        images = images.permute(0, 3, 1, 2).float() / 255
        images = LimboCuda.normFrame(self, images)
        result = self.model.forward(images.unsqueeze(0))
        return result["depth"][0].float().cpu().numpy()
