"""Depth-video streaming with DA3 Small/Base and Limbo checkpoints."""

import logging
from functools import partial

import torch

from src.depth.backends._shared import (
    OrderedFrameSink,
    disparityStretch,
    stackFrames,
)
from src.depth.backends.cuda import DepthV3Cuda, LimboCuda
from src.depth.chunk_stream import streamDepthBlocks, verifyFinite
from src.infra.progressBarLogic import ProgressBarLogic


class DA3StreamingCuda(DepthV3Cuda):
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
        # Views of ONE scene [1,N,3,H,W], rather than independent images
        # [N,1,3,H,W]. This enables the checkpoint's cross-view attention.
        images = self._modelInput(frames).unsqueeze(0)
        # output_processor's depth [N,H,W], left on the device.
        return self.model.forward(images)["depth"].squeeze(0).squeeze(-1)

    @staticmethod
    def _depthGray(depth):
        """[..., H, W] chunk depth -> [0, 1] disparity, each map on its own.

        10 valid pixels or fewer is a black map; the range is floored at 1e-6.
        The chunk stream has already zeroed every non-finite or negative depth.
        """
        return disparityStretch(depth, minValid=10, depthFloor=None)

    @torch.inference_mode()
    def process(self):

        def frames():
            # Consume the decoder's EOF sentinel exactly once.
            while (frame := self.readBuffer.read()) is not None:
                yield frame

        count = 0
        with ProgressBarLogic(self.totalFrames, title="Depth") as bar:
            # A block's stretch is queued behind its chunk's forward and written
            # once the next block is queued too, so the wait for it overlaps the
            # next forward. A failure still stops the run where the serial loop
            # stopped, and nothing after it is written.
            with OrderedFrameSink(
                self._writeGray, None, bar, maxPending=max(1, self.depthWindow // 2)
            ) as sink:
                for block in streamDepthBlocks(
                    frames(), self._inferChunk, self.depthWindow
                ):
                    # The check holds the block's flag, not the block: that
                    # views its whole chunk's depth, which must not outlive
                    # this iteration into the next forward.
                    sink.submit(
                        self._depthGray(block.depths),
                        check=partial(verifyFinite, block.nonFinite),
                    )
                    count += len(block.depths)
                    del block
        logging.info("Processed %d frames with DA3 streaming", count)
        self.writeBuffer.close()


class LimboStreamingCuda(DA3StreamingCuda):
    def handleModels(self):
        if self.depth_method not in ("video_limbo", "video_limbo_v2", "video_limbo_v3"):
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
        # Decoder supplies RGB at Limbo's input resolution. Preserve the
        # image backend's ImageNet normalization without the DA3 PIL resize.
        images = stackFrames(frames, self._modelDevice())
        images = images.permute(0, 3, 1, 2).float() / 255
        images = LimboCuda.normFrame(self, images)
        result = self.model.forward(images.unsqueeze(0))
        return result["depth"][0].float()
