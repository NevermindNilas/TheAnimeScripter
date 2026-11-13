"""Rotoscoping through the standalone RGBA writer.

``--segment_mask`` points at a folder of one matte per source frame
(``000123.png``, white on the object; see :func:`src.masking.listMaskFrames`) or at
a roto document (:mod:`src.segment.rotoDoc`) that adds hand-drawn shapes and edge
refinement to such a folder. ``--sam_session`` writes the folders for a front end's
click-to-track tool. The matte stands in for the segmentation model, so no weights
are loaded.
"""

import logging

import numpy as np
import torch

from src.constants import ADOBE
from src.segment.animeSegment import AnimeSegment
from src.segment.rotoDoc import RotoRenderer, decontaminate, loadRotoDoc, matteAlpha

if ADOBE:
    from src.server.aeComms import progressState

__all__ = ["TrackedMatteSegment", "matteAlpha"]


class TrackedMatteSegment(AnimeSegment):
    """Write each decoded frame with the roto document's alpha."""

    def __init__(self, *args, matteFolder: str, **kwargs):
        # Before super().__init__: it calls handleModel and runs the whole job.
        self.matteFolder = matteFolder
        super().__init__(*args, **kwargs)

    def handleModel(self):
        if ADOBE:
            progressState.update({"status": "Loading the rotoscope..."})

        from src.io.getVideoMetadata import trimFrameRange

        doc = loadRotoDoc(self.matteFolder)
        # Matte names count from the untrimmed input; decoding starts at the in point.
        offset, end = trimFrameRange(self.fps, self.inpoint, self.outpoint)
        self.renderer = RotoRenderer(doc, frameOffset=offset)
        self.position = 0
        if not self.renderer.frameFiles and not doc["shapes"]:
            raise ValueError(
                f"--segment_mask has no mattes and no shapes: {self.matteFolder}"
            )
        if end is None and self.totalFrames:
            end = offset + int(self.totalFrames)
        # A rotoscope that misses the trim would write a fully transparent video and
        # call it a success.
        wanted = (end - offset) if end is not None else None
        covered = self.renderer.framesCovered(offset, end)
        if covered == 0:
            raise ValueError(
                f"--segment_mask has no mattes or shapes on the frames this run renders "
                f"(source frames {offset} to {'the end' if end is None else end - 1}): {self.matteFolder}"
            )
        if wanted is not None and covered < wanted:
            logging.warning(
                f"--segment_mask covers {covered} of the {wanted} frames this run renders; "
                f"the other {wanted - covered} are written fully transparent."
            )
        logging.info(
            f"Rotoscoping from {len(self.renderer.frameFiles)} mattes and "
            f"{len(doc['shapes'])} shapes, starting at source frame {offset}"
        )

    @torch.inference_mode()
    def getMask(self, frames: list) -> torch.Tensor:
        rgb = frames[0] if len(frames) == 1 else torch.cat(frames, dim=0)
        rgb = rgb.float()
        height, width = rgb.shape[-2], rgb.shape[-1]
        # Advanced for the whole batch first: a batch that fails part way must not leave
        # every later frame paired with the matte of the frame before it.
        first = self.position
        self.position += rgb.shape[0]
        alphas = [
            self.renderer.alphaForPosition(first + i, height, width)
            for i in range(rgb.shape[0])
        ]
        alpha = torch.from_numpy(np.stack(alphas)).unsqueeze(1).to(rgb.device)
        if self.renderer.refine.get("decontaminate"):
            colours = rgb.permute(0, 2, 3, 1).cpu().numpy()
            cleaned = [
                decontaminate(c, a) for c, a in zip(colours, alphas, strict=True)
            ]
            rgb = (
                torch.from_numpy(np.stack(cleaned))
                .permute(0, 3, 1, 2)
                .to(rgb.device, rgb.dtype)
            )
        return torch.cat((rgb, alpha), dim=1)
