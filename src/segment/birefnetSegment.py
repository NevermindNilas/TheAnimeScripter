"""BiRefNet Lite anime segmentation through the standalone RGBA writer."""

import logging

import torch
import torch.nn.functional as F

from src.constants import ADOBE
from src.infra.isCudaInit import CudaChecker
from src.model.download import resolveWeightPath
from src.model.registry import modelsMap
from src.segment.animeSegment import AnimeSegment

if ADOBE:
    from src.server.aeComms import progressState


class BiRefNetSegment(AnimeSegment):
    """Run the 1024-square Real Anime checkpoint and write RGB plus alpha."""

    def handleModel(self):
        if ADOBE:
            progressState.update(
                {"status": "Loading BiRefNet background removal model..."}
            )

        from .birefnet_arch.birefnet import BiRefNet

        self.device = CudaChecker().device
        if self.device.type == "mps":
            # torchvision's deform_conv2d has no MPS kernel.
            self.device = torch.device("cpu")
            logging.warning(
                "BiRefNet uses CPU on Apple Silicon (deform_conv2d has no MPS kernel)"
            )

        filename = modelsMap("birefnet")
        modelPath = resolveWeightPath("birefnet", filename)
        self.model = BiRefNet(bb_pretrained=False).eval()
        state = torch.load(modelPath, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device)
        self.mean = torch.tensor((0.485, 0.456, 0.406), device=self.device).view(
            1, 3, 1, 1
        )
        self.std = torch.tensor((0.229, 0.224, 0.225), device=self.device).view(
            1, 3, 1, 1
        )

    @torch.inference_mode()
    def getMask(self, frames: list) -> torch.Tensor:
        rgb = frames[0] if len(frames) == 1 else torch.cat(frames, dim=0)
        rgb = rgb.to(self.device).float()
        normalized = F.interpolate(
            rgb, size=(1024, 1024), mode="bilinear", align_corners=False
        )
        normalized = (normalized - self.mean) / self.std
        logits = self.model(normalized)[-1]
        alpha = F.interpolate(
            logits.sigmoid().float(),
            size=rgb.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        rgba = torch.cat((rgb, alpha), dim=1)
        if self.device.type == "cuda":
            # The writer copies from a private CUDA stream.
            torch.cuda.current_stream().synchronize()
        return rgba
