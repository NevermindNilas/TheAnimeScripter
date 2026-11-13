"""The MoGe-3 refiner through FlexGEMM's Triton kernels: about 4x the plain path.

Same weights as `Sparse3DUNet`, used in place. Triton compiles its kernels on the
user's machine, which needs NVIDIA CUDA, a C compiler (`triton-windows` bundles
one) and the Python headers, so `prepareFastRefiner` tries it and hands back the
plain refiner when any of that is missing. macOS has no Triton at all, and ROCm is
untested, so neither gets here.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor, nn

from src.infra.logAndPrint import logAndPrint

from .model import refineStep, voxelBins
from .sparse import Sparse3DUNet

_WIDE_CHANNELS = 128
# Under --half, fp16 convs only from this level on (256 and 512 channels): they carry
# most of the FLOPs. Levels 0-2 in fp16 moved one test frame (input_720p frame 90,
# low) 2.7x past the fp16 floor while the baseline's own spread there is 1.0x.
_FP16_FROM_LEVEL = 3


def _withChannels(shape: torch.Size, channels: int) -> torch.Size:
    return torch.Size([*shape[:-1], channels])


def _pooledShape(shape: torch.Size) -> torch.Size:
    """Halve H, W and Z, rounding up; the batch and channel axes stay."""
    return torch.Size([shape[0], *((axis + 1) // 2 for axis in shape[1:4]), shape[4]])


class FlexGemmRefiner(nn.Module):
    def __init__(self, plain: Sparse3DUNet, half: bool = False):
        super().__init__()
        from .flex_gemm.nn import SparsePool3d, SparseUpsample3d
        from .flex_gemm.ops.spconv.submanifold_conv import submanifold_conv

        self.plain = plain
        self.numLevels = plain.numLevels
        self.pool = SparsePool3d(kernel_size=2, stride=2, reduce="mean")
        self.upsample = SparseUpsample3d(scale_factor=2, mode="nearest")
        self._submanifoldConv = submanifold_conv
        self.plainOnly = False
        # Under --half the wide levels run fp16, like upstream's use_fp16; voxel
        # binning and the first and last projections stay fp32.
        self.half = half

    def _conv(self, conv, feats, coords, shape, cache):
        channels = max(conv.weight.shape[0], conv.weight.shape[-1])
        return self._submanifoldConv(
            feats,
            coords,
            shape,
            conv.weight,
            conv.bias,
            dilation=(1, 1, 1),
            neighbor_cache=cache,
            algorithm="masked_implicit_gemm"
            if channels >= _WIDE_CHANNELS
            else "implicit_gemm",
        )

    def _levelPrecision(self, level: int):
        return torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=self.half and level >= _FP16_FROM_LEVEL,
        )

    def _stage(self, blocks, level, feats, coords, shape, cache):
        if level < _FP16_FROM_LEVEL:
            feats = feats.float()
        with self._levelPrecision(level):
            for block in blocks:
                feats, cache = self._block(block, feats, coords, shape, cache)
        return feats, cache

    def _block(self, block, feats, coords, shape, cache):
        hidden = torch.nn.functional.silu(block.norm1(feats))
        hidden, cache = self._conv(block.conv1, hidden, coords, shape, cache)
        hidden = torch.nn.functional.silu(hidden)
        hidden, cache = self._conv(block.conv2, hidden, coords, shape, cache)
        return hidden + feats, cache

    def _unet(
        self, feats: Tensor, coords: Tensor, shape: torch.Size, tokens: Tensor
    ) -> Tensor:
        net = self.plain
        depth = self.numLevels
        # The input carries log depth (0..4) at 1/256 bin size; fp16 rounds it to
        # about half a bin, so the first and last projections stay fp32.
        with torch.autocast(device_type="cuda", enabled=False):
            feats = net.input_proj(feats.float())
        shape = _withChannels(shape, feats.shape[-1])
        convCaches: list = [None] * depth
        downCaches: list = [None] * (depth - 1)
        skips: list = [None] * (depth - 1)

        for k in range(depth):
            feats, convCaches[k] = self._stage(
                net.down_stages[k], k, feats, coords, shape, convCaches[k]
            )
            if k < depth - 1:
                skips[k] = (feats, coords, shape)
                feats, coords, shape, downCaches[k] = self.pool(
                    feats, coords, shape, output_shape=_pooledShape(shape)
                )
                feats = net.downsample_blocks[k].linear(feats)
                shape = _withChannels(shape, feats.shape[-1])

        indices = coords.long()
        sampled = tokens[indices[:, 0], :, indices[:, 1], indices[:, 2]]
        feats = net.fuse_proj(torch.cat([feats, net.encoder_fuse(sampled)], dim=-1))
        for block in net.bottleneck_stage:
            feats, convCaches[-1] = self._block(
                block, feats, coords, shape, convCaches[-1]
            )
        convCaches[-1] = None

        for i in range(depth - 1):
            target = depth - 2 - i
            skipFeats, skipCoords, skipShape = skips[target]
            feats = net.upsample_blocks[i].linear(feats)
            shape = _withChannels(shape, feats.shape[-1])
            feats, coords, shape, _ = self.upsample(
                feats,
                coords,
                shape,
                output_coords=skipCoords,
                output_shape=skipShape,
                neighbor_cache=downCaches[target].T,
            )
            feats = feats + skipFeats
            # Each level's skip and caches are done with once its stage has run;
            # holding them to the end stacks the whole pyramid at level 0.
            skips[target] = downCaches[target] = None
            del skipFeats
            feats, convCaches[target] = self._stage(
                net.up_stages[i], target, feats, coords, shape, convCaches[target]
            )
            convCaches[target] = None
        with torch.autocast(device_type="cuda", enabled=False):
            return net.out_proj(feats.float())

    def _step(self, coords: Tensor, tokens: Tensor) -> Tensor:
        height, width = coords.shape[1:3]
        device = coords.device
        bins = voxelBins(coords[0, :, :, 2])
        row = torch.arange(height, device=device).view(-1, 1).expand(height, width)
        col = torch.arange(width, device=device).view(1, -1).expand(height, width)
        voxels = (
            torch.stack([torch.zeros_like(row), row, col, bins], dim=-1)
            .reshape(-1, 4)
            .to(torch.int32)
        )
        shape = torch.Size([1, height, width, int(bins.max().item()) + 1, 3])
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=self.half,
        ):
            update = self._unet(coords.reshape(-1, 3), voxels, shape, tokens)
        logDepth = coords[..., 2] + update.float().reshape(1, height, width)
        return torch.cat([coords[..., :2], logDepth.unsqueeze(-1)], dim=-1)

    def fastStep(self, coords: Tensor, tokens: Tensor) -> Tensor:
        """One refinement update; a kernel that will not compile drops to plain."""
        if not self.plainOnly:
            try:
                return self._step(coords, tokens)
            except Exception as e:
                self.plainOnly = True
                _warnFallback(e)
        return refineStep(coords, tokens, self.plain)


def _warnFallback(error: Exception) -> None:
    logging.warning("MoGe-3 FlexGEMM refiner failed: %s", error)
    logAndPrint(
        "MoGe-3 fast refiner is unavailable here, using the slower one. "
        "See TAS-Log.log for the reason.",
        "yellow",
    )


def prepareFastRefiner(
    refiner: Sparse3DUNet, device: torch.device, half: bool = False
) -> nn.Module:
    """The FlexGEMM refiner when its kernels really compile here, else ``refiner``."""
    if device.type != "cuda" or torch.version.cuda is None:
        return refiner
    try:
        fast = FlexGemmRefiner(refiner, half)
        coords = torch.zeros(1, 32, 48, 3, device=device)
        coords[..., 2] = torch.linspace(0.0, 1.0, 48, device=device)
        tokens = torch.zeros(1, 1026, 2, 3, device=device)
        with torch.inference_mode():
            fast._step(coords, tokens)
    except Exception as e:
        _warnFallback(e)
        return refiner
    logging.info("MoGe-3 refiner: FlexGEMM")
    return fast
