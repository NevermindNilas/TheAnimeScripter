"""MoGe-3 ViT-L: DINOv2 + conv decoder + sparse 3D log-depth refiner.

`encode()` is the fixed-shape half (tensor network, exportable to ONNX as is);
`refineCoords()` is the dynamic half that needs per-frame voxel tables. Every backend
drives both and shares `coordsToGray`.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.depth.dinov2 import DINOv2

from .geometry import recoverShift, solveSamples, viewPlaneAxes, viewPlaneUv
from .sparse import Sparse3DUNet, buildLevels

# Tokens per --depth_quality; upstream's range is 1200..3600 (resolution_level
# 0..9), so low and high are its two ends.
QUALITY_TOKENS = {"low": 1200, "medium": 2400, "high": 3600}
REFINE_STEPS = 3
DEPTH_RESOLUTION = 256
TOKEN_PIXELS = 14
OUTPUT_STRIDE = 16
TOKEN_CHANNELS = 1026
PERCENTILE_SAMPLE_LIMIT = 4_000_000

_DIM_RES_BLOCKS = [1024, 256, 128, 64, 32]
_RESAMPLERS = ["conv_transpose", "conv_transpose", "conv_transpose", "bilinear"]
UNUSED_PREFIXES = ("normal_head.", "scale_head.")


def tokenGrid(width: int, height: int, depthQuality: str) -> tuple[int, int]:
    """(rows, columns) of ViT tokens for one source size."""
    tokens = QUALITY_TOKENS[depthQuality]
    aspect = width / height
    return max(1, round((tokens / aspect) ** 0.5)), max(
        1, round((tokens * aspect) ** 0.5)
    )


def makeUvGrids(
    rows: int,
    columns: int,
    aspectRatio: float,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> list[Tensor]:
    """[1, 2, rows*2^k, cols*2^k] view-plane UV for the 5 decoder levels."""
    return [
        viewPlaneUv(columns * 2**level, rows * 2**level, aspectRatio, dtype, device)
        .permute(2, 0, 1)
        .unsqueeze(0)
        for level in range(5)
    ]


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode="replicate"),
            nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode="replicate"),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x) + x


def _resampler(inChannels: int, outChannels: int, kind: str) -> nn.Sequential:
    if kind == "conv_transpose":
        first: nn.Module = nn.ConvTranspose2d(inChannels, outChannels, 2, stride=2)
        convIn = outChannels
    else:
        first = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        convIn = inChannels
    return nn.Sequential(
        first,
        nn.Conv2d(convIn, outChannels, 3, padding=1, padding_mode="replicate"),
    )


class ConvStack(nn.Module):
    def __init__(
        self,
        dimIn: list[int],
        dimOut: list[int | None],
        numResBlocks: list[int],
    ):
        super().__init__()
        self.input_blocks = nn.ModuleList(
            nn.Conv2d(inChannels, channels, 1)
            for inChannels, channels in zip(dimIn, _DIM_RES_BLOCKS, strict=True)
        )
        self.resamplers = nn.ModuleList(
            _resampler(_DIM_RES_BLOCKS[i], _DIM_RES_BLOCKS[i + 1], _RESAMPLERS[i])
            for i in range(len(_RESAMPLERS))
        )
        self.res_blocks = nn.ModuleList(
            nn.Sequential(*(ResidualConvBlock(channels) for _ in range(count)))
            for channels, count in zip(_DIM_RES_BLOCKS, numResBlocks, strict=True)
        )
        self.output_blocks = nn.ModuleList(
            nn.Conv2d(channels, out, 1) if out is not None else nn.Identity()
            for channels, out in zip(_DIM_RES_BLOCKS, dimOut, strict=True)
        )

    def forward(self, inputs: list[Tensor], lastOnly: bool = False) -> list[Tensor]:
        """Every level's output, or with lastOnly just the last one.

        The heads' earlier levels are Identity outputs nobody reads; keeping them
        in the list holds each level's map until the head returns.
        """
        outputs = []
        x = None
        last = len(self.res_blocks) - 1
        for i in range(len(self.res_blocks)):
            feature = self.input_blocks[i](inputs[i])
            x = feature if x is None else x + feature
            x = self.res_blocks[i](x)
            if not lastOnly or i == last:
                outputs.append(self.output_blocks[i](x))
            if i < last:
                x = self.resamplers[i](x)
        return outputs


class _Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = DINOv2("vitl")
        self.output_projections = nn.ModuleList(
            nn.Conv2d(1024, 1024, 1) for _ in range(4)
        )
        self.register_buffer(
            "image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )


class MoGe3(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _Encoder()
        self.neck = ConvStack([1026, 2, 2, 2, 2], [None] * 5, [0, 2, 2, 2, 0])
        self.points_head = ConvStack(
            _DIM_RES_BLOCKS, [None, None, None, None, 3], [0, 1, 1, 1, 0]
        )
        self.mask_head = ConvStack(
            _DIM_RES_BLOCKS, [None, None, None, None, 1], [0, 1, 1, 1, 0]
        )
        self.refiner = Sparse3DUNet()

    def encode(
        self,
        image: Tensor,
        aspectRatio: float,
        half: bool = False,
        uvGrids: list[Tensor] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """image: [1, 3, rows*14, cols*14] in [0, 1], fp32.

        Returns tokens [1, 1026, rows, cols] and, at 16x the token grid, the raw
        coords [1, 3, H, W] (x/z, y/z, log z) and mask logit [1, 1, H, W].
        uvGrids are `makeUvGrids`' output; the ONNX export passes them as
        constants so `linspace` is not traced.
        """
        rows, columns = (
            image.shape[2] // TOKEN_PIXELS,
            image.shape[3] // TOKEN_PIXELS,
        )
        encoder = self.encoder
        with torch.autocast(
            device_type=image.device.type, dtype=torch.float16, enabled=half
        ):
            normalized = (image - encoder.image_mean) / encoder.image_std
            layers = encoder.backbone.get_intermediate_layers(
                normalized, n=[5, 11, 17, 23], return_class_token=True
            )
            features = torch.stack(
                [
                    projection(
                        layer[0]
                        .permute(0, 2, 1)
                        .unflatten(2, (rows, columns))
                        .contiguous()
                    )
                    for projection, layer in zip(
                        encoder.output_projections, layers, strict=True
                    )
                ],
                dim=1,
            ).sum(dim=1)

        features = features.float()
        if uvGrids is None:
            uvGrids = makeUvGrids(
                rows, columns, aspectRatio, features.dtype, features.device
            )
        pyramid = [torch.cat([features, uvGrids[0]], dim=1), *uvGrids[1:]]

        neck = self.neck(pyramid)
        coords = self.points_head(neck, lastOnly=True)[-1]
        mask = self.mask_head(neck, lastOnly=True)[-1]
        return pyramid[0], coords, mask


def voxelBins(logDepth: Tensor) -> Tensor:
    """[H, W] fp32 log depth -> non-negative int64 bins."""
    quantized = torch.round(logDepth.float() * DEPTH_RESOLUTION).long()
    return quantized - quantized.min()


def refineStep(
    coords: Tensor,
    tokens: Tensor,
    refiner: Sparse3DUNet,
    tableDevice: torch.device | str | None = None,
) -> Tensor:
    """One refinement update of [1, H, W, 3] coords.

    tableDevice builds the voxel tables elsewhere than the network runs: MPS has
    no reliable int64 sort, so its tables are built on the CPU and moved over.
    """
    fastStep = getattr(refiner, "fastStep", None)
    if fastStep is not None:
        return fastStep(coords, tokens)
    height, width = coords.shape[1:3]
    bins = voxelBins(coords[0, :, :, 2])
    if tableDevice is None:
        levels = buildLevels(bins, refiner.numLevels)
    else:
        levels = buildLevels(bins.to(tableDevice), refiner.numLevels)
        levels = [level.to(coords.device) for level in levels]
    update = refiner(coords.reshape(-1, 3), levels, tokens)
    logDepth = coords[..., 2] + update.reshape(1, height, width)
    return torch.cat([coords[..., :2], logDepth.unsqueeze(-1)], dim=-1)


def refineCoords(
    rawCoords: Tensor,
    tokens: Tensor,
    refiner: Sparse3DUNet,
    steps: int = REFINE_STEPS,
    tableDevice: torch.device | str | None = None,
) -> Tensor:
    """[1, 3, H, W] raw coords -> [1, H, W, 3] after `steps` refinement updates."""
    coords = rawCoords.permute(0, 2, 3, 1).float()
    for _ in range(steps):
        coords = refineStep(coords, tokens, refiner, tableDevice)
    return coords


def _percentileBounds(disparity: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
    """2/98 percentiles of the valid disparity, on a grid thinned to fit
    `nanquantile`. Thinning before the NaN fill keeps it off the full map."""
    stride = max(1, int((disparity.numel() / PERCENTILE_SAMPLE_LIMIT) ** 0.5) + 1)
    disparity = disparity[..., ::stride, ::stride]
    sample = torch.where(valid[..., ::stride, ::stride], disparity, float("nan"))
    sample = sample.flatten()
    low, high = torch.nanquantile(sample, sample.new_tensor([0.02, 0.98]))
    return low, high


def coordsToGray(
    coords: Tensor, maskLogit: Tensor, outputHeight: int, outputWidth: int
) -> Tensor:
    """[1, H, W, 3] refined coords + mask -> [1, 1, outH, outW] disparity in [0, 1].

    Upstream's post-process at full output size (bilinear on the coords and on the
    logit, then exp, shift recovery, projection mask), then the same 1/depth and
    2/98 percentile stretch every ``*_v3``-style TAS depth method writes.

    Only log depth and the mask are needed at output size: x/y feed the shift
    solve alone, which reads a 64x64 nearest sample, so they are gathered there
    and dropped. Every value equals the full-map computation's.
    """
    size = (outputHeight, outputWidth)
    rows, columns = solveSamples(outputHeight, outputWidth, coords.device)
    planar = coords.permute(0, 3, 1, 2)
    xy = F.interpolate(planar[:, :2], size, mode="bilinear", align_corners=False)
    xyLow = xy[0][:, rows][:, :, columns].permute(1, 2, 0)
    del xy
    z = torch.exp(
        F.interpolate(planar[:, 2:], size, mode="bilinear", align_corners=False)[:, 0]
    )
    logit = F.interpolate(maskLogit.float(), size, mode="bilinear", align_corners=False)
    mask = logit[:, 0].sigmoid() > 0.5
    del logit

    zLow = z[0][rows][:, columns].unsqueeze(-1)
    u, v = viewPlaneAxes(outputWidth, outputHeight, dtype=z.dtype, device=z.device)
    uvLow = torch.stack(torch.meshgrid(u[columns], v[rows], indexing="xy"), dim=-1)
    shift = recoverShift(
        torch.cat([xyLow * zLow, zLow], dim=-1), uvLow, mask[0][rows][:, columns]
    )
    depth = z + shift
    del z
    valid = mask & (depth > 0)

    disparity = torch.where(valid, 1.0 / depth.clamp_min(1e-6), 0.0)
    del depth
    low, high = _percentileBounds(disparity, valid)
    gray = (disparity - low) / (high - low).clamp_min(1e-6)
    return torch.nan_to_num(gray, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)[None]


def halveEncoder(model: nn.Module) -> None:
    """Store the encoder's fp16-run weights in fp16 for a run autocasting to fp16.

    Autocast would otherwise keep an fp16 copy beside the fp32 weights, so Linear
    and Conv weights become those same fp16 values. What autocast keeps in fp32
    stays there: LayerNorm (it runs in fp32 under autocast), and the parameters
    that only feed the fp32 residual stream (pos_embed, cls_token, mask_token,
    LayerScale gamma). The output is unchanged.
    """
    backbone = model.encoder.backbone
    normNames = {
        f"{moduleName}.{leaf}" if moduleName else leaf
        for moduleName, module in backbone.named_modules()
        if isinstance(module, nn.LayerNorm)
        for leaf in ("weight", "bias")
    }
    keptFp32 = {
        name: parameter.detach().clone()
        for name, parameter in backbone.named_parameters()
        if name in ("pos_embed", "cls_token", "mask_token")
        or name.endswith("gamma")
        or name in normNames
    }
    backbone.half()
    model.encoder.output_projections.half()
    for name, value in keptFp32.items():
        module, _, leaf = name.rpartition(".")
        getattr(backbone.get_submodule(module), leaf).data = value


def loadRefiner(
    checkpointPath: str, device: torch.device | str = "cpu"
) -> Sparse3DUNet:
    """Only the refiner, for TensorRT: reads just its tensors from the file."""
    from safetensors import safe_open

    prefix = "refiner."
    with safe_open(checkpointPath, framework="pt", device="cpu") as handle:
        state = {
            key[len(prefix) :]: handle.get_tensor(key)
            for key in handle.keys()
            if key.startswith(prefix)
        }
    refiner = Sparse3DUNet()
    refiner.load_state_dict(state, strict=True)
    return refiner.float().eval().requires_grad_(False).to(device)


def loadMoGe3(
    checkpointPath: str,
    device: torch.device | str = "cpu",
    halfEncoder: bool = False,
) -> MoGe3:
    from safetensors.torch import load_file

    state = load_file(checkpointPath, device="cpu")
    model = MoGe3()
    result = model.load_state_dict(state, strict=False)
    unexpected = [
        k for k in result.unexpected_keys if not k.startswith(UNUSED_PREFIXES)
    ]
    if result.missing_keys or unexpected:
        raise ValueError(
            f"MoGe-3 checkpoint mismatch: missing={result.missing_keys[:8]}, "
            f"unexpected={unexpected[:8]}"
        )
    logging.info("Loaded MoGe-3 weights from %s", checkpointPath)
    model = model.float().eval().requires_grad_(False)
    if halfEncoder:
        halveEncoder(model)
    return model.to(device)
