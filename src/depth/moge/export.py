"""MoGe-3 ONNX export for the TensorRT, DirectML and OpenVINO backends.

Two graphs, both exported once from the checkpoint into ``weights/moge3/onnx``:

- base: the fixed-shape half (`MoGe3.encode`) at one token grid. TensorRT,
  DirectML and OpenVINO all run it.
- refiner: one sparse U-Net update, with the voxel tables as inputs and every
  voxel count a dynamic axis. Only DirectML and OpenVINO need it, since TensorRT
  always has CUDA and runs the torch refiner.

Inactive taps and child slots point at one extra all-zero row (index M), which
turns the sparse conv and the mean pool into plain Gather + MatMul.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .model import MoGe3, makeUvGrids
from .sparse import KERNEL_TAPS, MAX_CHILDREN, Sparse3DUNet, VoxelLevel

EXPORT_VERSION = 1
OPSET = 17
BASE_INPUTS = ["image"]
BASE_OUTPUTS = ["tokens", "coords", "mask"]
# fp32 past the encoder under --half, matching the torch path.
_FP32_NODE_PREFIXES = ("/neck/", "/points_head/", "/mask_head/")


def refinerInputNames(numLevels: int = 5) -> list[str]:
    names = ["feats", "encoder", "encoderIndex"]
    names += [f"neighbors{k}" for k in range(numLevels)]
    for k in range(numLevels - 1):
        names += [f"parents{k}", f"children{k}", f"counts{k}"]
    return names


def refinerFeed(
    feats: np.ndarray,
    tokens: np.ndarray,
    levels: list[VoxelLevel],
) -> dict[str, np.ndarray]:
    """The numpy inputs of one refiner run, from CPU `buildLevels` tables."""
    _, channels, rows, columns = tokens.shape
    coarsest = levels[-1]
    feed: dict[str, np.ndarray] = {
        "feats": feats,
        "encoder": tokens.reshape(channels, rows * columns).T,
        "encoderIndex": (coarsest.row * columns + coarsest.col).cpu().numpy(),
    }
    for k, level in enumerate(levels):
        neighbors = level.neighbors.long()
        feed[f"neighbors{k}"] = (
            torch.where(neighbors < 0, level.count, neighbors).cpu().numpy()
        )
        if level.parents is None:
            continue
        parentCount = level.childCounts.shape[0]
        # Rank every child inside its parent to place it in a slot of the table.
        order = torch.argsort(level.parents, stable=True)
        sortedParents = level.parents[order]
        starts = torch.zeros_like(sortedParents)
        starts[1:] = (sortedParents[1:] != sortedParents[:-1]).long()
        groupStart = torch.nonzero(
            torch.cat([torch.ones(1).to(starts), starts[1:]])
        ).squeeze(1)
        slots = torch.arange(order.shape[0]) - groupStart[sortedParents]
        children = torch.full(
            (parentCount, MAX_CHILDREN), level.count, dtype=torch.long
        )
        children[sortedParents, slots] = order
        feed[f"parents{k}"] = level.parents.long().cpu().numpy()
        feed[f"children{k}"] = children.numpy()
        feed[f"counts{k}"] = level.childCounts[:, None].cpu().numpy()
    return feed


class BaseExport(nn.Module):
    def __init__(self, model: MoGe3, rows: int, columns: int, aspectRatio: float):
        super().__init__()
        self.model = model
        self.aspectRatio = aspectRatio
        self.uvGrids = makeUvGrids(rows, columns, aspectRatio)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return self.model.encode(image, self.aspectRatio, uvGrids=self.uvGrids)


class RefinerExport(nn.Module):
    """`Sparse3DUNet.forward` in padded-table form; same weights, same math."""

    def __init__(self, refiner: Sparse3DUNet):
        super().__init__()
        self.refiner = refiner

    @staticmethod
    def _withZeroRow(x: Tensor) -> Tensor:
        return torch.cat([x, torch.zeros(1, x.shape[1], dtype=x.dtype)], dim=0)

    def _conv(self, conv, feats: Tensor, neighbors: Tensor) -> Tensor:
        padded = self._withZeroRow(feats)
        weight = conv.weight.flatten(1, -2)
        output: Tensor | None = None
        for tap in range(KERNEL_TAPS):
            gathered = torch.index_select(padded, 0, neighbors[:, tap])
            term = F.linear(gathered, weight[:, tap])
            output = term if output is None else output + term
        return output + conv.bias

    def _block(self, block, feats: Tensor, neighbors: Tensor) -> Tensor:
        hidden = F.silu(block.norm1(feats))
        hidden = F.silu(self._conv(block.conv1, hidden, neighbors))
        return self._conv(block.conv2, hidden, neighbors) + feats

    def _pool(self, feats: Tensor, children: Tensor, counts: Tensor) -> Tensor:
        padded = self._withZeroRow(feats)
        total: Tensor | None = None
        for slot in range(MAX_CHILDREN):
            gathered = torch.index_select(padded, 0, children[:, slot])
            total = gathered if total is None else total + gathered
        return total / counts

    def forward(self, feats, encoder, encoderIndex, *tables):
        refiner = self.refiner
        depth = refiner.numLevels
        neighbors = tables[:depth]
        rest = tables[depth:]
        parents = rest[0::3]
        children = rest[1::3]
        counts = rest[2::3]

        feats = refiner.input_proj(feats)
        skips: list[Tensor] = []
        for k in range(depth):
            for block in refiner.down_stages[k]:
                feats = self._block(block, feats, neighbors[k])
            if k < depth - 1:
                skips.append(feats)
                pooled = self._pool(feats, children[k], counts[k])
                feats = refiner.downsample_blocks[k].linear(pooled)

        sampled = torch.index_select(encoder, 0, encoderIndex)
        feats = refiner.fuse_proj(
            torch.cat([feats, refiner.encoder_fuse(sampled)], dim=-1)
        )
        for block in refiner.bottleneck_stage:
            feats = self._block(block, feats, neighbors[-1])

        for i in range(depth - 1):
            target = depth - 2 - i
            upsampled = refiner.upsample_blocks[i].linear(feats)
            feats = torch.index_select(upsampled, 0, parents[target]) + skips[target]
            for block in refiner.up_stages[i]:
                feats = self._block(block, feats, neighbors[target])
        return refiner.out_proj(feats)


_DIGESTS: dict[tuple[str, int, int], str] = {}


def _digest(checkpoint: Path) -> str:
    """sha256 prefix of the checkpoint, hashed once per file version per run
    (DirectML asks for it twice, for the base and the refiner export)."""
    status = checkpoint.stat()
    key = (str(checkpoint.resolve()), status.st_size, status.st_mtime_ns)
    if key not in _DIGESTS:
        with checkpoint.open("rb") as source:
            _DIGESTS[key] = hashlib.file_digest(source, "sha256").hexdigest()[:16]
    return _DIGESTS[key]


def _dropDuplicateCasts(graph) -> None:
    """The fp16 converter gives a tensor that feeds two blocked nodes two Casts
    with one output name, which is invalid ONNX. They are identical; keep one."""
    seen: dict[str, tuple[str, ...]] = {}
    duplicates = []
    for node in graph.graph.node:
        for name in node.output:
            signature = (node.op_type, *node.input)
            if name not in seen:
                seen[name] = signature
                continue
            if node.op_type != "Cast" or seen[name] != signature:
                raise ValueError(f"Conflicting ONNX outputs named {name}")
            duplicates.append(node)
    for node in duplicates:
        graph.graph.node.remove(node)


def _toMixedFp16(src: str, dst: str) -> None:
    """fp16 encoder; neck and heads stay fp32. IO stays fp32 so drivers need no cast."""
    import onnx
    from onnxruntime.transformers import float16

    from src.depth.streaming_export import _strip_noop_float_casts, _topo_sort

    graph = onnx.load(src)
    _strip_noop_float_casts(graph)
    keepFp32 = [
        node.name
        for node in graph.graph.node
        if node.name.startswith(_FP32_NODE_PREFIXES)
    ]
    converted = float16.convert_float_to_float16(
        graph,
        keep_io_types=True,
        op_block_list=sorted(
            set(float16.DEFAULT_OP_BLOCK_LIST) | {"Resize", "LayerNormalization"}
        ),
        node_block_list=keepFp32,
        disable_shape_infer=False,
    )
    _dropDuplicateCasts(converted)
    _topo_sort(converted)
    del converted.graph.value_info[:]
    onnx.save(converted, dst)
    onnx.checker.check_model(dst, full_check=True)


def exportBase(
    checkpoint: str | Path,
    directory: str | Path,
    rows: int,
    columns: int,
    aspectRatio: float,
    half: bool,
) -> str:
    """The base graph for one token grid; cached by weights, shape and precision."""
    from .model import TOKEN_PIXELS, loadMoGe3

    checkpoint, directory = Path(checkpoint), Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # The UV grids are baked in, so the exact source aspect is part of the key.
    aspectKey = f"{aspectRatio:.6f}".replace(".", "p")
    destination = directory / (
        f"moge3_base_v{EXPORT_VERSION}_{_digest(checkpoint)}"
        f"_{columns}x{rows}_{aspectKey}_{'fp16' if half else 'fp32'}.onnx"
    )
    if destination.is_file():
        return str(destination)

    import onnx

    logging.info("Exporting MoGe-3 base ONNX to %s (first use only)", destination)
    model = loadMoGe3(str(checkpoint))
    wrapper = BaseExport(model, rows, columns, aspectRatio).eval()
    example = torch.rand(1, 3, rows * TOKEN_PIXELS, columns * TOKEN_PIXELS)
    with tempfile.TemporaryDirectory(prefix="moge-export-", dir=directory) as temporary:
        fp32 = str(Path(temporary) / "fp32.onnx")
        with torch.inference_mode():
            torch.onnx.export(
                wrapper,
                example,
                fp32,
                opset_version=OPSET,
                dynamo=False,
                input_names=BASE_INPUTS,
                output_names=BASE_OUTPUTS,
            )
        onnx.checker.check_model(fp32, full_check=True)
        result = fp32
        if half:
            result = str(Path(temporary) / "fp16.onnx")
            _toMixedFp16(fp32, result)
        os.replace(result, destination)
    return str(destination)


def exportRefiner(checkpoint: str | Path, directory: str | Path) -> str:
    """One refinement update; resolution independent, always fp32."""
    from .model import TOKEN_CHANNELS, loadMoGe3

    checkpoint, directory = Path(checkpoint), Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / (
        f"moge3_refiner_v{EXPORT_VERSION}_{_digest(checkpoint)}_fp32.onnx"
    )
    if destination.is_file():
        return str(destination)

    import onnx

    from .model import voxelBins
    from .sparse import buildLevels

    logging.info("Exporting MoGe-3 refiner ONNX to %s (first use only)", destination)
    wrapper = RefinerExport(loadMoGe3(str(checkpoint)).refiner).eval()
    numLevels = wrapper.refiner.numLevels

    # A small but real voxel layout: a slanted plane stacks voxels in a column
    # on coarse levels, so no table is degenerate while tracing.
    rows, columns = 8, 8
    height, width = rows * 16, columns * 16
    ramp = torch.linspace(0.0, 2.0, height)[None, :, None].expand(1, height, width)
    levels = buildLevels(voxelBins(ramp[0]), numLevels)
    feats = torch.rand(height * width, 3)
    tokens = torch.rand(1, TOKEN_CHANNELS, rows, columns)
    feed = refinerFeed(feats.numpy(), tokens.numpy(), levels)
    names = refinerInputNames(numLevels)
    example = tuple(torch.from_numpy(feed[name]) for name in names)

    dynamic: dict[str, dict[int, str]] = {
        "feats": {0: "voxels0"},
        "encoder": {0: "tokens"},
        "encoderIndex": {0: f"voxels{numLevels - 1}"},
        "update": {0: "voxels0"},
    }
    for k in range(numLevels):
        dynamic[f"neighbors{k}"] = {0: f"voxels{k}"}
    for k in range(numLevels - 1):
        dynamic[f"parents{k}"] = {0: f"voxels{k}"}
        dynamic[f"children{k}"] = {0: f"voxels{k + 1}"}
        dynamic[f"counts{k}"] = {0: f"voxels{k + 1}"}

    with tempfile.TemporaryDirectory(prefix="moge-export-", dir=directory) as temporary:
        fp32 = str(Path(temporary) / "fp32.onnx")
        with torch.inference_mode():
            torch.onnx.export(
                wrapper,
                example,
                fp32,
                opset_version=OPSET,
                dynamo=False,
                input_names=names,
                output_names=["update"],
                dynamic_axes=dynamic,
            )
        onnx.checker.check_model(fp32, full_check=True)
        os.replace(fp32, destination)
    return str(destination)
