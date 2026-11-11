"""MoGe-3's sparse 3D refiner in plain torch (no Triton, no FlexGEMM).

The refiner works on voxels: every pixel of the coarse point map becomes one
voxel at (row, column, quantized log-depth), and a 5-level U-Net convolves them.
Which voxel neighbours which changes every frame and every refinement step, but
the network itself is ordinary once those relations are known. So they are built
up front as integer tables (`buildLevels`), and the network only gathers through
them:

- `Sparse3DUNet` is the torch forward, used by CUDA, ROCm, MPS and TensorRT.
- `src/depth/moge/export.py` traces the same weights to ONNX with the tables as
  graph inputs, for DirectML and OpenVINO.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn

KERNEL_SIZE = 3
# FlexGEMM flattens the (Co, 3, 3, 3, Ci) weight in itertools.product order and
# reads the input at coord + delta, so tap v's delta is this table's row v.
KERNEL_DELTAS = tuple(
    (di, dj, dz) for di in (-1, 0, 1) for dj in (-1, 0, 1) for dz in (-1, 0, 1)
)
KERNEL_TAPS = len(KERNEL_DELTAS)
MAX_CHILDREN = 8


@dataclass
class VoxelLevel:
    """One resolution of the voxel grid.

    neighbors: [M, 27] index of the voxel at each kernel tap, -1 when inactive.
    parents: [M] index of each voxel's cell one level coarser; None on the last.
    childCounts: [M_next] fp32 number of voxels merged into each coarser cell.
    row, col: [M] coordinates at this level.
    """

    count: int
    neighbors: Tensor
    parents: Tensor | None
    childCounts: Tensor | None
    row: Tensor
    col: Tensor
    _edges: list[tuple[Tensor, Tensor]] | None = field(default=None, repr=False)

    def to(self, device: torch.device | str) -> VoxelLevel:
        """Moved copy. The edge lists are built here first, on the table device,
        so MPS gets them without 27 syncing nonzero calls of its own. row/col are
        read only on the coarsest level (to sample the encoder grid), so the
        finer levels keep theirs where they are."""
        edges = self.edges()
        coarsest = self.parents is None
        return VoxelLevel(
            self.count,
            self.neighbors.to(device),
            None if coarsest else self.parents.to(device),
            None if coarsest else self.childCounts.to(device),
            self.row.to(device) if coarsest else self.row,
            self.col.to(device) if coarsest else self.col,
            [(d.to(device), s.to(device)) for d, s in edges],
        )

    def edges(self) -> list[tuple[Tensor, Tensor]]:
        """(destination, source) pairs per tap, so a conv touches active ones only."""
        if self._edges is None:
            if self.neighbors.device.type == "cpu":
                # Per tap is faster on the CPU than one big nonzero.
                edges = []
                for tap in range(KERNEL_TAPS):
                    column = self.neighbors[:, tap]
                    destination = torch.nonzero(column >= 0).squeeze(1)
                    edges.append((destination, column[destination]))
            else:
                # On a GPU each nonzero is a host sync: do one for all 27 taps.
                # Row-major over the transposed mask keeps each tap's
                # destinations ascending, exactly like the per-tap form.
                tapIndex, destination = torch.nonzero(
                    self.neighbors.T >= 0, as_tuple=True
                )
                counts = torch.bincount(tapIndex, minlength=KERNEL_TAPS).tolist()
                source = self.neighbors[destination, tapIndex]
                edges = list(
                    zip(
                        destination.split(counts),
                        source.split(counts),
                        strict=True,
                    )
                )
            self._edges = edges
        return self._edges


def _gridNeighbors(zIndex: Tensor) -> Tensor:
    """Level 0 has exactly one voxel per pixel, so no search is needed: the voxel
    at tap (di, dj, dz) exists iff pixel (i+di, j+dj) has z == z(i, j) + dz."""
    height, width = zIndex.shape
    device = zIndex.device
    zIndex = zIndex.to(torch.int32)
    indices = torch.arange(height * width, device=device, dtype=torch.int32)
    indices = indices.view(height, width)
    # z differences never reach this, so the padding ring never matches.
    outside = int(zIndex.max().item()) + 16
    zPadded = F.pad(zIndex, (1, 1, 1, 1), value=outside)
    indexPadded = F.pad(indices, (1, 1, 1, 1), value=-1)

    neighbors = torch.empty(
        (height * width, KERNEL_TAPS), dtype=torch.int32, device=device
    )
    tap = 0
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            zOther = zPadded[1 + di : height + 1 + di, 1 + dj : width + 1 + dj]
            indexOther = indexPadded[1 + di : height + 1 + di, 1 + dj : width + 1 + dj]
            difference = zOther - zIndex
            for dz in (-1, 0, 1):
                neighbors[:, tap] = torch.where(
                    difference == dz, indexOther, -1
                ).reshape(-1)
                tap += 1
    return neighbors


def _searchNeighbors(row: Tensor, col: Tensor, z: Tensor) -> Tensor:
    """Coarser levels can stack several voxels in a column, so look them up.

    The voxels come from `torch.unique`, so their keys are already sorted and
    unique in voxel order. A column (di, dj) then needs one search: its three
    z taps are consecutive keys, each found or not at the next position.
    """
    # The +1 and +3 keep tap offsets of -1 / +1 from wrapping into the next
    # row or column of the key.
    colSpan = int(col.max().item()) + 3
    zSpan = int(z.max().item()) + 3

    def encode(r: Tensor, c: Tensor, d: Tensor) -> Tensor:
        return ((r + 1) * colSpan + c + 1) * zSpan + d + 1

    keys = encode(row, col, z)
    last = keys.shape[0] - 1
    neighbors = torch.empty(
        (keys.shape[0], KERNEL_TAPS), dtype=torch.int32, device=row.device
    )
    tap = 0
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            query = keys + ((di * colSpan + dj) * zSpan - 1)
            position = torch.searchsorted(keys, query)
            for _ in (-1, 0, 1):
                clamped = position.clamp_max(last)
                found = keys[clamped] == query
                neighbors[:, tap] = torch.where(found, clamped, -1)
                position = position + found
                query = query + 1
                tap += 1
    return neighbors


def buildLevels(zIndex: Tensor, numLevels: int) -> list[VoxelLevel]:
    """Voxel tables for one frame.

    zIndex: [H, W] non-negative int64 depth bin of every pixel. Level 0 voxels
    are in row-major pixel order, so a refiner output reshapes straight back.
    Each coarser level halves all three axes (mean-pool of up to 8 children).
    Runs on zIndex's device; MPS callers pass a CPU tensor.
    """
    height, width = zIndex.shape
    device = zIndex.device
    row = torch.arange(height, device=device).view(-1, 1).expand(height, width)
    col = torch.arange(width, device=device).view(1, -1).expand(height, width)
    row, col, z = row.reshape(-1), col.reshape(-1), zIndex.reshape(-1)

    levels: list[VoxelLevel] = []
    for level in range(numLevels):
        neighbors = (
            _gridNeighbors(zIndex) if level == 0 else _searchNeighbors(row, col, z)
        )
        parents = childCounts = None
        if level < numLevels - 1:
            parentRow, parentCol, parentZ = row >> 1, col >> 1, z >> 1
            colSpan = int(parentCol.max().item()) + 1
            zSpan = int(parentZ.max().item()) + 1
            key = (parentRow * colSpan + parentCol) * zSpan + parentZ
            uniqueKeys, parents = torch.unique(key, return_inverse=True)
            childCounts = torch.bincount(parents, minlength=uniqueKeys.shape[0])
            childCounts = childCounts.to(torch.float32)
        levels.append(
            VoxelLevel(row.shape[0], neighbors, parents, childCounts, row, col)
        )
        if parents is not None:
            cell = uniqueKeys // zSpan
            z = uniqueKeys % zSpan
            row, col = cell // colSpan, cell % colSpan
    return levels


class SubmanifoldConv3d(nn.Module):
    """3x3x3 convolution whose output lives on the input voxels."""

    def __init__(self, inChannels: int, outChannels: int):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(outChannels, KERNEL_SIZE, KERNEL_SIZE, KERNEL_SIZE, inChannels)
        )
        self.bias = nn.Parameter(torch.empty(outChannels))

    def forward(self, feats: Tensor, level: VoxelLevel) -> Tensor:
        weight = self.weight.flatten(1, -2)
        output = feats.new_zeros(feats.shape[0], self.weight.shape[0])
        for tap, (destination, source) in enumerate(level.edges()):
            if destination.numel():
                output.index_add_(
                    0, destination, F.linear(feats[source], weight[:, tap])
                )
        return output + self.bias


class SparseResBlock3d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels, eps=1e-6)
        self.conv1 = SubmanifoldConv3d(channels, channels)
        self.conv2 = SubmanifoldConv3d(channels, channels)

    def forward(self, feats: Tensor, level: VoxelLevel) -> Tensor:
        hidden = F.silu(self.norm1(feats))
        hidden = F.silu(self.conv1(hidden, level))
        return self.conv2(hidden, level) + feats


class PoolDown(nn.Module):
    def __init__(self, inChannels: int, outChannels: int):
        super().__init__()
        self.linear = nn.Linear(inChannels, outChannels)


class NearestUp(nn.Module):
    def __init__(self, inChannels: int, outChannels: int):
        super().__init__()
        self.linear = nn.Linear(inChannels, outChannels)


class Sparse3DUNet(nn.Module):
    """Upstream's Sparse3DUNet with the same parameter names."""

    def __init__(
        self,
        inChannels: int = 3,
        outChannels: int = 1,
        encoderChannels: int = 1026,
        modelChannels: tuple[int, ...] = (32, 64, 128, 256, 512),
    ):
        super().__init__()
        depth = len(modelChannels)
        self.numLevels = depth
        self.input_proj = nn.Linear(inChannels, modelChannels[0])
        self.encoder_fuse = nn.Linear(encoderChannels, modelChannels[-1])
        self.fuse_proj = nn.Sequential(
            nn.Linear(modelChannels[-1] * 2, modelChannels[-1]),
            nn.SiLU(),
            nn.Linear(modelChannels[-1], modelChannels[-1]),
        )
        self.down_stages = nn.ModuleList(
            nn.ModuleList([SparseResBlock3d(channels)]) for channels in modelChannels
        )
        self.downsample_blocks = nn.ModuleList(
            PoolDown(modelChannels[i], modelChannels[i + 1]) for i in range(depth - 1)
        )
        self.bottleneck_stage = nn.ModuleList([SparseResBlock3d(modelChannels[-1])])
        self.upsample_blocks = nn.ModuleList(
            NearestUp(modelChannels[depth - 1 - i], modelChannels[depth - 2 - i])
            for i in range(depth - 1)
        )
        self.up_stages = nn.ModuleList(
            nn.ModuleList([SparseResBlock3d(modelChannels[depth - 2 - i])])
            for i in range(depth - 1)
        )
        self.out_proj = nn.Linear(modelChannels[0], outChannels)

    def forward(
        self, feats: Tensor, levels: list[VoxelLevel], encoderFeature: Tensor
    ) -> Tensor:
        """feats: [M0, 3]. encoderFeature: [1, C, H/16, W/16]. Returns [M0, 1]."""
        feats = self.input_proj(feats)
        depth = self.numLevels
        skips: list[Tensor] = []
        for k in range(depth):
            for block in self.down_stages[k]:
                feats = block(feats, levels[k])
            if k < depth - 1:
                skips.append(feats)
                current = levels[k]
                pooled = feats.new_zeros(current.childCounts.shape[0], feats.shape[1])
                pooled.index_add_(0, current.parents, feats)
                feats = self.downsample_blocks[k].linear(
                    pooled / current.childCounts[:, None]
                )

        coarsest = levels[-1]
        sampled = encoderFeature[0, :, coarsest.row, coarsest.col].permute(1, 0)
        feats = self.fuse_proj(torch.cat([feats, self.encoder_fuse(sampled)], dim=-1))
        for block in self.bottleneck_stage:
            feats = block(feats, coarsest)

        for i in range(depth - 1):
            target = depth - 2 - i
            feats = self.upsample_blocks[i].linear(feats)[levels[target].parents]
            feats = feats + skips.pop()
            for block in self.up_stages[i]:
                feats = block(feats, levels[target])
        return self.out_proj(feats)
