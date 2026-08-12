"""Pure-PyTorch replacement for the upstream cupy correlation kernel.

Matches the original kernel: output channel c holds the channel-mean of
first * shift(second, (dx, dy)) with (dx, dy) = (c % 9 - 4, c // 9 - 4) and
zero padding outside the image. Inference-only; no backward.

The naive form is an 81-iteration python loop (162 kernel launches, ~3.5ms of
CPU issue time per call at the largest pyramid level — and the analysis stage
is CPU-bound, so launch overhead is wall time). Batching the 9 x-shifts per
row into one `unfold` view cuts it to 18 launches / ~0.4ms CPU and 3.6x the
GPU time. Values match the loop form to fp32 reduction-order noise (<= 1.2e-7
on randn inputs; bit-exact at the largest level), orders of magnitude under
the pipeline's own run-to-run nondeterminism.
"""

import torch
import torch.nn.functional as F


def FunctionCorrelation(tenFirst, tenSecond):
    B, C, H, W = tenFirst.shape
    padded = F.pad(tenSecond, (4, 4, 4, 4))
    first = tenFirst.unsqueeze(3)  # [B, C, H, 1, W]
    rows = []
    for dy in range(9):
        # [B, C, H, 9, W] view: [..., y, dx, x] = padded[..., y+dy, x+dx]
        view = padded.narrow(2, dy, H).unfold(3, W, 1)
        rows.append((first * view).mean(1))  # [B, H, 9, W]
    # channel c = dy*9 + dx, matching the original kernel's layout
    return torch.cat(rows, 2).permute(0, 2, 1, 3).reshape(B, 81, H, W)
