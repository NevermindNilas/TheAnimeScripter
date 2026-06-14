import torch
import torch.nn.functional as F


coords_grid_cache = {}
window_grid_cache = {}
normalize_coords_cache = {}


def coords_grid(b, h, w, homogeneous=False, device=None, dtype=torch.float32):
    key = (device, dtype, h, w, homogeneous)
    grid = coords_grid_cache.get(key)

    if grid is None:
        y, x = torch.meshgrid(
            torch.arange(h, device=device, dtype=dtype),
            torch.arange(w, device=device, dtype=dtype),
            indexing="ij",
        )  # [H, W]

        stacks = [x, y]

        if homogeneous:
            ones = torch.ones_like(x)  # [H, W]
            stacks.append(ones)

        grid = torch.stack(stacks, dim=0).unsqueeze(0)  # [1, 2, H, W] or [1, 3, H, W]
        coords_grid_cache[key] = grid

    return grid.expand(b, -1, -1, -1)


def generate_window_grid(
    h_min, h_max, w_min, w_max, len_h, len_w, device=None, dtype=torch.float32
):
    assert device is not None

    key = (device, dtype, h_min, h_max, w_min, w_max, len_h, len_w)
    grid = window_grid_cache.get(key)

    if grid is None:
        x, y = torch.meshgrid(
            [
                torch.linspace(w_min, w_max, len_w, device=device, dtype=dtype),
                torch.linspace(h_min, h_max, len_h, device=device, dtype=dtype),
            ],
            indexing="ij",
        )
        grid = torch.stack((x, y), -1).transpose(0, 1)  # [H, W, 2]
        window_grid_cache[key] = grid

    return grid


def normalize_coords(coords, h, w):
    # coords: [B, H, W, 2]
    key = (coords.device, coords.dtype, h, w)
    c = normalize_coords_cache.get(key)

    if c is None:
        c = torch.tensor(
            [(w - 1) / 2.0, (h - 1) / 2.0],
            dtype=coords.dtype,
            device=coords.device,
        )
        normalize_coords_cache[key] = c

    return (coords - c) / c  # [-1, 1]


def bilinear_sample(
    img, sample_coords, mode="bilinear", padding_mode="zeros", return_mask=False
):
    # img: [B, C, H, W]
    # sample_coords: [B, 2, H, W] in image scale
    if sample_coords.size(1) != 2:  # [B, H, W, 2]
        sample_coords = sample_coords.permute(0, 3, 1, 2)

    b, _, h, w = sample_coords.shape

    # Normalize to [-1, 1]
    x_grid = 2 * sample_coords[:, 0] / (w - 1) - 1
    y_grid = 2 * sample_coords[:, 1] / (h - 1) - 1

    grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, H, W, 2]

    img = F.grid_sample(
        img, grid, mode=mode, padding_mode=padding_mode, align_corners=True
    )

    if return_mask:
        mask = (
            (x_grid >= -1) & (y_grid >= -1) & (x_grid <= 1) & (y_grid <= 1)
        )  # [B, H, W]

        return img, mask

    return img


def flow_warp(feature, flow, mask=False, padding_mode="zeros"):
    b, c, h, w = feature.size()
    assert flow.size(1) == 2

    grid = (
        coords_grid(b, h, w, device=flow.device, dtype=flow.dtype) + flow
    )  # [B, 2, H, W]

    return bilinear_sample(feature, grid, padding_mode=padding_mode, return_mask=mask)


def forward_backward_consistency_check(fwd_flow, bwd_flow, alpha=0.01, beta=0.5):
    # fwd_flow, bwd_flow: [B, 2, H, W]
    # alpha and beta values are following UnFlow (https://arxiv.org/abs/1711.07837)
    assert fwd_flow.dim() == 4 and bwd_flow.dim() == 4
    assert fwd_flow.size(1) == 2 and bwd_flow.size(1) == 2

    b = fwd_flow.shape[0]

    # Batched flow_warp: warp [bwd, fwd] with grid from [fwd, bwd].
    inp = torch.cat([bwd_flow, fwd_flow], dim=0)
    grid_flow = torch.cat([fwd_flow, bwd_flow], dim=0)
    warped = flow_warp(inp, grid_flow)
    warped_bwd_flow, warped_fwd_flow = warped[:b], warped[b:]

    # Batched norm over the two flows.
    mag = torch.linalg.vector_norm(torch.stack([fwd_flow, bwd_flow], dim=0), dim=2)
    flow_mag = mag[0] + mag[1]

    diffs = torch.linalg.vector_norm(
        torch.stack([fwd_flow + warped_bwd_flow, bwd_flow + warped_fwd_flow], dim=0),
        dim=2,
    )
    diff_fwd, diff_bwd = diffs[0], diffs[1]

    threshold = alpha * flow_mag + beta

    fwd_occ = (diff_fwd > threshold).to(fwd_flow.dtype)
    bwd_occ = (diff_bwd > threshold).to(bwd_flow.dtype)

    return fwd_occ, bwd_occ
