import torch


def grid_sample_directml(
    input_tensor: torch.Tensor,
    grid: torch.Tensor,
    padding_mode: str = "border",
    align_corners: bool = True,
) -> torch.Tensor:
    """
    DirectML-compatible grid_sample using bilinear interpolation.
    Simplified version optimized for DirectML.
    """
    B, C, H, W = input_tensor.shape
    _, H_out, W_out, _ = grid.shape

    # Extract coordinates
    grid_x = grid[..., 0]
    grid_y = grid[..., 1]

    # Denormalize
    if align_corners:
        pixel_x = ((grid_x + 1) / 2) * (W - 1)
        pixel_y = ((grid_y + 1) / 2) * (H - 1)
    else:
        pixel_x = ((grid_x + 1) * W - 1) / 2
        pixel_y = ((grid_y + 1) * H - 1) / 2

    # Find neighbors
    x0 = torch.floor(pixel_x)
    y0 = torch.floor(pixel_y)
    x1 = x0 + 1
    y1 = y0 + 1

    # Weights
    dx = pixel_x - x0
    dy = pixel_y - y0

    w00 = ((1 - dx) * (1 - dy)).unsqueeze(1)
    w01 = ((1 - dx) * dy).unsqueeze(1)
    w10 = (dx * (1 - dy)).unsqueeze(1)
    w11 = (dx * dy).unsqueeze(1)

    # Clamp and convert to indices
    x0 = torch.clamp(x0, 0, W - 1).long()
    x1 = torch.clamp(x1, 0, W - 1).long()
    y0 = torch.clamp(y0, 0, H - 1).long()
    y1 = torch.clamp(y1, 0, H - 1).long()

    # Compute indices
    idx00 = (y0 * W + x0).view(B, 1, H_out * W_out).expand(-1, C, -1)
    idx01 = (y1 * W + x0).view(B, 1, H_out * W_out).expand(-1, C, -1)
    idx10 = (y0 * W + x1).view(B, 1, H_out * W_out).expand(-1, C, -1)
    idx11 = (y1 * W + x1).view(B, 1, H_out * W_out).expand(-1, C, -1)

    # Flatten and gather
    input_flat = input_tensor.view(B, C, H * W)

    v00 = torch.gather(input_flat, 2, idx00).view(B, C, H_out, W_out)
    v01 = torch.gather(input_flat, 2, idx01).view(B, C, H_out, W_out)
    v10 = torch.gather(input_flat, 2, idx10).view(B, C, H_out, W_out)
    v11 = torch.gather(input_flat, 2, idx11).view(B, C, H_out, W_out)

    return w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11
