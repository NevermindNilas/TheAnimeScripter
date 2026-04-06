import torch

backwarp_tenGrid = {}


def get_backwarp_grid(tenFlow):
    key = (tenFlow.device, tenFlow.dtype, tenFlow.shape[2], tenFlow.shape[3])
    grid = backwarp_tenGrid.get(key)

    if grid is None:
        tenHorizontal = torch.linspace(
            -1.0,
            1.0,
            tenFlow.shape[3],
            dtype=tenFlow.dtype,
            device=tenFlow.device,
        ).view(1, 1, 1, tenFlow.shape[3])
        tenVertical = torch.linspace(
            -1.0,
            1.0,
            tenFlow.shape[2],
            dtype=tenFlow.dtype,
            device=tenFlow.device,
        ).view(1, 1, tenFlow.shape[2], 1)
        grid = torch.cat(
            [
                tenHorizontal.expand(1, -1, tenFlow.shape[2], -1),
                tenVertical.expand(1, -1, -1, tenFlow.shape[3]),
            ],
            1,
        )
        backwarp_tenGrid[key] = grid

    return grid.expand(tenFlow.shape[0], -1, -1, -1)


def warp(tenInput, tenFlow):
    orig_dtype = tenInput.dtype
    if tenInput.dtype != torch.float32:
        tenInput = tenInput.float()

    if tenFlow.dtype != torch.float32:
        tenFlow = tenFlow.float()

    tenFlow = torch.cat(
        [
            tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
            tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0),
        ],
        1,
    )

    grid = (get_backwarp_grid(tenFlow) + tenFlow).permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(
        input=tenInput,
        grid=grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True,
    ).to(orig_dtype)
