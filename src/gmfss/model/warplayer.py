import torch

backwarpTenGrid = None


def warp(tenInput, tenFlow):
    global backwarpTenGrid
    origDtype = tenInput.dtype
    if tenInput.dtype != torch.float32:
        tenInput = tenInput.float()

    if tenFlow.dtype != torch.float32:
        tenFlow = tenFlow.float()

    if backwarpTenGrid is None:
        tenHorizontal = (
            torch.linspace(
                -1.0, 1.0, tenFlow.shape[3], dtype=torch.float, device=tenFlow.device
            )
            .view(1, 1, 1, tenFlow.shape[3])
            .expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        )
        tenVertical = (
            torch.linspace(
                -1.0, 1.0, tenFlow.shape[2], dtype=torch.float, device=tenFlow.device
            )
            .view(1, 1, tenFlow.shape[2], 1)
            .expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
        )
        backwarpTenGrid = torch.cat([tenHorizontal, tenVertical], 1)

    tenFlow = torch.cat(
        [
            tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
            tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0),
        ],
        1,
    )

    g = (backwarpTenGrid + tenFlow).permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(
        input=tenInput,
        grid=g,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    ).to(origDtype)
