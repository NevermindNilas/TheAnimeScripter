import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tenGrid = None
xNorm = None
yNorm = None


def warp(tenInput, tenFlow):
    global tenGrid, xNorm, yNorm
    if tenGrid is None or tenGrid.size() != tenFlow.size():
        N, _, H, W = tenFlow.shape
        tenHorizontal = (
            torch.linspace(-1.0, 1.0, W, device=device)
            .view(1, 1, 1, W)
            .expand(N, -1, H, -1)
        )
        tenVertical = (
            torch.linspace(-1.0, 1.0, H, device=device)
            .view(1, 1, H, 1)
            .expand(N, -1, -1, W)
        )
        tenGrid = torch.cat([tenHorizontal, tenVertical], 1)

    tenFlow = torch.cat(
        [
            tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
            tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0),
        ],
        1,
    )

    g = (tenGrid + tenFlow).permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(
        input=tenInput,
        grid=g,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
