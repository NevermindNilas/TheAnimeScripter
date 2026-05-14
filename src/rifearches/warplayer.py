import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tenGrid = {}
tenFlowDiv = {}  # precomputed inverse divisors [2/(W-1), 2/(H-1)] per (device, size, dtype)


def warp(tenInput, tenFlow):
    k = (str(tenFlow.device), str(tenFlow.size()), str(tenFlow.dtype))
    if k not in tenGrid:
        H, W = tenFlow.shape[2], tenFlow.shape[3]
        tenHorizontal = (
            torch.linspace(-1.0, 1.0, W, device=tenFlow.device, dtype=torch.float32)
            .view(1, 1, 1, W)
            .expand(tenFlow.shape[0], -1, H, -1)
        )
        tenVertical = (
            torch.linspace(-1.0, 1.0, H, device=tenFlow.device, dtype=torch.float32)
            .view(1, 1, H, 1)
            .expand(tenFlow.shape[0], -1, -1, W)
        )
        tenGrid[k] = torch.cat([tenHorizontal, tenVertical], 1).to(tenFlow.dtype)
        # divisor key uses INPUT spatial (which equals flow spatial in this codebase's usage)
        tenFlowDiv[k] = torch.tensor(
            [2.0 / (W - 1), 2.0 / (H - 1)],
            dtype=tenFlow.dtype,
            device=tenFlow.device,
        ).view(1, 2, 1, 1)

    g = (tenGrid[k] + tenFlow * tenFlowDiv[k]).permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(
        input=tenInput,
        grid=g,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
