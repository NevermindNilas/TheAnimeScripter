import torch

tenGrid = None
multiply = None

# Inspo from Rife V2
def warp(tenInput, tenFlow):
    global tenGrid, multiply
    if tenGrid is None:
        hMul = 2 / (tenInput.shape[3] - 1)
        vMul = 2 / (tenInput.shape[2] - 1)
        multiply = torch.tensor(
            [hMul, vMul], dtype=tenInput.dtype, device=tenInput.device
        ).reshape(1, 2, 1, 1)

        tenHorizontal = (
            (torch.arange(tenInput.shape[3], device=tenInput.device) * hMul - 1)
            .reshape(1, 1, 1, -1)
            .expand(-1, -1, tenInput.shape[2], -1)
        )
        tenVertical = (
            (torch.arange(tenInput.shape[2], device=tenInput.device) * vMul - 1)
            .reshape(1, 1, -1, 1)
            .expand(-1, -1, -1, tenInput.shape[3])
        )
        tenGrid = torch.cat((tenHorizontal, tenVertical), 1).to(tenInput.device).to(tenInput.dtype)

    tenFlow = tenFlow * multiply
    g = (tenGrid + tenFlow).permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(
        input=tenInput,
        grid=g,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
 