import torch

tenGrid = None
multiply = None

# Inspo from Rife V2
def warp(tenInput, tenFlow):
    global tenGrid, multiply
    if tenGrid is None:
        device = tenInput.device
        hMul = 2 / (tenInput.shape[3] - 1)
        vMul = 2 / (tenInput.shape[2] - 1)
        multiply = torch.tensor(
            [hMul, vMul], dtype=torch.float32, device=device
        ).reshape(1, 2, 1, 1)

        if tenInput.dtype == torch.float16:
            multiply = multiply.half()

        tenHorizontal = (
            (torch.arange(tenInput.shape[3], device=device) * hMul - 1)
            .reshape(1, 1, 1, -1)
            .expand(-1, -1, tenInput.shape[2], -1)
        )
        tenVertical = (
            (torch.arange(tenInput.shape[2], device=device) * vMul - 1)
            .reshape(1, 1, -1, 1)
            .expand(-1, -1, -1, tenInput.shape[3])
        )
        tenGrid = torch.cat((tenHorizontal, tenVertical), 1)

        if tenInput.dtype == torch.float16:
            tenGrid = tenGrid.half()

    tenFlow = tenFlow * multiply
    g = (tenGrid + tenFlow).permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(
        input=tenInput,
        grid=g,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
