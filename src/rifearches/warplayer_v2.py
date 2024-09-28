import torch

"""
Credit to HolyWU ( VS-Rife ) for many of the improvements seen
    https://github.com/HolyWu/vs-rife
"""


def warp(tenInput, tenFlow, tenFlowDiv, backWarp):
    tenFlow = torch.cat(
        [tenFlow[:, 0:1] / tenFlowDiv[0], tenFlow[:, 1:2] / tenFlowDiv[1]], 1
    )

    g = (backWarp + tenFlow).permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(
        input=tenInput,
        grid=g,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
