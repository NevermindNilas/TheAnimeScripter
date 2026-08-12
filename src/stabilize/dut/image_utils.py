"""RFDet score-map helpers, pruned to the detection-only inference path."""

import torch
import torch.nn.functional as F


def L2Norm(input, dim=-1):
    input = input / torch.norm(input, p=2, dim=dim, keepdim=True).clamp(min=1e-8)
    return input


def filtbordmask(imscore, radius):
    bs, height, width, c = imscore.size()
    mask = imscore.new_full(
        (1, height - 2 * radius, width - 2 * radius, 1), fill_value=1
    )
    mask = F.pad(
        input=mask,
        pad=(0, 0, radius, radius, radius, radius, 0, 0),
        mode="constant",
        value=0,
    )
    return mask


def filter_border(imscore, radius=8):
    imscore = imscore * filtbordmask(imscore, radius=radius)
    return imscore


def nms(input, thresh=0.0, ksize=5):
    """
    non maximum depression in each pixel if it is not maximum probability in its ksize*ksize range
    :param input: (B, H, W, 1)

    Upstream built the window maximum by concatenating ksize*ksize shifted
    slices; a single max_pool2d is bit-identical (the threshold step makes the
    input non-negative, so upstream's zero padding and max_pool's -inf padding
    yield the same window maxima) and skips 25 tensor copies per call.
    """
    pad = ksize // 2
    zeros = torch.zeros_like(input)
    input = torch.where(input < thresh, zeros, input)
    nchw = input.permute(0, 3, 1, 2)
    window_max = F.max_pool2d(nchw, kernel_size=ksize, stride=1, padding=pad)
    mask = torch.ge(nchw, window_max).permute(0, 2, 3, 1)
    return mask.type_as(input)


def topk_map(maps, k=512):
    """
    find the top k maximum pixel probability in a maps
    :param maps: (B, H, W, 1)
    """
    batch, height, width, _ = maps.size()
    maps_flat = maps.view(batch, -1)

    indices = maps_flat.sort(dim=-1, descending=True)[1][:, :k]
    batch_idx = (
        torch.arange(0, batch, dtype=indices.dtype, device=indices.device)
        .unsqueeze(-1)
        .repeat(1, k)
    )

    topk_mask_flat = torch.zeros(
        maps_flat.size(), dtype=torch.uint8, device=maps.device
    )
    topk_mask_flat[batch_idx.view(-1), indices.contiguous().view(-1)] = 1

    mask = topk_mask_flat.view(batch, height, width, -1)
    return mask


def get_gauss_filter_weight(ksize, sig):
    mu_x = mu_y = ksize // 2
    if sig == 0:
        psf = torch.zeros((ksize, ksize)).float()
        psf[mu_y, mu_x] = 1.0
    else:
        sig = torch.tensor(sig).float()
        x = torch.arange(ksize)[None, :].repeat(ksize, 1).float()
        y = torch.arange(ksize)[:, None].repeat(1, ksize).float()
        psf = torch.exp(
            -((x - mu_x) ** 2 / (2 * sig**2) + (y - mu_y) ** 2 / (2 * sig**2))
        )
    return psf


def soft_nms_3d(scale_logits, ksize, com_strength):
    """
    calculate probability for each pixel in each scale space
    :param scale_logits: (B, H, W, C)
    """
    num_scales = scale_logits.size(-1)

    max_each_scale = F.max_pool2d(
        input=scale_logits.permute(0, 3, 1, 2),
        kernel_size=ksize,
        padding=ksize // 2,
        stride=1,
    ).permute(0, 2, 3, 1)  # (B, H, W, C)
    max_all_scale, max_all_scale_idx = max_each_scale.max(dim=-1, keepdim=True)
    exp_maps = torch.exp(com_strength * (scale_logits - max_all_scale))
    sum_exp = F.conv2d(
        input=exp_maps.permute(0, 3, 1, 2).contiguous(),
        weight=exp_maps.new_full(
            [1, num_scales, ksize, ksize], fill_value=1
        ).contiguous(),
        stride=1,
        padding=ksize // 2,
    ).permute(0, 2, 3, 1)
    probs = exp_maps / (sum_exp + 1e-8)
    return probs


def soft_max_and_argmax_1d(
    input, orint_maps, scale_list, com_strength1, com_strength2, dim=-1, keepdim=True
):
    """
    calculate the final pixel probability summary from all scale and each pixel correspond scale
    :param input: scale_probs(B, H, W, 10)
    :param orint_maps: (B, H, W, 10, 2)
    """
    inputs_exp1 = torch.exp(
        com_strength1 * (input - torch.max(input, dim=dim, keepdim=True)[0])
    )
    input_softmax1 = inputs_exp1 / (inputs_exp1.sum(dim=dim, keepdim=True) + 1e-8)

    inputs_exp2 = torch.exp(
        com_strength2 * (input - torch.max(input, dim=dim, keepdim=True)[0])
    )
    input_softmax2 = inputs_exp2 / (inputs_exp2.sum(dim=dim, keepdim=True) + 1e-8)

    score_map = torch.sum(input * input_softmax1, dim=dim, keepdim=keepdim)

    scale_list_shape = [1] * len(input.size())
    scale_list_shape[dim] = -1
    scale_list = scale_list.view(scale_list_shape).to(input_softmax2.device)
    scale_map = torch.sum(scale_list * input_softmax2, dim=dim, keepdim=keepdim)

    if orint_maps is not None:
        orint_map = torch.sum(
            orint_maps * input_softmax1.unsqueeze(-1), dim=dim - 1, keepdim=keepdim
        )  # (B, H, W, 1, 2)
        orint_map = L2Norm(orint_map, dim=-1)
        return score_map, scale_map, orint_map
    else:
        return score_map, scale_map
