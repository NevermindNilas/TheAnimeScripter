import torch
import torch.nn as nn


device = "cuda" if torch.cuda.is_available() else "cpu"

grid_cache = {}
base_index_cache = {}
out_cache = {}

torch.set_float32_matmul_precision("medium")
torch.set_grad_enabled(False)


def _parse_mode(strMode: str):
    mode_parts = strMode.split("-")
    raw_mode = mode_parts[0]
    mode_main = {
        "summation": "sum",
        "average": "avg",
        "softmax": "soft",
    }.get(raw_mode, raw_mode)
    mode_sub = mode_parts[1] if len(mode_parts) > 1 else None

    if mode_sub is None and raw_mode in {"average", "linear", "softmax"}:
        mode_sub = "zeroeps"

    return mode_main, mode_sub


def _prepare_input(
    tenIn: torch.Tensor,
    tenMetric: torch.Tensor | None,
    mode_main: str,
):
    if mode_main == "avg":
        return torch.cat(
            [
                tenIn,
                tenIn.new_ones((tenIn.shape[0], 1, tenIn.shape[2], tenIn.shape[3])),
            ],
            1,
        )

    if mode_main == "linear":
        return torch.cat([tenIn * tenMetric, tenMetric], 1)

    if mode_main == "soft":
        metric_exp = tenMetric.exp()
        return torch.cat([tenIn * metric_exp, metric_exp], 1)

    return tenIn


def _normalize_output(tenOut: torch.Tensor, mode_sub: str | None):
    tenNormalize = tenOut[:, -1:, :, :]

    if mode_sub in (None, "addeps"):
        tenNormalize = tenNormalize + 0.0000001
    elif mode_sub == "zeroeps":
        tenNormalize = torch.where(
            tenNormalize == 0.0,
            torch.ones_like(tenNormalize),
            tenNormalize,
        )
    elif mode_sub == "clipeps":
        tenNormalize = tenNormalize.clip(0.0000001, None)

    return tenOut[:, :-1, :, :] / tenNormalize


def _get_grid(height: int, width: int, dev: torch.device, dtype: torch.dtype):
    key = (height, width, dev, dtype)
    cached = grid_cache.get(key)

    if cached is None:
        grid_y, grid_x = torch.meshgrid(
            torch.arange(height, device=dev, dtype=dtype),
            torch.arange(width, device=dev, dtype=dtype),
            indexing="ij",
        )
        cached = (grid_y.view(1, 1, height, width), grid_x.view(1, 1, height, width))
        grid_cache[key] = cached

    return cached


def _get_base_index(n: int, height: int, width: int, dev: torch.device):
    key = (n, height, width, dev)
    cached = base_index_cache.get(key)

    if cached is None:
        cached = torch.arange(n, device=dev, dtype=torch.int64)
        cached = cached.repeat_interleave(height * width) * (height * width)
        base_index_cache[key] = cached

    return cached


@torch.inference_mode()
def softsplat(
    tenIn: torch.Tensor,
    tenFlow: torch.Tensor,
    tenMetric: torch.Tensor | None,
    strMode: str,
):
    mode_main, mode_sub = _parse_mode(strMode)

    if mode_main not in {"sum", "avg", "linear", "soft"}:
        raise ValueError(f"Unsupported softsplat mode: {strMode}")
    if mode_main in {"sum", "avg"} and tenMetric is not None:
        raise ValueError(f"softsplat mode '{strMode}' does not accept a metric tensor")
    if mode_main in {"linear", "soft"} and tenMetric is None:
        raise ValueError(f"softsplat mode '{strMode}' requires a metric tensor")
    if tenFlow.shape[1] != 2:
        raise ValueError(
            f"softsplat expects flow shaped [N, 2, H, W], got {tuple(tenFlow.shape)}"
        )
    if tenIn.shape[0] != tenFlow.shape[0] or tenIn.shape[2:] != tenFlow.shape[2:]:
        raise ValueError(
            f"softsplat requires matching input and flow shapes, got input {tuple(tenIn.shape)} and flow {tuple(tenFlow.shape)}"
        )
    if tenMetric is not None and tenMetric.shape[0] != tenIn.shape[0]:
        raise ValueError(
            f"softsplat requires matching input and metric batch sizes, got input {tuple(tenIn.shape)} and metric {tuple(tenMetric.shape)}"
        )
    if tenMetric is not None and tenMetric.shape[2:] != tenIn.shape[2:]:
        raise ValueError(
            f"softsplat requires matching input and metric spatial sizes, got input {tuple(tenIn.shape)} and metric {tuple(tenMetric.shape)}"
        )

    tenPrepared = _prepare_input(tenIn, tenMetric, mode_main)
    tenOut = softsplat_func.apply(tenPrepared, tenFlow)

    if mode_main != "sum":
        tenOut = _normalize_output(tenOut, mode_sub)

    return tenOut


class softsplat_func(torch.autograd.Function):
    @staticmethod
    @torch.inference_mode()
    @torch.amp.custom_fwd(device_type=device)
    def forward(ctx, tenIn: torch.Tensor, tenFlow: torch.Tensor):
        n, channels, height, width = tenIn.shape
        dev = tenIn.device
        dtype = tenIn.dtype

        grid_y, grid_x = _get_grid(height, width, dev, dtype)
        flow_x = grid_x + tenFlow[:, 0:1]
        flow_y = grid_y + tenFlow[:, 1:2]

        finite_mask = torch.isfinite(flow_x) & torch.isfinite(flow_y)
        safe_flow_x = torch.where(finite_mask, flow_x, torch.zeros_like(flow_x))
        safe_flow_y = torch.where(finite_mask, flow_y, torch.zeros_like(flow_y))

        x0 = torch.floor(safe_flow_x)
        y0 = torch.floor(safe_flow_y)
        x1 = x0 + 1
        y1 = y0 + 1

        x0l = x0.to(torch.int64)
        y0l = y0.to(torch.int64)
        x1l = x1.to(torch.int64)
        y1l = y1.to(torch.int64)

        w00 = (x1 - safe_flow_x) * (y1 - safe_flow_y)
        w10 = (safe_flow_x - x0) * (y1 - safe_flow_y)
        w01 = (x1 - safe_flow_x) * (safe_flow_y - y0)
        w11 = (safe_flow_x - x0) * (safe_flow_y - y0)

        m00 = finite_mask & (x0l >= 0) & (x0l < width) & (y0l >= 0) & (y0l < height)
        m10 = finite_mask & (x1l >= 0) & (x1l < width) & (y0l >= 0) & (y0l < height)
        m01 = finite_mask & (x0l >= 0) & (x0l < width) & (y1l >= 0) & (y1l < height)
        m11 = finite_mask & (x1l >= 0) & (x1l < width) & (y1l >= 0) & (y1l < height)

        x0c = x0l.clamp(0, width - 1)
        y0c = y0l.clamp(0, height - 1)
        x1c = x1l.clamp(0, width - 1)
        y1c = y1l.clamp(0, height - 1)

        base = _get_base_index(n, height, width, dev)
        feats_flat = tenIn.permute(0, 2, 3, 1).reshape(-1, channels)

        idx00 = base + (y0c.reshape(-1) * width) + x0c.reshape(-1)
        idx10 = base + (y0c.reshape(-1) * width) + x1c.reshape(-1)
        idx01 = base + (y1c.reshape(-1) * width) + x0c.reshape(-1)
        idx11 = base + (y1c.reshape(-1) * width) + x1c.reshape(-1)

        w00 = (w00 * m00.to(dtype)).reshape(-1, 1)
        w10 = (w10 * m10.to(dtype)).reshape(-1, 1)
        w01 = (w01 * m01.to(dtype)).reshape(-1, 1)
        w11 = (w11 * m11.to(dtype)).reshape(-1, 1)

        indices = torch.cat((idx00, idx10, idx01, idx11), 0)
        values = torch.cat(
            (
                feats_flat * w00,
                feats_flat * w10,
                feats_flat * w01,
                feats_flat * w11,
            ),
            0,
        )

        cache_key = (n, channels, height, width, dev, dtype)
        out = out_cache.get(cache_key)
        if out is None or out.numel() != n * height * width * channels:
            out = torch.zeros(n * height * width, channels, device=dev, dtype=dtype)
            out_cache[cache_key] = out
        else:
            out.zero_()

        out.index_add_(0, indices, values)
        return out.view(n, height, width, channels).permute(0, 3, 1, 2)


def FunctionSoftsplat(
    tenInput: torch.Tensor,
    tenFlow: torch.Tensor,
    tenMetric: torch.Tensor | None,
    strType: str,
):
    strMode = {
        "summation": "sum",
        "average": "avg-zeroeps",
        "linear": "linear-zeroeps",
        "softmax": "soft-zeroeps",
    }.get(strType, strType)
    return softsplat(tenInput, tenFlow, tenMetric, strMode)


class Softsplat(nn.Module):
    def __init__(self, strType: str = "softmax"):
        super().__init__()
        self.strType = strType

    def forward(self, img, flow, z):
        return FunctionSoftsplat(img, flow, z, self.strType)
