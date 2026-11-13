"""Depth recovery from MoGe's affine point map, without SciPy."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

# Upstream solves on this nearest-sampled grid.
SOLVE_SIZE = (64, 64)
MAX_ITERATIONS = 100


def viewPlaneAxes(
    width: int,
    height: int,
    aspectRatio: float | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor]:
    """The u (per column) and v (per row) values `viewPlaneUv` is the grid of."""
    if aspectRatio is None:
        aspectRatio = width / height
    spanX = aspectRatio / (1 + aspectRatio**2) ** 0.5
    spanY = 1 / (1 + aspectRatio**2) ** 0.5
    u = torch.linspace(
        -spanX * (width - 1) / width,
        spanX * (width - 1) / width,
        width,
        dtype=dtype,
        device=device,
    )
    v = torch.linspace(
        -spanY * (height - 1) / height,
        spanY * (height - 1) / height,
        height,
        dtype=dtype,
        device=device,
    )
    return u, v


def viewPlaneUv(
    width: int,
    height: int,
    aspectRatio: float | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """[H, W, 2] grid; corners at +-(width, height) / diagonal."""
    u, v = viewPlaneAxes(width, height, aspectRatio, dtype, device)
    u, v = torch.meshgrid(u, v, indexing="xy")
    return torch.stack([u, v], dim=-1)


def _projectionResidual(
    shift: float, uv: np.ndarray, xy: np.ndarray, z: np.ndarray
) -> np.ndarray:
    projected = xy / (z + shift)[:, None]
    focal = (projected * uv).sum() / np.square(projected).sum()
    return (focal * projected - uv).ravel()


def solveShift(uv: np.ndarray, xyz: np.ndarray, ftol: float = 1e-3) -> float:
    """min over shift of |focal * xy / (z + shift) - uv|, focal solved in closed form.

    Upstream calls scipy.optimize.least_squares(method="lm", x0=0, ftol=1e-3);
    this is the same damped Gauss-Newton in one variable with the same stop.
    """
    uv = uv.reshape(-1, 2).astype(np.float64)
    xyz = xyz.reshape(-1, 3).astype(np.float64)
    xy, z = xyz[:, :2], xyz[:, 2]

    def residual(value: float) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return _projectionResidual(value, uv, xy, z)

    shift, damping = 0.0, 1e-3
    current = residual(shift)
    cost = float(current @ current)
    if not np.isfinite(cost):
        return 0.0

    for _ in range(MAX_ITERATIONS):
        # MINPACK's forward-difference step: relative, or absolute at zero. A
        # relative-only step at the starting point (shift = 0) is ~1e-16, which
        # drowns the Jacobian in rounding and stopped the solve at 0 on some
        # frames (a whole-frame depth jump).
        step = np.sqrt(np.finfo(np.float64).eps) * (abs(shift) or 1.0)
        jacobian = (residual(shift + step) - current) / step
        hessian = float(jacobian @ jacobian)
        gradient = float(jacobian @ current)
        if hessian == 0.0 or not np.isfinite(hessian):
            break

        improved = False
        while damping < 1e12:
            delta = -gradient / (hessian * (1.0 + damping))
            candidate = residual(shift + delta)
            candidateCost = float(candidate @ candidate)
            if np.isfinite(candidateCost) and candidateCost < cost:
                improved = True
                break
            damping *= 10.0
        if not improved:
            break

        reduction = (cost - candidateCost) / max(cost, 1e-30)
        shift, current, cost = shift + delta, candidate, candidateCost
        damping = max(damping * 0.1, 1e-12)
        if reduction <= ftol:
            break
    return shift


def solveSamples(
    height: int, width: int, device: torch.device | str
) -> tuple[Tensor, Tensor]:
    """Rows and columns a nearest resize to SOLVE_SIZE reads, exactly as
    `F.interpolate(mode="nearest")` picks them (it is separable per axis)."""

    def pick(size: int, target: int) -> Tensor:
        source = torch.arange(size, dtype=torch.float32, device=device)
        picked = F.interpolate(source.view(1, 1, size, 1), (target, 1), mode="nearest")
        return picked.view(-1).long()

    return pick(height, SOLVE_SIZE[0]), pick(width, SOLVE_SIZE[1])


def recoverShift(pointsLow: Tensor, uvLow: Tensor, maskLow: Tensor | None) -> Tensor:
    """Z shift taking an affine point map to camera space, from its SOLVE_SIZE
    nearest sample: points [h, w, 3], uv [h, w, 2], mask [h, w] bool."""
    uvNumpy = uvLow.cpu().numpy()
    pointsNumpy = pointsLow.cpu().numpy()
    if maskLow is not None:
        maskNumpy = maskLow.cpu().numpy()
        uvNumpy, pointsNumpy = uvNumpy[maskNumpy], pointsNumpy[maskNumpy]
    if uvNumpy.shape[0] < 2:
        return pointsLow.new_zeros(())
    return pointsLow.new_tensor(float(solveShift(uvNumpy, pointsNumpy)))
