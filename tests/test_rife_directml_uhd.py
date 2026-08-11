"""Regression tests for the UHD (scale=0.5) forward path of the DirectML/
OpenVINO RIFE arches (src/rifearches/Rife_directml.py).

RifeDirectML sets scale=0.5 for inputs above 1920x1080, which turns
scaleList = [8/scale, ..., 1/scale] into [16, 8, 4, 2]. The final full-image
warp used to be guarded by `if scale == 1:`, which no iteration satisfies at
scale=0.5, so `warpedImgs` was never assigned and ONNX export died with
UnboundLocalError before processing a single frame. The guard is now
`scale == self.scaleList[-1]` (the pattern the 4.25 family already used).
"""

import pytest

torch = pytest.importorskip("torch")

from src.rifearches.Rife_directml import (
    IFNet_415,
    IFNet_420,
    IFNet_422,
    IFNet_422_lite,
)

ARCHES = [IFNet_415, IFNet_420, IFNet_422, IFNet_422_lite]

# Small dims that pad identically under mod-32 and the driver's UHD mod-64.
WIDTH, HEIGHT = 128, 64


def _forward(arch, scale, ensemble):
    torch.manual_seed(0)
    model = arch(
        scale=scale,
        ensemble=ensemble,
        dtype=torch.float32,
        device="cpu",
        width=WIDTH,
        height=HEIGHT,
    ).eval()
    img0 = torch.rand(1, 3, model.ph, model.pw)
    img1 = torch.rand(1, 3, model.ph, model.pw)
    timestep = torch.full((1, 1, model.ph, model.pw), 0.5)
    with torch.inference_mode():
        return model(img0, img1, timestep)


@pytest.mark.parametrize("arch", ARCHES)
def testUhdScaleForwardCompletes(arch):
    """scale=0.5 -> scaleList ends at 2; used to raise UnboundLocalError."""
    out = _forward(arch, scale=0.5, ensemble=False)
    assert out.shape == (1, 3, HEIGHT, WIDTH)


@pytest.mark.parametrize("arch", ARCHES)
def testDefaultScaleForwardStillCompletes(arch):
    """At scale=1 the new guard is equivalent (scaleList[-1] == 1)."""
    out = _forward(arch, scale=1.0, ensemble=False)
    assert out.shape == (1, 3, HEIGHT, WIDTH)


def testUhdScaleForwardCompletesWithEnsemble():
    """The failing field config: rife4.22 UHD with ensemble on."""
    out = _forward(IFNet_422, scale=0.5, ensemble=True)
    assert out.shape == (1, 3, HEIGHT, WIDTH)
