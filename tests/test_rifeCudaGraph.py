"""GPU regression tests for RifeCuda's CUDA-graph path.

Skipped without CUDA and locally-present weights, so CI stays green; run them on
a box with a GPU after touching src/interpolate/rife.py.

Covers two defects:
  * The capture's self-check ran an eager forward AFTER capture, and a
    head-bearing arch's forward does ``self.f1 = self.encode(img1)``. That left
    ``model.f1`` pointing at a tensor the graph never writes, so
    ``model.cache()`` (``f0.copy_(f1)``) fed the same init-time features into
    every frame for the whole run -- graph output sat ~28.6 dB from the eager
    reference on a 540p sequence instead of matching it. rife4.6 was immune
    (no encoder head).
  * Replay never runs the arch's ``if self.f0 is None`` branch, so the first
    interpolated pair used whatever features the warmup had left behind.
"""

import pathlib
import queue

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)

W, H = 320, 192
WEIGHTS = pathlib.Path(__file__).resolve().parents[1] / "weights"

# (method, weight file) -- rife4.25 has an encoder head, rife4.6 does not.
CASES = [("rife4.25", "rife4.25/rife425.pth"), ("rife4.6", "rife4.6/rife46.pth")]


def _requireWeights(rel):
    if not (WEIGHTS / rel).exists():
        pytest.skip(f"{rel} not downloaded")


def _frames(n):
    g = torch.Generator().manual_seed(0)
    base = torch.rand(1, 3, H, W, generator=g)
    out = []
    for i in range(n):
        f = torch.roll(base, shifts=(3 * i, 2 * i), dims=(2, 3))
        out.append((f + torch.randn(f.shape, generator=g) * 0.01).clamp(0, 1).cuda())
    return out


def _collect(model, frames, factor=2):
    q = queue.Queue()
    for f in frames:
        model(f, q, factor)
    torch.cuda.synchronize()
    return [q.get() for _ in range(q.qsize())]


def _eagerTwin(rife, *args, **kwargs):
    """A RifeCuda built with capture skipped -- the reference implementation."""
    real = rife.RifeCuda._setupCudaGraph

    def noGraph(self):
        self.cudaGraph = self._graphOut = None
        self._graphs = {}
        self._graphFeats = {}
        self._armedScale = None
        self.useGraph = False

    rife.RifeCuda._setupCudaGraph = noGraph
    try:
        return rife.RifeCuda(*args, **kwargs)
    finally:
        rife.RifeCuda._setupCudaGraph = real


@pytest.mark.parametrize("method,weight", CASES)
@pytest.mark.parametrize("dynamicScale", [False, True])
def testGraphPathMatchesEagerFrameForFrame(method, weight, dynamicScale):
    _requireWeights(weight)
    from src.interpolate import rife

    frames = _frames(12)

    graphed = rife.RifeCuda(
        True, W, H, method, dynamicScale=dynamicScale, interpolateFactor=2
    )
    if not graphed.useGraph:
        pytest.skip("graph capture unavailable for this configuration")
    got = _collect(graphed, frames)
    del graphed
    torch.cuda.empty_cache()

    eager = _eagerTwin(
        rife, True, W, H, method, dynamicScale=dynamicScale, interpolateFactor=2
    )
    want = _collect(eager, frames)

    for i, (a, b) in enumerate(zip(got, want, strict=True)):
        diff = (a.float() - b.float()).abs().max().item()
        assert diff < 5e-3, f"frame {i} diverged by {diff}"


def testDynamicScaleCapturesOneGraphPerScale():
    _requireWeights(CASES[0][1])
    from src.interpolate import rife
    from src.rifearches.dynamic_scale import DYNAMIC_SCALES

    m = rife.RifeCuda(True, W, H, "rife4.25", dynamicScale=True, interpolateFactor=2)
    if not m.useGraph:
        pytest.skip("graph capture unavailable")
    assert sorted(m._graphs) == sorted(DYNAMIC_SCALES)
    # Every capture must own a distinct f1, else cache() would propagate another
    # scale's features.
    f1s = [id(f1) for _, f1 in m._graphFeats.values() if f1 is not None]
    assert len(f1s) == len(set(f1s))


def testArmingSelectsTheMatchingGraphAndRebindsFeatures():
    _requireWeights(CASES[0][1])
    from src.interpolate import rife

    m = rife.RifeCuda(True, W, H, "rife4.25", dynamicScale=True, interpolateFactor=2)
    if not m.useGraph:
        pytest.skip("graph capture unavailable")
    for scale in sorted(m._graphs):
        m._armGraph(scale)
        assert m.cudaGraph is m._graphs[scale][0]
        assert m._graphOut is m._graphs[scale][1]
        assert m.model.f1 is m._graphFeats[scale][1]


def testDynamicScaleFallsBackToEagerAboveFactorTwo():
    """Replay cannot advance the arch's per-call encoder-refresh counter, so the
    multi-scale capture is restricted to the 2x path."""
    _requireWeights(CASES[0][1])
    from src.interpolate import rife

    m = rife.RifeCuda(True, W, H, "rife4.25", dynamicScale=True, interpolateFactor=4)
    assert not m.useGraph
    assert m._graphs == {}
