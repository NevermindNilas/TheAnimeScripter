"""Tests for src/dedup/dedup.py deduplication backends.

Regression coverage for three bugs in the MSE dedup path:
  * DedupMSE.processFrame used np.resize, which tiles/truncates the flat byte
    buffer instead of resizing the image, so the MSE comparison was meaningless.
  * The MSE comparison direction was inverted (it treated HIGH MSE / different
    frames as duplicates, like the SSIM/VMAF "high score == similar" backends);
    MSE is the opposite -- 0 == identical.
  * DedupMSECuda was selectable via --dedup_method mse-cuda but the class did
    not exist (ImportError at init).
"""

import pytest

torch = pytest.importorskip("torch")

from src.dedup.dedup import DedupMSE, DedupMSECuda, DedupSSIM


def _frame(value, h=64, w=64):
    # NCHW float frame in [0, 1], matching the pipeline's tensor layout.
    return torch.full((1, 3, h, w), float(value), dtype=torch.float32)


def testDedupMSEProcessFrameReturnsResizedTensor():
    # Must be a torch tensor shaped (1, 3, sampleSize, sampleSize); the old
    # np.resize path returned a (224, 224, 3) numpy array of tiled bytes.
    out = DedupMSE().processFrame(_frame(0.5))
    assert isinstance(out, torch.Tensor)
    assert tuple(out.shape) == (1, 3, 224, 224)
    # A uniform 0.5 frame resizes to all pixels ~127.5 on the 0-255 scale (a real
    # resize, not tiled/truncated bytes).
    assert torch.allclose(out, torch.full_like(out, 127.5), atol=1e-3)


def testDedupMSEFlagsIdenticalFramesAsDuplicates():
    d = DedupMSE(mseThreshold=10.0)
    assert d(_frame(0.5)) is False  # first frame: nothing to compare against
    assert d(_frame(0.5)) is True   # identical -> MSE 0 -> duplicate
    assert d(_frame(0.0)) is False  # black vs 0.5 reference -> distinct -> keep


def testDedupMSEKeepsReferenceOnDuplicate():
    # After a duplicate, the reference frame must NOT advance, so a later frame
    # is still compared against the original kept frame.
    d = DedupMSE(mseThreshold=10.0)
    d(_frame(0.5))                  # keep (first)
    assert d(_frame(0.5)) is True   # duplicate, reference stays at 0.5
    assert d(_frame(0.5)) is True   # still a duplicate of the 0.5 reference


def testDedupMSECudaExistsWithExpectedInterface():
    # Regression: the class must exist (mse-cuda is an advertised CLI choice).
    for name in ("__call__", "processFrame"):
        assert hasattr(DedupMSECuda, name)


def testDedupMSECudaLogicOnCpu():
    # __init__ allocates no CUDA tensors, so the dedup logic is exercisable on
    # CPU with half=False.
    d = DedupMSECuda(mseThreshold=10.0, half=False)
    assert d(_frame(0.5)) is False
    assert d(_frame(0.5)) is True
    assert d(_frame(0.0)) is False


# --------------------------------------------------------------------------- #
# DedupSSIM (CPU): comparison direction is OPPOSITE to MSE -- high score (1.0)
# means identical. Pinned so the inverted-comparison bug fixed in the MSE
# backend can never sneak into the SSIM one.
# --------------------------------------------------------------------------- #

def _noisyFrame(seed, h=64, w=64):
    g = torch.Generator().manual_seed(seed)
    return torch.rand((1, 3, h, w), generator=g, dtype=torch.float32)


def testDedupSSIMFlagsIdenticalFramesAsDuplicates():
    d = DedupSSIM(ssimThreshold=0.9)
    f = _noisyFrame(0)
    assert d(f) is False           # first frame: nothing to compare against
    assert d(f.clone()) is True    # identical -> SSIM 1.0 -> duplicate


def testDedupSSIMKeepsDistinctFrames():
    d = DedupSSIM(ssimThreshold=0.9)
    assert d(_noisyFrame(0)) is False
    assert d(_noisyFrame(1)) is False  # uncorrelated noise -> SSIM ~0 -> keep


def testDedupSSIMKeepsReferenceOnDuplicate():
    # After a duplicate the reference must NOT advance (same contract as MSE).
    d = DedupSSIM(ssimThreshold=0.9)
    f = _noisyFrame(0)
    d(f)
    assert d(f.clone()) is True
    assert d(f.clone()) is True
    assert d(_noisyFrame(1)) is False


def testDedupSSIMAdvancesReferenceOnKeptFrame():
    d = DedupSSIM(ssimThreshold=0.9)
    a, b = _noisyFrame(0), _noisyFrame(1)
    d(a)
    assert d(b) is False           # distinct -> kept -> becomes the reference
    assert d(b.clone()) is True    # duplicate of the NEW reference

