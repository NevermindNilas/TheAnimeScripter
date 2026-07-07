"""Tests for src/sceneChange streaming scene-cut detectors.

These gate the interpolation cut-skip: a detector returns True on a hard cut
(so the loop holds instead of morphing across it). Semantics differ from dedup:
  * dedup detects DUPLICATES (SSIM high / MSE low) and only advances its
    reference on a kept frame.
  * a scene detector detects CUTS (SSIM low / MSE high) and advances its
    reference EVERY frame (each frame becomes the reference for the next).
"""

import pytest

torch = pytest.importorskip("torch")

from src.sceneChange.detector import (
    SceneChangeMSE,
    SceneChangeMSECuda,
    SceneChangeSSIM,
)


def _frame(value, h=64, w=64):
    # NCHW float frame in [0, 1], matching the pipeline's tensor layout.
    return torch.full((1, 3, h, w), float(value), dtype=torch.float32)


def _noisyFrame(seed, h=64, w=64):
    g = torch.Generator().manual_seed(seed)
    return torch.rand((1, 3, h, w), generator=g, dtype=torch.float32)


# --------------------------------------------------------------------------- #
# SSIM cut detection (high == similar; cut when ssim < threshold)
# --------------------------------------------------------------------------- #


def testSSIMFirstFrameNeverCut():
    d = SceneChangeSSIM(threshold=0.5)
    assert d(_noisyFrame(0)) is False  # no predecessor


def testSSIMIdenticalIsNotCut():
    d = SceneChangeSSIM(threshold=0.5)
    f = _noisyFrame(0)
    assert d(f) is False
    assert d(f.clone()) is False  # SSIM 1.0 -> similar -> no cut


def testSSIMDisjointIsCut():
    d = SceneChangeSSIM(threshold=0.5)
    assert d(_noisyFrame(0)) is False
    assert d(_noisyFrame(1)) is True  # uncorrelated -> SSIM ~0 -> cut


def testSSIMReferenceAdvancesEveryFrame():
    # After a cut A->B, B is the new reference: a repeat of B is NOT a cut.
    d = SceneChangeSSIM(threshold=0.5)
    a, b = _noisyFrame(0), _noisyFrame(1)
    assert d(a) is False
    assert d(b) is True  # cut A->B
    assert d(b.clone()) is False  # B->B: reference advanced, no cut


# --------------------------------------------------------------------------- #
# MSE cut detection (low == similar; cut when mse > threshold). Direction is
# OPPOSITE to SSIM -- pinned so the inversion can't sneak in.
# --------------------------------------------------------------------------- #


def testMSEIdenticalIsNotCut():
    d = SceneChangeMSE(threshold=1000.0)
    assert d(_frame(0.5)) is False  # first
    assert d(_frame(0.5)) is False  # identical -> MSE 0 -> no cut


def testMSEDisjointIsCut():
    d = SceneChangeMSE(threshold=1000.0)
    assert d(_frame(0.0)) is False  # first (black)
    assert d(_frame(1.0)) is True  # black->white -> MSE 255^2 -> cut


def testMSECudaLogicOnCpu():
    # half=False allocates no CUDA tensors, so the logic runs on CPU frames.
    d = SceneChangeMSECuda(threshold=1000.0, half=False)
    assert d(_frame(0.0)) is False
    assert d(_frame(1.0)) is True
    assert d(_frame(1.0)) is False  # reference advanced to white -> no cut


def testCudaDetectorsExist():
    # The CUDA / ONNX detector classes must exist (they are advertised CLI
    # choices routed by the factory) even though constructing them needs a GPU
    # or model weights.
    import src.sceneChange.detector as m

    for name in ("SceneChangeSSIMCuda", "SceneChangeScorer6chDetector"):
        assert isinstance(getattr(m, name), type)


def testFactoryMapsMethods():
    import types

    from src.factories.sceneChange import buildSceneChangeProcess

    fake = types.SimpleNamespace(
        sceneChangeMethod="ssim",
        sceneChangeThreshold=0.5,
        half=False,
    )
    det = buildSceneChangeProcess(fake)
    assert type(det).__name__ == "SceneChangeSSIM"

    fake.sceneChangeMethod = "mse"
    assert type(buildSceneChangeProcess(fake)).__name__ == "SceneChangeMSE"

    fake.sceneChangeMethod = "bogus"
    with pytest.raises(ValueError):
        buildSceneChangeProcess(fake)


def _scArgs(method):
    import argparse

    return argparse.Namespace(
        slowmo=False,
        static_step=False,
        interpolate_factor=2,
        dedup=False,
        autoclip=False,
        compile_mode="default",
        interpolate=True,
        scenechange=True,
        scenechange_method=method,
        scenechange_sens=50.0,
    )


def testValidatorDowngradesCudaMethodsWithoutCuda(monkeypatch):
    # On a box without CUDA, the default ssim-cuda (and other CUDA/TRT
    # detectors) would crash at device init; the validator must downgrade them.
    import src.infra.isCudaInit as ic
    from src.cli.validator import _configureProcessingSettings

    class FakeChecker:
        cudaAvailable = False

    monkeypatch.setattr(ic, "CudaChecker", FakeChecker)

    for src_m, dst_m in (
        ("ssim-cuda", "ssim"),
        ("mse-cuda", "mse"),
        ("maxxvit-tensorrt", "maxxvit-directml"),
    ):
        a = _scArgs(src_m)
        _configureProcessingSettings(a)
        assert a.scenechange_method == dst_m, f"{src_m} should downgrade to {dst_m}"
        assert a.scenechange_threshold is not None


def testValidatorKeepsCudaMethodsWithCuda(monkeypatch):
    import src.infra.isCudaInit as ic
    from src.cli.validator import _configureProcessingSettings

    class FakeChecker:
        cudaAvailable = True

    monkeypatch.setattr(ic, "CudaChecker", FakeChecker)
    a = _scArgs("ssim-cuda")
    _configureProcessingSettings(a)
    assert a.scenechange_method == "ssim-cuda"  # unchanged when CUDA present
