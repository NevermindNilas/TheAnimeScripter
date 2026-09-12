"""Streaming contracts independent of weights, video codecs and CUDA."""

from types import SimpleNamespace

import numpy as np
import pytest

from src.depth.chunk_stream import alignDepthOverlap, streamDepthChunks


@pytest.mark.parametrize("chunk", [4, 8, 16, 32])
@pytest.mark.parametrize(
    "count", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 31, 32, 33, 65]
)
def testEveryFrameOnceInOrder(chunk, count):
    consumed = []
    calls = []

    def frames():
        for index in range(count):
            consumed.append(index)
            yield index

    def infer(window):
        calls.append(list(window))
        # Scale changes per chunk simulate the ambiguity of any-view depth.
        return np.stack([np.full((8, 8), i + 1.0) for i in window]) * len(calls)

    output = streamDepthChunks(frames(), infer, chunk)
    for i, depth in enumerate(output):
        np.testing.assert_allclose(depth, i + 1, rtol=1e-5)
        assert len(consumed) <= i + chunk + 1  # bounded lookahead
    assert len(consumed) == count
    assert all(len(call) <= chunk for call in calls)
    if count:
        assert i + 1 == count
        assert len(calls) == max(
            1, (count - chunk // 2 + chunk // 2 - 1) // (chunk // 2)
        )
    else:
        assert calls == []


def testAlignIgnoresInvalidValuesAndResistsOutliers():
    previous = np.full((2, 16, 16), 6, dtype=np.float32)
    current = np.full((4, 16, 16), 2, dtype=np.float32)
    current[0, :2] = 2000
    previous[1, 0] = 0
    result = alignDepthOverlap(previous, current)
    np.testing.assert_allclose(result[2:], 6)
    np.testing.assert_allclose(result[1], 6)


def testAtMostOneChunkOfRgbFramesRetained():
    import weakref

    class Frame:
        pass

    alive = weakref.WeakSet()

    def frames():
        for _ in range(100):
            frame = Frame()
            alive.add(frame)
            assert len(alive) <= 8
            yield frame

    result = streamDepthChunks(frames(), lambda window: np.ones((len(window), 8, 8)), 8)
    assert sum(1 for _ in result) == 100


def testInvalidOverlapStartsNewScaleWithoutBlending():
    current = np.full((4, 8, 8), 3, dtype=np.float32)
    np.testing.assert_array_equal(alignDepthOverlap(np.zeros((2, 8, 8)), current), 3)


def testInvalidDepthIsBlackAndFinite():
    def infer(window):
        result = np.ones((len(window), 8, 8))
        result[:, 0, :4] = [np.nan, np.inf, -np.inf, -1]
        return result

    output = list(streamDepthChunks(range(9), infer, 4))
    assert len(output) == 9
    assert np.isfinite(output).all()
    np.testing.assert_array_equal(np.array(output)[:, 0, :4], 0)


def testInferenceFailurePropagates():
    def fail(window):
        raise RuntimeError("inference failed")

    with pytest.raises(RuntimeError, match="inference failed"):
        list(streamDepthChunks(range(5), fail, 4))


def testWrongOutputCountIsRejected():
    with pytest.raises(ValueError, match="one.*depth map per frame"):
        list(streamDepthChunks(range(5), lambda _: np.ones((1, 8, 8)), 4))


@pytest.mark.parametrize("chunk,overlap", [(1, None), (4, 0), (4, 3)])
def testInvalidWindowsRejected(chunk, overlap):
    with pytest.raises(ValueError):
        list(streamDepthChunks([], lambda _: None, chunk, overlap))


@pytest.mark.parametrize(
    "method,base", [("video_small_v3", "small_v3"), ("video_base_v3", "base_v3")]
)
def testMethodsReusePermittedCheckpointsAndDisableIndependentBatching(
    monkeypatch, tmp_path, method, base
):
    from src.cli.parser import _buildParser, capabilityMethods
    from src.cli.validator import _handleDepthSettings
    from src.model import download
    from src.model.registry import modelsList, modelsMap

    assert method in capabilityMethods(_buildParser("."))["depth"]
    assert method in modelsList()
    assert modelsMap(method, modelType="pth") == modelsMap(base, modelType="pth")
    downloads = []
    monkeypatch.setattr(download, "weightsDir", str(tmp_path))
    monkeypatch.setattr(
        download, "downloadAndLog", lambda *args: downloads.append(args)
    )
    download.downloadModels(method)
    assert downloads[0][3] == str(tmp_path / base)
    assert downloads[0][1] == modelsMap(base, modelType="pth")

    args = SimpleNamespace(
        depth=True, depth_method=method, depth_quality="low", depth_batch=8
    )
    _handleDepthSettings(args)
    assert args.depth_batch == 1


def testChunkInferenceUsesViewsOfOneScene():
    torch = pytest.importorskip("torch")
    pytest.importorskip("cv2")
    pytest.importorskip("nelux")

    from src.depth.backends.da3_streaming import DA3StreamingCuda

    driver = DA3StreamingCuda.__new__(DA3StreamingCuda)
    seen = []

    def forward(batch):
        seen.append(tuple(batch.shape))
        return {"depth": torch.ones((1, batch.shape[1], 8, 8))}

    driver.model = SimpleNamespace(
        input_processor=lambda frames, *_: (
            torch.zeros(len(frames), 3, 28, 42),
            None,
            None,
        ),
        _get_model_device=lambda: torch.device("cpu"),
        forward=forward,
        output_processor=lambda result: SimpleNamespace(
            depth=result["depth"][0].numpy()
        ),
    )
    driver.processRes = 42
    driver.processResMethod = "upper_bound_resize"
    assert driver._inferChunk([None] * 4).shape == (4, 8, 8)
    assert seen == [(1, 4, 3, 28, 42)]


def testFailedChunkClosesWriterAndDrainsReader(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("cv2")
    pytest.importorskip("nelux")
    from contextlib import nullcontext

    from src.depth.backends import da3_streaming
    from src.io import ffmpegSettings

    driver = da3_streaming.DA3StreamingCuda.__new__(da3_streaming.DA3StreamingCuda)
    driver.depthWindow = 4
    driver.totalFrames = 9
    driver.processingError = None
    source = iter([np.zeros((8, 8, 3))] * 9 + [None])
    driver.readBuffer = SimpleNamespace(read=lambda: next(source))
    closed = []
    drained = []
    driver.writeBuffer = SimpleNamespace(close=lambda: closed.append(True))
    monkeypatch.setattr(
        ffmpegSettings, "drainReader", lambda reader: drained.append(reader)
    )
    monkeypatch.setattr(
        da3_streaming, "ProgressBarLogic", lambda _: nullcontext(lambda _: None)
    )

    def fail(_):
        raise RuntimeError("out of memory")

    driver._inferChunk = fail
    driver.guardedProcess()
    assert str(driver.processingError) == "out of memory"
    assert closed == [True]
    assert drained == [driver.readBuffer]
