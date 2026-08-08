import importlib
import importlib.util
import sys
import types


def _installFakeTorch(monkeypatch):
    if importlib.util.find_spec("torch") is not None:
        return

    functionalModule = types.ModuleType("torch.nn.functional")
    nnModule = types.ModuleType("torch.nn")
    nnModule.functional = functionalModule

    fakeTorch = types.ModuleType("torch")
    fakeTorch.__version__ = "test"
    fakeTorch.uint8 = "uint8"
    fakeTorch.uint16 = "uint16"
    fakeTorch.cuda = types.SimpleNamespace(is_available=lambda: False)
    fakeTorch.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: False, is_built=lambda: False),
        cudnn=types.SimpleNamespace(benchmark=False, enabled=False),
    )
    fakeTorch.device = lambda name: name
    fakeTorch.nn = nnModule

    monkeypatch.setitem(sys.modules, "torch", fakeTorch)
    monkeypatch.setitem(sys.modules, "torch.nn", nnModule)
    monkeypatch.setitem(sys.modules, "torch.nn.functional", functionalModule)


def testWriteBufferInitialNoneSentinelExitsCleanly(monkeypatch, tmp_path):
    _installFakeTorch(monkeypatch)
    monkeypatch.setitem(sys.modules, "nelux", types.SimpleNamespace())
    ffmpegSettings = importlib.import_module("src.io.ffmpegSettings")

    wb = ffmpegSettings.WriteBuffer(output=str(tmp_path / "out.mp4"))
    wb.writeBuffer.put(None)

    wb()


def testNeluxWriteBufferBenchmarkConsumesFramesWithoutEncoder(monkeypatch, tmp_path):
    _installFakeTorch(monkeypatch)
    monkeypatch.setitem(sys.modules, "nelux", types.SimpleNamespace())
    ffmpegSettings = importlib.import_module("src.io.ffmpegSettings")

    wb = ffmpegSettings.NeluxWriteBuffer(
        output=str(tmp_path / "out.mp4"),
        benchmark=True,
    )
    wb.writeBuffer.put(object())
    wb.writeBuffer.put(None)

    wb()

    assert wb.writtenFrames == 1
    assert wb.encoder is None


def testNeluxWriteBufferPassesTheExactFpsThrough(monkeypatch, tmp_path):
    """nelux 0.17.0 converts the float fps to an exact rational itself. TAS must
    hand it the unrounded value and must NOT pass a `time_base` override --
    0.16.0 needed one and tagged 24000/1001 sources as 24/1 without it, so this
    pins that the workaround stays gone and the exact rate still goes in."""
    _installFakeTorch(monkeypatch)
    captured = {}

    class FakeEncoder:
        def __init__(self, output, **kwargs):
            captured.update(kwargs)

        def encode_frame(self, frame):
            pass

        def close(self):
            pass

    monkeypatch.setitem(
        sys.modules, "nelux", types.SimpleNamespace(VideoEncoder=FakeEncoder)
    )
    ffmpegSettings = importlib.import_module("src.io.ffmpegSettings")
    monkeypatch.setattr(ffmpegSettings, "nelux", sys.modules["nelux"])

    wb = ffmpegSettings.NeluxWriteBuffer(
        output=str(tmp_path / "out.mp4"),
        encode_method="x264_nelux",
        fps=2 * 24000 / 1001,
    )
    wb.writeBuffer.put(None)

    wb()

    assert captured["fps"] == 2 * 24000 / 1001
    assert "options" not in captured


def testNeluxWriteBufferMapsVp9ToConstantQuality(monkeypatch, tmp_path):
    """libvpx-vp9 only honours `cq` as a true CRF when the target bitrate is 0;
    without it libvpx falls back to constrained quality at ~3.4x the size."""
    _installFakeTorch(monkeypatch)
    monkeypatch.setitem(sys.modules, "nelux", types.SimpleNamespace())
    ffmpegSettings = importlib.import_module("src.io.ffmpegSettings")

    wb = ffmpegSettings.NeluxWriteBuffer(
        output=str(tmp_path / "out.webm"), encode_method="vp9_nelux"
    )

    assert wb.encoderKwargs["codec"] == "libvpx-vp9"
    assert wb.encoderKwargs["bit_rate"] == 0
    assert wb.encoderKwargs["cq"] == 15
    assert not wb.expectHardware
