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
