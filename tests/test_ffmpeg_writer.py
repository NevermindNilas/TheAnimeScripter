import importlib
import sys
import types


def testWriteBufferInitialNoneSentinelExitsCleanly(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "nelux", types.SimpleNamespace())
    ffmpegSettings = importlib.import_module("src.io.ffmpegSettings")

    wb = ffmpegSettings.WriteBuffer(output=str(tmp_path / "out.mp4"))
    wb.writeBuffer.put(None)

    wb()
