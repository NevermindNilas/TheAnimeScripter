"""Tests for src/infra/providerCheck.warnIfProviderMissing.

ONNX Runtime can construct a session with a GPU provider requested and still run
on CPU; the helper compares get_providers() against the requested provider and
warns (visibly + to the log) when the GPU provider is absent.
"""

from src.infra.providerCheck import warnIfProviderMissing


class _FakeSession:
    def __init__(self, providers):
        self._providers = providers

    def get_providers(self):
        return self._providers


class _BrokenSession:
    def get_providers(self):
        raise RuntimeError("session not initialized")


def testProviderPresentReturnsTrueNoWarning(capsys):
    session = _FakeSession(["DmlExecutionProvider", "CPUExecutionProvider"])
    assert warnIfProviderMissing(session, "DmlExecutionProvider", "upscale") is True
    assert capsys.readouterr().out == ""


def testProviderMissingReturnsFalseAndWarns(capsys):
    session = _FakeSession(["CPUExecutionProvider"])
    assert warnIfProviderMissing(session, "DmlExecutionProvider", "upscale") is False
    out = capsys.readouterr().out
    assert "DmlExecutionProvider" in out
    assert "upscale" in out


def testCpuRequestedIsNeverAWarning(capsys):
    session = _FakeSession(["CPUExecutionProvider"])
    assert warnIfProviderMissing(session, "CPUExecutionProvider", "restore") is True
    assert capsys.readouterr().out == ""


def testBrokenSessionIsSwallowed(capsys):
    assert (
        warnIfProviderMissing(_BrokenSession(), "DmlExecutionProvider", "depth") is True
    )
    assert capsys.readouterr().out == ""
