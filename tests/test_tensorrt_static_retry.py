import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _loadTensorrtUpscale(monkeypatch):
    def inferenceMode():
        def decorator(func):
            return func

        return decorator

    if importlib.util.find_spec("torch") is None:
        fakeTorch = SimpleNamespace(
            __version__="test",
            float16="float16",
            float32="float32",
            tensor=object,
            inference_mode=inferenceMode,
            set_float32_matmul_precision=lambda *_args, **_kwargs: None,
            device=lambda name: name,
            cuda=SimpleNamespace(is_available=lambda: False),
            backends=SimpleNamespace(
                mps=SimpleNamespace(is_available=lambda: False, is_built=lambda: False),
                cudnn=SimpleNamespace(benchmark=False, enabled=False),
            ),
        )
        monkeypatch.setitem(sys.modules, "torch", fakeTorch)

    modulePath = Path(__file__).resolve().parents[1] / "src" / "upscale" / "tensorrt.py"
    spec = importlib.util.spec_from_file_location("test_tensorrt_upscale", modulePath)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _makeUniversalTensorRT(monkeypatch, creator):
    tensorrt_upscale = _loadTensorrtUpscale(monkeypatch)
    monkeypatch.setattr(tensorrt_upscale, "logAndPrint", lambda *args, **kwargs: None)
    instance = tensorrt_upscale.UniversalTensorRT.__new__(
        tensorrt_upscale.UniversalTensorRT
    )
    instance.modelPath = "model.onnx"
    instance.half = True
    instance.height = 720
    instance.width = 960
    instance.forceStatic = False
    instance.tensorRTEngineCreator = creator
    return instance


def testUniversalTensorRTRetriesStaticProfileAfterDynamicBuildFailure(monkeypatch):
    calls = []

    def creator(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return None, None
        return object(), object()

    instance = _makeUniversalTensorRT(monkeypatch, creator)

    instance._createEngineWithStaticRetry("model.engine")

    assert instance.engine is not None
    assert instance.context is not None
    assert calls[0]["forceStatic"] is False
    assert calls[0]["inputsMin"] == [1, 3, 8, 8]
    assert calls[0]["inputsOpt"] == [1, 3, 720, 960]
    assert calls[0]["inputsMax"] == [1, 3, 1080, 1920]
    assert calls[1]["forceStatic"] is True
    assert calls[1]["inputsMin"] == [1, 3, 720, 960]
    assert calls[1]["inputsOpt"] == [1, 3, 720, 960]
    assert calls[1]["inputsMax"] == [1, 3, 720, 960]


def testUniversalTensorRTRaisesClearErrorWhenStaticRetryFails(monkeypatch):
    def creator(**_kwargs):
        return None, None

    instance = _makeUniversalTensorRT(monkeypatch, creator)

    with pytest.raises(RuntimeError, match="Failed to build TensorRT engine"):
        instance._createEngineWithStaticRetry("model.engine")
