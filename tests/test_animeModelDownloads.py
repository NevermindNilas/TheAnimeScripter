"""Published upstream assets must resolve into version-isolated caches."""

import os

import pytest

from src.model import download


@pytest.mark.parametrize(
    "method",
    [
        "cyte",
        "cyte-mps",
        "cyte-tensorrt",
        "cyte-directml",
        "cyte-openvino",
        "limbo_v2",
        "limbo_v2-mps",
        "limbo_v2-tensorrt",
        "limbo_v2_43-tensorrt",
        "limbo_v2-directml",
        "limbo_v2_43-directml",
    ],
)
@pytest.mark.parametrize("half", [True, False])
def testPublishedModelRoute(method, half, tmp_path, monkeypatch):
    monkeypatch.setattr(download, "weightsDir", str(tmp_path))
    calls = []
    monkeypatch.setattr(download, "downloadAndLog", lambda *args: calls.append(args))
    onnx = method.endswith(("-tensorrt", "-directml", "-openvino"))
    download.downloadModels(method, modelType="onnx" if onnx else "pth", half=half)
    _, filename, url, folder = calls[0]
    if method.startswith("limbo"):
        assert filename.startswith("LimboV2")
        assert os.path.basename(folder).startswith("limbo_v2")
        assert url.endswith("/Limbo-V2/" + filename.replace("LimboV2", "Limbo"))
    else:
        assert url.endswith("/Cyte-V1-SuperUltraCompact/" + filename)
        if onnx:
            assert ("fp16" if half else "fp32") in filename
