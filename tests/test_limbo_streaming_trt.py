"""Local export and source-weight routing without TensorRT or real weights."""

from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(
    "version,filename",
    [("limbo", "Limbo.safetensors"), ("limbo_v2", "LimboV2.safetensors")],
)
def testStreamingDownloadsSourceWeights(version, filename, tmp_path, monkeypatch):
    from src.model import download

    calls = []
    monkeypatch.setattr(download, "weightsDir", str(tmp_path))
    monkeypatch.setattr(download, "downloadAndLog", lambda *args: calls.append(args))
    download.downloadModels(f"video_{version}-tensorrt", modelType="onnx")
    assert calls[0][1] == filename
    assert calls[0][3] == str(tmp_path / version)
    assert calls[0][2].endswith(".safetensors")


@pytest.mark.parametrize("method", ["video_limbo-tensorrt", "video_limbo_v2-tensorrt"])
def testStreamingFixedQualityAndNoIndependentBatch(method):
    from src.cli.validator import _handleDepthSettings

    args = SimpleNamespace(
        depth=True, depth_method=method, depth_quality="high", depth_batch=8
    )
    _handleDepthSettings(args)
    assert args.depth_quality == "low"
    assert args.depth_batch == 1


def testExportRejectsIncompleteCheckpoint(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("onnx")
    from src.depth import streaming_export as export

    checkpoint = tmp_path / "broken.safetensors"
    checkpoint.write_bytes(b"incomplete checkpoint")
    module = SimpleNamespace(
        MonocularDepthAnything3=lambda _: SimpleNamespace(
            load_state_dict=lambda *a, **kw: SimpleNamespace(
                missing_keys=["model.backbone.weight"], unexpected_keys=[]
            )
        ),
        _normalize_state_dict=lambda state, _: state,
        _unwrap_state_dict=lambda state: state,
        _load_checkpoint_payload=lambda _: {},
    )
    monkeypatch.setattr(export.importlib, "import_module", lambda _: module)
    with pytest.raises(ValueError, match="Invalid Limbo checkpoint"):
        export.exportStreamingDepth(checkpoint, tmp_path / "cache", 280, 504)
    assert not list((tmp_path / "cache").glob("*.onnx"))


def testUnsupportedExportShapeFailsBeforeLoading():
    pytest.importorskip("torch")
    from src.depth.streaming_export import exportStreamingDepth

    with pytest.raises(ValueError, match="504x280 or 504x378"):
        exportStreamingDepth("missing.safetensors", "unused", 512, 512)


def testHalfExportKeepsAttentionProductsInFloat32(tmp_path):
    pytest.importorskip("torch")
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime.transformers.float16")
    from src.depth.streaming_export import to_fp16

    helper, tensor = onnx.helper, onnx.TensorProto
    graph = helper.make_graph(
        [
            helper.make_node(
                "MatMul",
                ["image", "key"],
                ["depth"],
                name="/backbone/blocks.4/attn/MatMul",
            )
        ],
        "attention",
        [
            helper.make_tensor_value_info(name, tensor.FLOAT, [2, 2])
            for name in ("image", "key")
        ],
        [helper.make_tensor_value_info("depth", tensor.FLOAT, [2, 2])],
    )
    src, dst = tmp_path / "float.onnx", tmp_path / "half.onnx"
    onnx.save(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), src
    )
    to_fp16(str(src), str(dst))
    model = onnx.shape_inference.infer_shapes(onnx.load(dst))
    types = {
        v.name: v.type.tensor_type.elem_type
        for v in [*model.graph.input, *model.graph.value_info, *model.graph.output]
    }
    (product,) = [node for node in model.graph.node if node.op_type == "MatMul"]
    assert all(
        types[name] == tensor.FLOAT for name in [*product.input, *product.output]
    )
    assert types["image"] == tensor.FLOAT16
