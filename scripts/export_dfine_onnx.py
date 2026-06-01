"""
Export a D-FINE checkpoint to the TAS RAW-head ONNX contract.

TAS consumes D-FINE as a SINGLE-input / TWO-output ONNX (no baked post-processor,
no ``orig_target_sizes`` input):

    input   images       float [1, 3, 640, 640]   RGB, /255, NCHW
    output  pred_logits  float [1, 300, 80]        raw per-class logits
    output  pred_boxes   float [1, 300, 4]         cxcywh, normalized to [0, 1]

This matches ``src/objectDetection/dfine.py::decodeDFine`` (sigmoid -> top-k ->
cxcywh->xyxy -> de-normalize), and keeps the engine single-input so TAS's TRT
binding loop and ORT feed dict stay unchanged.

------------------------------------------------------------------------------
THIS SCRIPT IS A MAINTAINER-ONLY OFFLINE HELPER. It must be run from inside a
D-FINE checkout (https://github.com/Peterande/D-FINE, Apache-2.0) where
``src.core.YAMLConfig`` is importable and the matching config + checkpoint exist.
It is intentionally NOT imported by the TAS runtime.

Usage (run inside the D-FINE repo, with TAS scripts/ on PYTHONPATH or copied in):

    python export_dfine_onnx.py \
        --config configs/dfine/dfine_hgnetv2_s_coco.yml \
        --resume dfine_s_coco.pth \
        --size 640 \
        --out dfine_small

Produces dfine_small_fp32.onnx and (with --fp16) dfine_small_fp16.onnx.
Map the D-FINE sizes onto TAS names: S -> dfine_small, M -> dfine_medium,
L -> dfine_large. Prefer DEIM-trained D-FINE weights (identical graph) for the
accuracy bump. Use COCO-only checkpoints for clean commercial provenance.
Then host both files on the TAS-Models-Host ``main`` release.
"""

import argparse
import os


def parseArgs():
    parser = argparse.ArgumentParser(description="Export D-FINE to TAS RAW-head ONNX")
    parser.add_argument("--config", required=True, help="D-FINE YAML config path")
    parser.add_argument("--resume", required=True, help="D-FINE .pth checkpoint path")
    parser.add_argument("--size", type=int, default=640, help="square input size")
    parser.add_argument(
        "--out",
        required=True,
        help="output basename, e.g. dfine_small (suffix _fp32/_fp16 added)",
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="skip onnxsim simplification",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="also emit a real fp16 ONNX (via onnxconverter_common, sensitive ops kept fp32)",
    )
    parser.add_argument(
        "--dml-compat",
        action="store_true",
        help="replace grid_sample with a manual bilinear sampler (Pad/Gather). NOTE: "
        "D-FINE's DETR decoder has other ops DirectML still rejects, so this alone "
        "does NOT make DirectML work — default export keeps native GridSample, which "
        "is faster on the supported TensorRT/OpenVINO/CPU backends.",
    )
    return parser.parse_args()


def _patchGridSampleForDML():
    """Replace F.grid_sample with a manual bilinear sampler built from Pad/Gather/
    arithmetic. D-FINE's deformable attention calls grid_sample (bilinear/zeros/
    align_corners=False); the native ONNX `GridSample` op is rejected by the
    DirectML execution provider ("parameter is incorrect" at init). The manual
    version exports to ops every backend supports, so ONE ONNX runs on DirectML,
    OpenVINO, TensorRT and CPU. Numerically equivalent for this exact mode."""
    import torch
    import torch.nn.functional as F

    def _bilinearGridSample(im, grid, mode="bilinear", padding_mode="zeros", align_corners=False):
        n, c, h, w = im.shape
        gh, gw = grid.shape[1], grid.shape[2]
        x = grid[..., 0].reshape(n, -1)
        y = grid[..., 1].reshape(n, -1)
        if align_corners:
            x = (x + 1) / 2 * (w - 1)
            y = (y + 1) / 2 * (h - 1)
        else:
            x = ((x + 1) * w - 1) / 2
            y = ((y + 1) * h - 1) / 2

        x0 = torch.floor(x)
        y0 = torch.floor(y)
        x1 = x0 + 1
        y1 = y0 + 1
        wa = ((x1 - x) * (y1 - y)).unsqueeze(1)
        wb = ((x1 - x) * (y - y0)).unsqueeze(1)
        wc = ((x - x0) * (y1 - y)).unsqueeze(1)
        wd = ((x - x0) * (y - y0)).unsqueeze(1)

        imPadded = F.pad(im, (1, 1, 1, 1), mode="constant", value=0)
        ph, pw = h + 2, w + 2
        x0 = (x0 + 1).clamp(0, pw - 1).long()
        x1 = (x1 + 1).clamp(0, pw - 1).long()
        y0 = (y0 + 1).clamp(0, ph - 1).long()
        y1 = (y1 + 1).clamp(0, ph - 1).long()

        imFlat = imPadded.reshape(n, c, -1)

        def _gather(yy, xx):
            idx = (xx + yy * pw).unsqueeze(1).expand(-1, c, -1)
            return torch.gather(imFlat, 2, idx)

        out = (
            _gather(y0, x0) * wa
            + _gather(y1, x0) * wb
            + _gather(y0, x1) * wc
            + _gather(y1, x1) * wd
        )
        return out.reshape(n, c, gh, gw)

    F.grid_sample = _bilinearGridSample


def buildRawModel(config, resume, dmlCompat=False):
    """Wrap cfg.model.deploy() so forward() returns (pred_logits, pred_boxes) only."""
    import torch

    if dmlCompat:
        _patchGridSampleForDML()

    from src.core import YAMLConfig  # provided by the D-FINE repo

    cfg = YAMLConfig(config, resume=resume)

    if "HGNetv2" in getattr(cfg, "yaml_cfg", {}):
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    checkpoint = torch.load(resume, map_location="cpu")
    state = checkpoint.get("ema", {}).get("module", checkpoint.get("model", checkpoint))
    cfg.model.load_state_dict(state)

    class RawDFine(torch.nn.Module):
        def __init__(self, deployModel):
            super().__init__()
            self.model = deployModel

        def forward(self, images):
            out = self.model(images)
            # cfg.model.deploy() returns a dict with DETR-convention keys.
            return out["pred_logits"], out["pred_boxes"]

    return RawDFine(cfg.model.deploy()).eval()


def exportFp32(model, size, opset, outPath, simplify):
    import torch

    dummy = torch.randn(1, 3, size, size)
    torch.onnx.export(
        model,
        dummy,
        outPath,
        input_names=["images"],
        output_names=["pred_logits", "pred_boxes"],
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes=None,  # static 1x3xSIZExSIZE — matches TAS static engines
    )
    print(f"[fp32] wrote {outPath}")

    if simplify:
        try:
            import onnx
            from onnxsim import simplify as onnxsim_simplify

            simplified, ok = onnxsim_simplify(onnx.load(outPath))
            if ok:
                onnx.save(simplified, outPath)
                print(f"[fp32] simplified {outPath}")
            else:
                print("[fp32] onnxsim reported the model was not validated; kept raw export")
        except ImportError:
            print("[fp32] onnxsim not installed; skipping simplification")


def _convertToFloat16(model):
    """Full-fp16 conversion mirroring TAS src/utils/onnxConverter.convertToFloat16.

    Uses onnxconverter_common with keep_io_types=False (so Resize's float-only
    `scales` input and friends are handled by the converter's op rules instead of
    being left as a dangling fp32 island that breaks neighbouring ops), then forces
    graph IO + value_info to fp16 and fixes Cast targets. Produces a genuine fp16
    model with fp16 IO — the same shape every other TAS ONNX ships in. onnxslim then
    cleans up the inserted casts.
    """
    import copy
    import numpy as np
    import onnx
    from onnx import TensorProto
    from onnxconverter_common import float16

    model = copy.deepcopy(model)
    model = float16.convert_float_to_float16(
        model, keep_io_types=False, disable_shape_infer=False
    )

    for tensors in (model.graph.input, model.graph.output, model.graph.value_info):
        for t in tensors:
            if t.type.HasField("tensor_type") and (
                t.type.tensor_type.elem_type == TensorProto.FLOAT
            ):
                t.type.tensor_type.elem_type = TensorProto.FLOAT16

    def _castInit(tensor):
        if tensor.data_type == TensorProto.FLOAT:
            tensor.data_type = TensorProto.FLOAT16
            if tensor.HasField("raw_data"):
                arr = np.frombuffer(tensor.raw_data, dtype=np.float32).astype(np.float16)
                tensor.raw_data = arr.tobytes()
            elif len(tensor.float_data) > 0:
                arr = np.array(tensor.float_data, dtype=np.float32).astype(np.float16)
                tensor.ClearField("float_data")
                tensor.raw_data = arr.tobytes()

    for node in model.graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == TensorProto.FLOAT:
                    attr.i = TensorProto.FLOAT16
        for attr in node.attribute:
            if attr.HasField("t"):
                _castInit(attr.t)
            for tensor in attr.tensors:
                _castInit(tensor)

    return model


def exportFp16(fp32Path, fp16Path):
    """Real fp16 ONNX (fp16 IO), TAS-style full conversion + onnxslim cleanup."""
    import onnx

    fp16Model = _convertToFloat16(onnx.load(fp32Path))
    onnx.save(fp16Model, fp16Path)

    try:
        import onnxslim

        slimPath = fp16Path.replace(".onnx", "_slim.onnx")
        onnxslim.slim(fp16Path, slimPath)
        if os.path.exists(slimPath):
            os.replace(slimPath, fp16Path)
            print(f"[fp16] wrote + slimmed {fp16Path}")
            return
    except ImportError:
        pass
    print(f"[fp16] wrote {fp16Path} (onnxslim unavailable; not slimmed)")


def main():
    args = parseArgs()
    model = buildRawModel(args.config, args.resume, dmlCompat=args.dml_compat)

    fp32Path = f"{args.out}_fp32.onnx"
    exportFp32(model, args.size, args.opset, fp32Path, not args.no_simplify)

    if args.fp16:
        fp16Path = f"{args.out}_fp16.onnx"
        exportFp16(fp32Path, fp16Path)

    print("\nDone. Verify the two output shapes are [1,300,80] (logits) and "
          "[1,300,4] (boxes), then host on TAS-Models-Host.")
    print(f"Files: {os.path.abspath(fp32Path)}"
          + (f", {os.path.abspath(args.out + '_fp16.onnx')}" if args.fp16 else ""))


if __name__ == "__main__":
    main()
