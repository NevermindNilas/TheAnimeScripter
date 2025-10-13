import onnx
import os
import sys
import torch
from pathlib import Path
from onnxconverter_common import float16

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import onnxslim

    isOnnxSlim = True
except ImportError:
    print("onnxslim not found. Please install onnx-slim using: pip install onnxslim")
    isOnnxSlim = False

OPSET = 22
modelList = [r"C:\Users\nilas\Downloads\1xDeH264_realplksr.pth"]


def convertAndSaveModel(model, modelPath, precision, opset):
    if precision == "fp16":
        model = float16.convert_float_to_float16(model)
    newModelPath = modelPath.replace(".onnx", f"_{precision}_op{opset}.onnx")
    onnx.save(model, newModelPath)
    savedModel = onnx.load(newModelPath)
    print(f"Opset version for {precision}: {savedModel.opset_import[0].version}")
    print(f"IR version for {precision}: {savedModel.ir_version}")
    return newModelPath


def slimModel(modelPath, slimPath):
    if isOnnxSlim:
        onnxslim.slim(modelPath, slimPath)
        if os.path.exists(slimPath):
            os.remove(modelPath)
            return slimPath
        return modelPath
    else:
        print(f"onnxslim not found. Skipping {modelPath} slimming")
        return modelPath


def pthToOnnx(
    pthPath,
    outputPath=None,
    inputShape=(1, 3, 256, 256),
    precision="fp32",
    opset=OPSET,
    slim=True,
):
    from src.spandrel import ModelLoader, ImageModelDescriptor

    print(f"\n{'=' * 60}")
    print(f"Converting PyTorch model to ONNX: {pthPath}")
    print(f"{'=' * 60}")

    if not os.path.exists(pthPath):
        raise FileNotFoundError(f"Model file not found: {pthPath}")

    if outputPath is None:
        outputPath = os.path.splitext(pthPath)[0] + ".onnx"

    print("Loading model with spandrel...")
    loader = ModelLoader(device="cpu")
    modelDescriptor = loader.load_from_file(pthPath)

    if not isinstance(modelDescriptor, ImageModelDescriptor):
        raise ValueError(f"Model is not an image model. Got: {type(modelDescriptor)}")

    model = modelDescriptor.model
    model.eval()

    print("Model loaded successfully:")
    print(
        f"  - Architecture: {modelDescriptor.architecture.id if hasattr(modelDescriptor, 'architecture') else 'Unknown'}"
    )
    print(f"  - Scale: {modelDescriptor.scale}x")
    print(f"  - Input channels: {modelDescriptor.input_channels}")
    print(f"  - Output channels: {modelDescriptor.output_channels}")

    inputShape = (
        inputShape[0],
        modelDescriptor.input_channels,
        inputShape[2],
        inputShape[3],
    )

    dummyInput = torch.randn(inputShape, dtype=torch.float32)

    print(f"\nExporting to ONNX (opset {opset})...")
    print(f"  - Input shape: {inputShape}")
    print(f"  - Output path: {outputPath}")

    useDynamo = opset > 20
    if useDynamo:
        print(
            f"  - Using torch.export-based ONNX exporter (dynamo=True) for opset {opset}"
        )

    try:
        torch.onnx.export(
            model,
            dummyInput,
            outputPath,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch", 2: "height", 3: "width"},
                "output": {0: "batch", 2: "height", 3: "width"},
            },
            opset_version=opset,
            do_constant_folding=True,
            dynamo=useDynamo,
            verbose=False,
        )
    except Exception as e:
        if useDynamo and opset > 20:
            print(f"⚠ Dynamo export failed, falling back to opset 20: {e}")
            opset = 20
            torch.onnx.export(
                model,
                dummyInput,
                outputPath,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input": {0: "batch", 2: "height", 3: "width"},
                    "output": {0: "batch", 2: "height", 3: "width"},
                },
                opset_version=opset,
                do_constant_folding=True,
                verbose=False,
            )
        else:
            raise

    print("✓ ONNX export successful!")

    if precision == "fp16":
        print("\nConverting to FP16...")
        onnxModel = onnx.load(outputPath)
        onnxModel = float16.convert_float_to_float16(onnxModel)
        fp16Path = outputPath.replace(".onnx", f"_fp16_op{opset}.onnx")
        onnx.save(onnxModel, fp16Path)
        os.remove(outputPath)
        outputPath = fp16Path
        print("✓ FP16 conversion successful!")
    else:
        fp32Path = outputPath.replace(".onnx", f"_fp32_op{opset}.onnx")
        os.rename(outputPath, fp32Path)
        outputPath = fp32Path

    if slim and isOnnxSlim:
        print("\nOptimizing with onnxslim...")
        slimPath = outputPath.replace(".onnx", "_slim.onnx")
        onnxslim.slim(outputPath, slimPath)
        if os.path.exists(slimPath):
            os.remove(outputPath)
            outputPath = slimPath
            print("✓ Optimization successful!")
        else:
            print("⚠ Optimization failed, keeping original")

    if outputPath and os.path.exists(outputPath):
        finalModel = onnx.load(outputPath)
        print("\nFinal ONNX model info:")
        print(f"  - Path: {outputPath}")
        print(f"  - Opset version: {finalModel.opset_import[0].version}")
        print(f"  - IR version: {finalModel.ir_version}")
        print(f"  - File size: {os.path.getsize(outputPath) / (1024 * 1024):.2f} MB")

    print(f"\n{'=' * 60}")
    print("Conversion complete!")
    print(f"{'=' * 60}\n")

    return outputPath


for modelPath in modelList:
    if not os.path.exists(modelPath):
        print(f"Warning: Model file not found: {modelPath}")
        continue

    if modelPath.endswith(".onnx"):
        print(f"Processing ONNX model: {modelPath}")
        model = onnx.load(modelPath)

        newModelPathFp16 = convertAndSaveModel(model, modelPath, "fp16", OPSET)
        slimPathFp16 = newModelPathFp16.replace(".onnx", "_slim.onnx")
        print(f"{newModelPathFp16} -> {slimPathFp16}")
        slimModel(newModelPathFp16, slimPathFp16)

        newModelPathFp32 = convertAndSaveModel(model, modelPath, "fp32", OPSET)
        slimPathFp32 = newModelPathFp32.replace(".onnx", "_slim.onnx")
        print(f"{newModelPathFp32} -> {slimPathFp32}")
        slimModel(newModelPathFp32, slimPathFp32)

    elif modelPath.endswith((".pth", ".pt", ".ckpt", ".safetensors")):
        try:
            pthToOnnx(modelPath, precision="fp32", opset=OPSET, slim=isOnnxSlim)
            pthToOnnx(modelPath, precision="fp16", opset=OPSET, slim=isOnnxSlim)
        except Exception as e:
            print(f"Error converting {modelPath}: {e}")
            import traceback

            traceback.print_exc()
    else:
        print(f"Warning: Unsupported file type: {modelPath}")
