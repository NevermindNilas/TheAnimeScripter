import onnx
import os
from onnx import version_converter
from onnxconverter_common import float16

try:
    import onnxslim

    isOnnxSlim = True
except ImportError:
    print("onnxslim not found. Please install onnx-slim using: pip install onnxslim")
    isOnnxSlim = False

OPSET = 21
modelList = [r"C:\Users\nilas\AppData\Roaming\TheAnimeScripter\weights\rife4.6\rife46.onnx"]


def convertAndSaveModel(model, modelPath, precision, opset):
    if precision == "fp16":
        model = float16.convert_float_to_float16(model)
    model = version_converter.convert_version(model, opset)
    newModelPath = modelPath.replace(".onnx", f"_{precision}_op{opset}.onnx")
    onnx.save(model, newModelPath)
    savedModel = onnx.load(newModelPath)
    print(f"Opset version for {precision}: {savedModel.opset_import[0].version}")
    print(f"IR version for {precision}: {savedModel.ir_version}")
    return newModelPath


def slimModel(modelPath, slimPath):
    if isOnnxSlim:
        slimPath = onnxslim.slim(modelPath, slimPath)
        os.remove(modelPath)
    else:
        print(f"onnxslim not found. Skipping {modelPath} slimming")
    return slimPath


for modelPath in modelList:
    model = onnx.load(modelPath)

    newModelPathFp16 = convertAndSaveModel(model, modelPath, "fp16", OPSET)
    slimPathFp16 = newModelPathFp16.replace(".onnx", "_slim.onnx")
    print(f"{newModelPathFp16} {slimPathFp16}")
    slimModel(newModelPathFp16, slimPathFp16)

    newModelPathFp32 = convertAndSaveModel(model, modelPath, "fp32", OPSET)
    slimPathFp32 = newModelPathFp32.replace(".onnx", "_slim.onnx")
    print(f"{newModelPathFp32} {slimPathFp32}")
    slimModel(newModelPathFp32, slimPathFp32)
