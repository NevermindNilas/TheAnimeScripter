import onnx
import os
import onnxslim

from onnx import version_converter
from onnxconverter_common import float16

OPSET = 21
IR_VERSION = 9

modelList = [r"C:\Users\nilas\Downloads\AniOnnx.onnx"]


def set_ir_version(model, ir_version):
    model.ir_version = ir_version
    return model


for modelPath in modelList:
    model = onnx.load(modelPath)

    # Convert to fp16
    modelFp16 = float16.convert_float_to_float16(model)
    modelFp16 = version_converter.convert_version(modelFp16, OPSET)
    modelFp16 = set_ir_version(modelFp16, IR_VERSION)
    newModelPathFp16 = modelPath.replace(".onnx", f"_fp16_op{OPSET}.onnx")
    onnx.save(modelFp16, newModelPathFp16)
    savedModelFp16 = onnx.load(newModelPathFp16)
    print("Opset version for fp16:", savedModelFp16.opset_import[0].version)
    print("IR version for fp16:", savedModelFp16.ir_version)
    slimPathFp16 = newModelPathFp16.replace(".onnx", "_slim.onnx")
    print(f"{newModelPathFp16} {slimPathFp16}")
    slimPathFp16 = onnxslim.slim(newModelPathFp16, slimPathFp16)
    os.remove(newModelPathFp16)

    # Convert to fp32
    modelFp32 = version_converter.convert_version(model, OPSET)
    modelFp32 = set_ir_version(modelFp32, IR_VERSION)
    newModelPathFp32 = modelPath.replace(".onnx", f"_fp32_op{OPSET}.onnx")
    onnx.save(modelFp32, newModelPathFp32)
    savedModelFp32 = onnx.load(newModelPathFp32)
    print("Opset version for fp32:", savedModelFp32.opset_import[0].version)
    print("IR version for fp32:", savedModelFp32.ir_version)
    slimPathFp32 = newModelPathFp32.replace(".onnx", "_slim.onnx")
    print(f"{newModelPathFp32} {slimPathFp32}")
    slimPathFp32 = onnxslim.slim(newModelPathFp32, slimPathFp32)
    os.remove(newModelPathFp32)
