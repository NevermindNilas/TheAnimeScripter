import torch
import tensorrt as trt
import logging
from typing import List, Tuple
from .coloredPrints import yellow, cyan


def logAndPrint(message: str, colorFunc):
    print(colorFunc(message))
    logging.info(message)


def createNetworkAndConfig(
    builder: trt.Builder, maxWorkspaceSize: int, fp16: bool
) -> Tuple[trt.INetworkDefinition, trt.IBuilderConfig]:
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, maxWorkspaceSize)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    return network, config


def parseModel(parser: trt.OnnxParser, modelPath: str) -> bool:
    with open(modelPath, "rb") as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False
    return True


def setOptimizationProfile(
    builder: trt.Builder,
    config: trt.IBuilderConfig,
    inputName: List[str],
    inputsMin: List[torch.Tensor],
    inputsOpt: List[torch.Tensor],
    inputsMax: List[torch.Tensor],
    isMultiInput: bool,
):
    profile = builder.create_optimization_profile()
    if isMultiInput:
        for name, minShape, optShape, maxShape in zip(
            inputName, inputsMin, inputsOpt, inputsMax
        ):
            profile.set_shape(name, minShape, optShape, maxShape)
    else:
        profile.set_shape(inputName[0], inputsMin[0], inputsOpt[0], inputsMax[0])
    config.add_optimization_profile(profile)


def tensorRTEngineCreator(
    modelPath: str = "",
    enginePath: str = "model.engine",
    fp16: bool = False,
    inputsMin: List[torch.Tensor] = [],
    inputsOpt: List[torch.Tensor] = [],
    inputsMax: List[torch.Tensor] = [],
    inputName: List[str] = ["input"],
    maxWorkspaceSize: int = (1 << 30),
    optimizationLevel: int = 3,
    forceStatic: bool = False,
    isMultiInput: bool = False,
) -> Tuple[trt.ICudaEngine, trt.IExecutionContext]:
    """
    Create a TensorRT engine from an ONNX model.

    Parameters:
        modelPath (str): The path to the ONNX model.
        enginePath (str): The path to save the engine.
        fp16 (bool): Use half precision for the engine.
        inputsMin (List[torch.Tensor]): The minimum shape that the profile will support.
        inputsOpt (List[torch.Tensor]): The shape for which TensorRT will optimize the engine.
        inputsMax (List[torch.Tensor]): The maximum shape that the profile will support.
        inputName (List[str]): The names of the input tensors.
        maxWorkspaceSize (int): The maximum GPU memory that the engine will use.
        optimizationLevel (int): The optimization level for the engine.
    """
    logAndPrint(
        f"Model engine not found, creating engine for model: {modelPath}, this may take a while...",
        yellow,
    )

    if forceStatic:
        inputsMin = inputsOpt
        inputsMax = inputsOpt

    shapeInfo = "\n".join(
        [
            f"  {name:<10}: Min: {str(minShape):<20}, Opt: {str(optShape):<20}, Max: {str(maxShape):<20}".replace(
                "]  ,", "],"
            )
            for name, minShape, optShape, maxShape in zip(
                inputName, inputsMin, inputsOpt, inputsMax
            )
        ]
    )
    logAndPrint(f"Optimizing for shapes:\n{shapeInfo}", cyan)

    TRTLOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRTLOGGER)
    network, config = createNetworkAndConfig(builder, maxWorkspaceSize, fp16)

    parser = trt.OnnxParser(network, TRTLOGGER)
    if not parseModel(parser, modelPath):
        return None, None

    setOptimizationProfile(
        builder, config, inputName, inputsMin, inputsOpt, inputsMax, isMultiInput
    )

    serializedEngine = builder.build_serialized_network(network, config)
    with open(enginePath, "wb") as f:
        f.write(serializedEngine)

    with open(enginePath, "rb") as f, trt.Runtime(TRTLOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

    return engine, context


def tensorRTEngineLoader(
    enginePath: str,
) -> Tuple[trt.ICudaEngine, trt.IExecutionContext]:
    """
    Load a TensorRT engine from a file.

    Parameters:
        enginePath (str): The path to the engine file.
    """
    try:
        with open(enginePath, "rb") as f, trt.Runtime(
            trt.Logger(trt.Logger.INFO)
        ) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
            context = engine.create_execution_context()
            return engine, context
    except FileNotFoundError:
        return None, None
    except Exception as e:
        print(
            f"Model engine was found but it is outdated due to a Driver or TensorRT Update, creating a new engine. Error: {e}"
        )
        return None, None


def tensorRTEngineNameHandler(
    modelPath: str = "",
    fp16: bool = False,
    optInputShape: List[int] = [],
    ensemble: bool = False,
    isRife: bool = False,
) -> str:
    """
    Create a name for the TensorRT engine file.

    Parameters:
        modelPath (str): The path to the ONNX / PTH model.
        fp16 (bool): Use half precision for the engine.
        optInputShape (List[int]): The shape for which TensorRT will optimize the engine.
    """
    enginePrecision = "fp16" if fp16 else "fp32"
    height, width = optInputShape[2], optInputShape[3]

    if modelPath.endswith(".onnx"):
        extension = ".onnx"
    elif modelPath.endswith(".pth"):
        extension = ".pth"
    else:
        raise ValueError(
            "Unsupported model file extension. Only .onnx and .pth are supported."
        )

    name = [f"_{enginePrecision}_{height}x{width}"]
    if isRife:
        if ensemble:
            name.append("_ensemble")
        return modelPath.replace(extension, "".join(name) + ".engine")

    return modelPath.replace(extension, f"_{enginePrecision}_{height}x{width}.engine")
