import torch
import tensorrt as trt
import logging

from typing import List, Tuple
from src.coloredPrints import yellow

from polygraphy.backend.trt import (
    engine_from_network,
    network_from_onnx_path,
    CreateConfig,
    Profile,
    SaveEngine,
)


def TensorRTEngineCreator(
    modelPath: str = "",
    enginePath: str = "model.engine",
    fp16: bool = False,
    inputsMin: List[torch.Tensor] = [],
    inputsOpt: List[torch.Tensor] = [],
    inputsMax: List[torch.Tensor] = [],
    inputName: str = "input",
    maxWorkspaceSize: int = (1 << 30),
    optimizationLevel: int = 3,
    forceRebuild: bool = False,
    forceStatic: bool = False,
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
        inputName (str): The name of the input tensor.
        maxWorkspaceSize (int): The maximum GPU memory that the engine will use.
        optimizationLevel (int): The optimization level for the engine.
    """
    toPrint = f"Model engine not found, creating engine for model: {modelPath}, this may take a while..."
    print(yellow(toPrint))
    logging.info(toPrint)
    if forceStatic:
        inputsMin = inputsOpt
        inputsMax = inputsOpt
        
    profiles = [
        Profile().add(
            inputName,
            min=tuple(inputsMin),
            opt=tuple(inputsOpt),
            max=tuple(inputsMax),
        ),
    ]
    engine = engine_from_network(
        network_from_onnx_path(modelPath),
        config=CreateConfig(fp16=fp16, profiles=profiles, preview_features=[]),
    )
    engine = SaveEngine(engine, enginePath)
    engine.__call__()

    with open(enginePath, "rb") as f, trt.Runtime(
        trt.Logger(trt.Logger.INFO)
    ) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

    return engine, context


def TensorRTEngineLoader(
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

    except Exception:
        print(yellow("Model engine was found but it is outdated due to a Driver or TensorRT Update, creating a new engine."))
        return None, None


def TensorRTEngineNameHandler(
    modelPath: str = "",
    fp16: bool = False,
    optInputShape: List[int] = [],
) -> str:
    """
    Create a name for the TensorRT engine file.
    
    Parameters:
        modelPath (str): The path to the ONNX model.
        fp16 (bool): Use half precision for the engine.
        optInputShape (List[int]): The shape for which TensorRT will optimize the engine.
    """
    enginePrecision = "fp16" if fp16 else "fp32"
    height, width = optInputShape[2], optInputShape[3]
    return modelPath.replace(
        ".onnx", f"_{enginePrecision}_{height}x{width}.engine"
    )
