import torch
import tensorrt as trt
import os
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Union

from src.utils.logAndPrint import logAndPrint, coloredPrint
from src.constants import ADOBE


def createNetworkAndConfig(
    builder: trt.Builder, maxWorkspaceSize: int, fp16: bool
) -> Tuple[trt.INetworkDefinition, trt.IBuilderConfig]:
    """Create TensorRT network and builder configuration."""
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, maxWorkspaceSize)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    return network, config


def parseModel(parser: trt.OnnxParser, modelPath: str) -> bool:
    """Parse ONNX model with enhanced error handling."""
    if not os.path.exists(modelPath):
        logAndPrint(f"Model file not found: {modelPath}", "red")
        return False

    try:
        with open(modelPath, "rb") as model:
            modelData = model.read()
            if not modelData:
                logAndPrint(f"Empty model file: {modelPath}", "red")
                return False

            if not parser.parse(modelData):
                logAndPrint("Failed to parse ONNX model:", "red")
                for error in range(parser.num_errors):
                    errorMSG = parser.get_error(error)
                    logAndPrint(f"  Parser error {error}: {errorMSG}", "red")
                    logging.error(f"ONNX parser error {error}: {errorMSG}")
                return False
        return True
    except Exception as e:
        logAndPrint(f"Error reading model file {modelPath}: {e}", "red")
        logging.error(f"Error reading model file {modelPath}: {e}")
        return False


def setOptimizationProfile(
    builder: trt.Builder,
    config: trt.IBuilderConfig,
    inputName: List[str],
    inputsMin: Union[List[torch.Tensor], torch.Tensor],
    inputsOpt: Union[List[torch.Tensor], torch.Tensor],
    inputsMax: Union[List[torch.Tensor], torch.Tensor],
    isMultiInput: bool,
) -> bool:
    """Set optimization profile with improved error handling and validation."""
    try:
        profile = builder.create_optimization_profile()

        if isMultiInput:
            if not all(isinstance(x, list) for x in [inputsMin, inputsOpt, inputsMax]):
                logAndPrint("Multi-input mode requires list inputs", "red")
                return False

            if not all(
                len(x) == len(inputName) for x in [inputsMin, inputsOpt, inputsMax]
            ):
                logAndPrint("Input tensors and names must have same length", "red")
                return False

            for name, minShape, optShape, maxShape in zip(
                inputName, inputsMin, inputsOpt, inputsMax
            ):
                profile.set_shape(
                    name, tuple(minShape), tuple(optShape), tuple(maxShape)
                )
                _logInputShapes(name, minShape, optShape, maxShape)
        else:
            if len(inputName) == 0:
                logAndPrint("Input name list cannot be empty", "red")
                return False

            profile.set_shape(
                inputName[0], tuple(inputsMin), tuple(inputsOpt), tuple(inputsMax)
            )
            _logInputShapes(inputName[0], inputsMin, inputsOpt, inputsMax)

        config.add_optimization_profile(profile)
        return True

    except Exception as e:
        logAndPrint(f"Error setting optimization profile: {e}", "red")
        logging.error(f"Error setting optimization profile: {e}")
        return False


def _logInputShapes(name: str, minShape, optShape, maxShape) -> None:
    """Helper function to log input shapes consistently."""
    if not ADOBE:
        # UTF8 Parsing of those lines sucks
        coloredPrint(
            f"╭─ Input: {name}\n"
            f"├─ Min: {minShape}\n"
            f"├─ Opt: {optShape}\n"
            f"╰─ Max: {maxShape}",
        )
    logging.info(f"Input: {name} - Min: {minShape}, Opt: {optShape}, Max: {maxShape}")


def tensorRTEngineCreator(
    modelPath: str = "",
    enginePath: str = "model.engine",
    fp16: bool = False,
    inputsMin: Union[List[torch.Tensor], torch.Tensor] = [],
    inputsOpt: Union[List[torch.Tensor], torch.Tensor] = [],
    inputsMax: Union[List[torch.Tensor], torch.Tensor] = [],
    inputName: List[str] = None,
    maxWorkspaceSize: int = (1 << 30),
    optimizationLevel: int = 3,
    forceStatic: bool = False,
    isMultiInput: bool = False,
) -> Tuple[Optional[trt.ICudaEngine], Optional[trt.IExecutionContext]]:
    """
    Create a TensorRT engine from an ONNX model with enhanced validation and error handling.

    Parameters:
        modelPath (str): The path to the ONNX model.
        enginePath (str): The path to save the engine.
        fp16 (bool): Use half precision for the engine.
        inputsMin: The minimum shape(s) that the profile will support.
        inputsOpt: The shape(s) for which TensorRT will optimize the engine.
        inputsMax: The maximum shape(s) that the profile will support.
        inputName (List[str]): The names of the input tensors.
        maxWorkspaceSize (int): The maximum GPU memory that the engine will use.
        optimizationLevel (int): The optimization level for the engine.
        forceStatic (bool): Force static shapes for all inputs.
        isMultiInput (bool): Whether the model has multiple inputs.

    Returns:
        Tuple of (engine, context) or (None, None) on failure.
    """
    # Input validation
    if not modelPath or not os.path.exists(modelPath):
        logAndPrint(f"Invalid model path: {modelPath}", "red")
        return None, None

    if inputName is None:
        inputName = ["input"]

    if not inputName:
        logAndPrint("Input name list cannot be empty", "red")
        return None, None

    if not all([inputsMin, inputsOpt, inputsMax]) and not forceStatic:
        logAndPrint("Input shapes must be provided unless forceStatic is True", "red")
        return None, None

    logAndPrint(
        f"Model engine not found, creating engine for model: {modelPath}",
        "yellow",
    )

    if forceStatic:
        inputsMin = inputsOpt
        inputsMax = inputsOpt

    try:
        TRTLOGGER = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(TRTLOGGER)
        network, config = createNetworkAndConfig(builder, maxWorkspaceSize, fp16)

        parser = trt.OnnxParser(network, TRTLOGGER)
        if not parseModel(parser, modelPath):
            return None, None

        if not setOptimizationProfile(
            builder, config, inputName, inputsMin, inputsOpt, inputsMax, isMultiInput
        ):
            return None, None

        logAndPrint(
            f"Building a serialized engine for {os.path.basename(modelPath)}. This may take a moment.",
            "green",
        )

        serializedEngine = builder.build_serialized_network(network, config)
        if not serializedEngine:
            logAndPrint("Failed to build serialized engine", "red")
            return None, None

        logAndPrint("Serialized engine built successfully!", "green")

        os.makedirs(os.path.dirname(enginePath), exist_ok=True)

        with open(enginePath, "wb") as f:
            f.write(serializedEngine)

        engine, context = tensorRTEngineLoader(enginePath)
        if engine is None:
            logAndPrint("Failed to load created engine", "red")
            return None, None

        logAndPrint(f"Engine saved to {enginePath}", "yellow")
        return engine, context

    except Exception as e:
        logAndPrint(f"Error creating TensorRT engine: {e}", "red")
        logging.error(f"Error creating TensorRT engine: {e}")
        return None, None


def tensorRTEngineLoader(
    enginePath: str,
) -> Tuple[Optional[trt.ICudaEngine], Optional[trt.IExecutionContext]]:
    """
    Load a TensorRT engine from a file with enhanced error handling.

    Parameters:
        enginePath (str): The path to the engine file.

    Returns:
        Tuple of (engine, context) or (None, None) on failure.
    """
    if not enginePath or not os.path.exists(enginePath):
        return None, None

    try:
        with (
            open(enginePath, "rb") as f,
            trt.Runtime(trt.Logger(trt.Logger.INFO)) as runtime,
        ):
            engine_data = f.read()
            if not engine_data:
                logAndPrint(f"Empty engine file: {enginePath}", "red")
                return None, None

            engine = runtime.deserialize_cuda_engine(engine_data)
            if not engine:
                logAndPrint(f"Failed to deserialize engine: {enginePath}", "red")
                return None, None

            context = engine.create_execution_context()
            if not context:
                logAndPrint(f"Failed to create execution context: {enginePath}", "red")
                return None, None

            return engine, context

    except FileNotFoundError:
        return None, None
    except Exception as e:
        logAndPrint(
            f"Model engine is outdated due to a TensorRT Update, creating a new engine. Error: {e}",
            "yellow",
        )
        logging.warning(f"Engine loading failed: {e}")
        return None, None


def tensorRTEngineNameHandler(
    modelPath: str = "",
    fp16: bool = False,
    optInputShape: List[int] = None,
    ensemble: bool = False,
    isRife: bool = False,
) -> str:
    """
    Create a name for the TensorRT engine file with validation.

    Parameters:
        modelPath (str): The path to the ONNX / PTH model.
        fp16 (bool): Use half precision for the engine.
        optInputShape (List[int]): The shape for which TensorRT will optimize the engine.
        ensemble (bool): Whether this is an ensemble model.
        isRife (bool): Whether this is a RIFE model.

    Returns:
        str: The generated engine file path.

    Raises:
        ValueError: If inputs are invalid.
    """
    if not modelPath:
        raise ValueError("Model path cannot be empty")

    if optInputShape is None or len(optInputShape) < 4:
        raise ValueError("optInputShape must have at least 4 dimensions")

    enginePrecision = "fp16" if fp16 else "fp32"
    height, width = optInputShape[2], optInputShape[3]

    modelPath = Path(modelPath)
    if modelPath.suffix not in [".onnx", ".pth"]:
        raise ValueError(
            f"Unsupported model file extension: {modelPath.suffix}. Only .onnx and .pth are supported."
        )

    nameParts = [f"_{enginePrecision}_{height}x{width}"]

    if isRife and ensemble:
        nameParts.append("_ensemble")

    engineName = "".join(nameParts) + ".engine"
    return str(modelPath.with_suffix("")) + engineName
