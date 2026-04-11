import tensorrt as trt
import os
import logging
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Union

from src.utils.logAndPrint import logAndPrint, coloredPrint
from src.constants import ADOBE

if ADOBE:
    from src.utils.aeComms import progressState


if hasattr(trt, "IProgressMonitor"):

    class TensorRTProgressMonitor(trt.IProgressMonitor):
        """ASCII progress monitor for TensorRT engine builds.

        Ported verbatim (with minor renaming) from NVIDIA's reference sample
        at samples/python/simple_progress_monitor/simple_progress_monitor.py
        in the NVIDIA/TensorRT repository.
        """

        def __init__(self):
            trt.IProgressMonitor.__init__(self)
            self._active_phases = {}
            self._step_result = True

        def phase_start(self, phase_name, parent_phase, num_steps):
            try:
                if parent_phase is not None:
                    nbIndents = 1 + self._active_phases[parent_phase]["nbIndents"]
                else:
                    nbIndents = 0
                self._active_phases[phase_name] = {
                    "title": phase_name,
                    "steps": 0,
                    "num_steps": num_steps,
                    "nbIndents": nbIndents,
                }
                self._redraw()
            except KeyboardInterrupt:
                self._step_result = False

        def phase_finish(self, phase_name):
            try:
                del self._active_phases[phase_name]
                self._redraw(blank_lines=1)
            except KeyboardInterrupt:
                self._step_result = False

        def step_complete(self, phase_name, step):
            try:
                self._active_phases[phase_name]["steps"] = step
                self._redraw()
                return self._step_result
            except KeyboardInterrupt:
                return False

        def _redraw(self, *, blank_lines=0):
            def clear_line():
                print("\x1B[2K", end="")

            def move_to_start_of_line():
                print("\x1B[0G", end="")

            def move_cursor_up(lines):
                print("\x1B[{}A".format(lines), end="")

            def progress_bar(steps, num_steps):
                INNER_WIDTH = 10
                completed_bar_chars = int(INNER_WIDTH * steps / float(num_steps))
                return "[{}{}]".format(
                    "=" * completed_bar_chars,
                    "-" * (INNER_WIDTH - completed_bar_chars),
                )

            max_cols = (
                os.get_terminal_size().columns if sys.stdout.isatty() else 200
            )

            move_to_start_of_line()
            for phase in self._active_phases.values():
                phase_prefix = "{indent}{bar} {title}".format(
                    indent=" " * phase["nbIndents"],
                    bar=progress_bar(phase["steps"], phase["num_steps"]),
                    title=phase["title"],
                )
                phase_suffix = "{steps}/{num_steps}".format(**phase)
                allowable_prefix_chars = max_cols - len(phase_suffix) - 2
                if allowable_prefix_chars < len(phase_prefix):
                    phase_prefix = (
                        phase_prefix[0 : allowable_prefix_chars - 3] + "..."
                    )
                clear_line()
                print(phase_prefix, phase_suffix)
            for _ in range(blank_lines):
                clear_line()
                print()
            move_cursor_up(len(self._active_phases) + blank_lines)
            sys.stdout.flush()

else:

    class TensorRTProgressMonitor:
        pass


def _attachProgressMonitor(config: trt.IBuilderConfig) -> None:
    if ADOBE:
        return

    if not sys.stdout.isatty():
        return

    if not hasattr(trt, "IProgressMonitor"):
        return

    try:
        config.progress_monitor = TensorRTProgressMonitor()
    except Exception as error:
        logging.debug(f"TensorRT progress monitor is unavailable: {error}")


def createNetworkAndConfig(
    builder: trt.Builder,
    maxWorkspaceSize: int,
) -> Tuple[trt.INetworkDefinition, trt.IBuilderConfig]:
    """Create TensorRT network and builder configuration."""
    networkFlags = 0
    networkFlags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)

    network = builder.create_network(networkFlags)

    config = builder.create_builder_config()
    _attachProgressMonitor(config)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, maxWorkspaceSize)
    return network, config


def parseModel(parser: trt.OnnxParser, modelPath: str) -> bool:
    """Parse ONNX model with enhanced error handling."""
    if not os.path.exists(modelPath):
        logAndPrint(f"Model file not found: {modelPath}", "red")
        return False

    try:
        if not parser.parse_from_file(modelPath):
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
    inputsMin: Union[List[Tuple[int, ...]], Tuple[int, ...]],
    inputsOpt: Union[List[Tuple[int, ...]], Tuple[int, ...]],
    inputsMax: Union[List[Tuple[int, ...]], Tuple[int, ...]],
    isMultiInput: bool,
    fp16: bool = False,
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
                _logInputShapes(name, minShape, optShape, maxShape, fp16)
        else:
            if len(inputName) == 0:
                logAndPrint("Input name list cannot be empty", "red")
                return False

            profile.set_shape(
                inputName[0], tuple(inputsMin), tuple(inputsOpt), tuple(inputsMax)
            )
            _logInputShapes(inputName[0], inputsMin, inputsOpt, inputsMax, fp16)

        config.add_optimization_profile(profile)
        return True

    except Exception as e:
        logAndPrint(f"Error setting optimization profile: {e}", "red")
        logging.error(f"Error setting optimization profile: {e}")
        return False


def _logInputShapes(name: str, minShape, optShape, maxShape, fp16) -> None:
    """Helper function to log input shapes consistently."""
    if not ADOBE:
        # UTF8 Parsing of those lines sucks
        precision = "FP16" if fp16 else "FP32"
        coloredPrint(
            f"╭─ Input: {name} | {precision} \n"
            f"├─ Min: {minShape}\n"
            f"├─ Opt: {optShape}\n"
            f"╰─ Max: {maxShape}",
        )
    logging.info(f"Input: {name} - Min: {minShape}, Opt: {optShape}, Max: {maxShape}")


def tensorRTEngineCreator(
    modelPath: str = "",
    enginePath: str = "model.engine",
    fp16: bool = False,
    inputsMin: Union[List[Tuple[int, ...]], Tuple[int, ...]] = [],
    inputsOpt: Union[List[Tuple[int, ...]], Tuple[int, ...]] = [],
    inputsMax: Union[List[Tuple[int, ...]], Tuple[int, ...]] = [],
    inputName: Optional[List[str]] = None,
    maxWorkspaceSize: int = (1 << 30),
    optimizationLevel: int = 3,
    forceStatic: bool = False,
    isMultiInput: bool = False,
    isRife: bool = False,
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
        isRife (bool): Whether the model is a RIFE model.

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

    if ADOBE:
        progressState.update(
            {
                "status": f"Creating a TensorRT engine for {os.path.basename(modelPath)}.",
            }
        )

    if forceStatic:
        inputsMin = inputsOpt
        inputsMax = inputsOpt

    try:
        TRTLOGGER = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(TRTLOGGER)
        network, config = createNetworkAndConfig(builder, maxWorkspaceSize)

        parser = trt.OnnxParser(network, TRTLOGGER)
        if not parseModel(parser, modelPath):
            return None, None

        if not setOptimizationProfile(
            builder,
            config,
            inputName,
            inputsMin,
            inputsOpt,
            inputsMax,
            isMultiInput,
            fp16,
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

        engineDir = os.path.dirname(enginePath)
        if engineDir:
            os.makedirs(engineDir, exist_ok=True)

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
            engineData = f.read()
            if not engineData:
                logAndPrint(f"Empty engine file: {enginePath}", "red")
                return None, None

            engine = runtime.deserialize_cuda_engine(engineData)
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
