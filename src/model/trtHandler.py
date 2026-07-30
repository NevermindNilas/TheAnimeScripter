import hashlib
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from src.constants import ADOBE
from src.infra.logAndPrint import coloredPrint, logAndPrint
from src.utils.tensorrt_import import trt

if ADOBE:
    from src.server.aeComms import progressState


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
                print("\x1b[2K", end="")

            def move_to_start_of_line():
                print("\x1b[0G", end="")

            def move_cursor_up(lines):
                print(f"\x1b[{lines}A", end="")

            def progress_bar(steps, num_steps):
                INNER_WIDTH = 10
                completed_bar_chars = int(INNER_WIDTH * steps / float(num_steps))
                return "[{}{}]".format(
                    "=" * completed_bar_chars,
                    "-" * (INNER_WIDTH - completed_bar_chars),
                )

            max_cols = os.get_terminal_size().columns if sys.stdout.isatty() else 200

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
                    phase_prefix = phase_prefix[0 : allowable_prefix_chars - 3] + "..."
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


_TIMING_CACHE_MAGIC = b"TASTIMINGCACHE1\n"


def _timingCachePath(maxWorkspaceSize: int) -> str | None:
    """Path of the build-time tactic timing cache for this workspace budget.

    The cache only stores kernel *measurements*, never an engine, so one file is
    shared by every model built with the same budget. In practice that sharing
    buys nothing across TAS's models -- the key covers layer shape, dtype and
    format, and priming the cache with the compact upscaler left a first
    `rife4.25` build unchanged (27.0s against 24.0s cold, i.e. noise). What it
    does pay for is rebuilding the *same* engine: a retry after an interrupted
    build went 8.9s -> 2.4s.

    ``maxWorkspaceSize`` is part of the name because it is NOT part of TensorRT's
    own cache key. Tactics that do not fit the workspace are dropped before they
    are timed, so a 1 GiB build that reuses entries measured under the 4 GiB
    budget `src/depth/backends/tensorrt.py` requests for VideoDepthAnything can
    adopt a tactic it would never have picked alone -- making the engine depend
    on the order unrelated models happened to be built in.

    The rest of the name is the GPU and the TensorRT and CUDA versions, so a
    cache from elsewhere is skipped without ever being opened. That matters more
    than it looks: see _readTimingCache for what TensorRT does with bad bytes.
    """
    if not all(
        hasattr(trt.IBuilderConfig, name)
        for name in ("create_timing_cache", "set_timing_cache", "get_timing_cache")
    ):
        return None

    try:
        import torch

        from src.model.registry import weightsDir

        if not torch.cuda.is_available():
            return None
        device = torch.cuda.get_device_name(0)
        # The CUDA toolkit torch was built against, NOT the installed driver --
        # it does not move when the driver is updated. It is in the name because
        # a torch/CUDA rebuild is its own invalidation axis; a driver bump is
        # caught later instead, by set_timing_cache refusing the verification
        # header, which _attachTimingCache recovers from with a fresh cache.
        cudaVersion = torch.version.cuda or "nocuda"
    except Exception as error:
        logging.debug(f"Timing cache disabled: {error}")
        return None

    def slugify(value: str) -> str:
        return "".join(c if c.isalnum() else "_" for c in str(value)).strip("_")

    version = getattr(trt, "__version__", "unknown")
    workspace = f"ws{int(maxWorkspaceSize) >> 20}m"
    name = (
        f".trt_timing_{slugify(device)}_{slugify(version)}"
        f"_cu{slugify(cudaVersion)}_{workspace}.cache"
    )
    return os.path.join(weightsDir, name)


def _readTimingCache(cachePath: str) -> bytes:
    """Return the stored tactic blob, or b"" if it cannot be trusted.

    Handing TensorRT a damaged cache does not raise and cannot be caught: a
    single flipped byte in the serialized blob makes create_timing_cache abort
    the process outright with `LLVM ERROR: out of memory` (reproduced on 10.16 by
    flipping byte 124 of a 198-byte cache; exit code 127). A torn write, bit rot
    or a power loss mid-replace would therefore brick every subsequent TAS run
    that builds an engine, until the user found and deleted a hidden dotfile.

    So the payload is wrapped in our own magic + SHA-256 header and verified here
    first. Anything that fails verification is deleted and treated as absent --
    TensorRT never sees bytes we have not checksummed.
    """
    try:
        with open(cachePath, "rb") as f:
            raw = f.read()
    except OSError:
        return b""

    try:
        if raw.startswith(_TIMING_CACHE_MAGIC):
            digest, _, payload = raw[len(_TIMING_CACHE_MAGIC) :].partition(b"\n")
            if payload and hashlib.sha256(payload).hexdigest().encode() == digest:
                return payload
        logging.debug(f"Discarding an unverifiable timing cache at {cachePath}")
    except Exception as error:
        logging.debug(f"Could not verify the timing cache at {cachePath}: {error}")

    try:
        os.remove(cachePath)
    except OSError:
        pass
    return b""


def _attachTimingCache(config: trt.IBuilderConfig, maxWorkspaceSize: int) -> str | None:
    """Attach the persistent timing cache to ``config``.

    Returns the path to write back to after the build, or None when caching is
    unavailable. A missing or unverifiable cache must never abort an engine build
    -- the worst outcome is a normal, uncached build.
    """
    cachePath = _timingCachePath(maxWorkspaceSize)
    if cachePath is None:
        return None

    buffer = _readTimingCache(cachePath)

    # A stored cache is tried first, an empty one second. The retry is what makes
    # the feature self-healing: set_timing_cache REFUSES a cache whose
    # verification header does not match this machine, and it reports that by
    # returning False rather than raising, so without re-attaching a fresh cache
    # the build would run uncached, get_timing_cache() would hand back None, and
    # the unusable file would never be overwritten -- permanently, silently dead.
    for candidate in (buffer, b"") if buffer else (b"",):
        try:
            cache = config.create_timing_cache(candidate)
            if cache is None:
                continue
            # ignore_mismatch stays False on purpose: accepting entries timed on
            # different hardware would pick tactics benchmarked on someone
            # else's GPU.
            if not config.set_timing_cache(cache, False):
                logging.debug(
                    f"TensorRT refused the timing cache at {cachePath} "
                    f"({len(candidate)} bytes); starting a new one"
                )
                continue
            return cachePath
        except Exception as error:
            logging.debug(f"Timing cache unusable ({len(candidate)} bytes): {error}")

    return None


def _saveTimingCache(config: trt.IBuilderConfig, cachePath: str | None) -> None:
    """Persist the timings measured during this build, atomically."""
    if not cachePath:
        return

    tempPath = f"{cachePath}.{os.getpid()}.tmp"
    try:
        cache = config.get_timing_cache()
        if cache is None:
            return

        # Fold in whatever landed on disk while this build was running. Without
        # it, two concurrent builds each write their own snapshot and the last
        # one silently discards the other's measurements. combine() skips
        # conflicting and foreign-version entries by itself.
        if hasattr(cache, "combine"):
            try:
                # Through _readTimingCache, never a raw read: the file is wrapped
                # in our checksum header, and create_timing_cache aborts the
                # process on bytes it cannot parse.
                onDisk = _readTimingCache(cachePath)
                if onDisk and not cache.combine(
                    config.create_timing_cache(onDisk), False
                ):
                    logging.debug(
                        "TensorRT skipped merging the on-disk timing cache; "
                        "writing this build's measurements alone"
                    )
            except Exception as error:
                logging.debug(f"Could not merge the on-disk timing cache: {error}")

        data = cache.serialize()
        # serialize() hands back an IHostMemory, which supports the buffer
        # protocol but neither len() nor a meaningful truth value -- an empty
        # cache still serializes to a non-empty header. Size comes from nbytes.
        written = getattr(data, "nbytes", 0)
        if not written:
            return

        # A build that aborted before TensorRT committed any tactic (an early
        # Ctrl-C) leaves an entry-less cache that still serializes to a ~198 byte
        # header. Writing that out would replace a file holding real measurements
        # with an empty one whenever the merge above could not read it.
        emptySize = getattr(config.create_timing_cache(b"").serialize(), "nbytes", 0)
        if emptySize and written <= emptySize:
            logging.debug("Timing cache has no entries to store; keeping the old file")
            return

        payload = bytes(memoryview(data))
        blob = (
            _TIMING_CACHE_MAGIC
            + hashlib.sha256(payload).hexdigest().encode()
            + b"\n"
            + payload
        )

        os.makedirs(os.path.dirname(cachePath), exist_ok=True)
        with open(tempPath, "wb") as f:
            f.write(blob)
            # Flush to the platter before the swap. Without it a power loss can
            # leave the directory entry pointing at unwritten blocks, i.e. the
            # exact byte-level damage _readTimingCache exists to catch.
            f.flush()
            os.fsync(f.fileno())
        # Atomic swap so a second TAS process building concurrently can never
        # read a half-written cache.
        os.replace(tempPath, cachePath)
        logging.info(f"Timing cache updated: {cachePath} ({written} bytes)")
    except Exception as error:
        logging.debug(f"Failed to write timing cache {cachePath}: {error}")
    finally:
        try:
            os.remove(tempPath)
        except OSError:
            pass


def createNetworkAndConfig(
    builder: trt.Builder,
    maxWorkspaceSize: int,
) -> tuple[trt.INetworkDefinition, trt.IBuilderConfig]:
    """Create TensorRT network and builder configuration."""
    networkFlags = 0
    networkFlags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)

    network = builder.create_network(networkFlags)

    config = builder.create_builder_config()
    _attachProgressMonitor(config)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, maxWorkspaceSize)
    return network, config


def _formatShape(shape) -> str:
    if isinstance(shape, list) and shape and isinstance(shape[0], list):
        return "[" + ", ".join(str(tuple(item)) for item in shape) + "]"
    return str(tuple(shape)) if shape else "[]"


def _logTensorRTBuildDiagnostics(
    modelPath: str,
    enginePath: str,
    fp16: bool,
    inputsMin,
    inputsOpt,
    inputsMax,
    maxWorkspaceSize: int,
    forceStatic: bool,
) -> None:
    diagnostics = [
        f"TensorRT build diagnostics: trt={getattr(trt, '__version__', 'unknown')}",
        f"precision={'fp16' if fp16 else 'fp32'}",
        f"force_static={forceStatic}",
        f"workspace_mib={maxWorkspaceSize // (1024 * 1024)}",
        f"min={_formatShape(inputsMin)}",
        f"opt={_formatShape(inputsOpt)}",
        f"max={_formatShape(inputsMax)}",
        f"model={modelPath}",
        f"engine={enginePath}",
    ]

    try:
        import torch

        diagnostics.append(f"torch={torch.__version__}")
        diagnostics.append(f"torch_cuda={torch.version.cuda}")
        if torch.cuda.is_available():
            diagnostics.append(f"gpu={torch.cuda.get_device_name(0)}")
            freeBytes, totalBytes = torch.cuda.mem_get_info()
            diagnostics.append(f"torch_vram_free_mib={freeBytes // (1024 * 1024)}")
            diagnostics.append(f"torch_vram_total_mib={totalBytes // (1024 * 1024)}")
    except Exception as error:
        diagnostics.append(f"torch_diagnostics_error={error}")

    smiPath = shutil.which("nvidia-smi")
    if smiPath:
        try:
            result = subprocess.run(
                [
                    smiPath,
                    "--query-gpu=name,memory.total,memory.used,memory.free,driver_version,compute_cap",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                diagnostics.append(f"nvidia_smi={result.stdout.strip()}")
            elif result.stderr.strip():
                diagnostics.append(f"nvidia_smi_error={result.stderr.strip()}")
        except Exception as error:
            diagnostics.append(f"nvidia_smi_error={error}")

    logging.info(" | ".join(diagnostics))


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
    inputName: list[str],
    inputsMin: list[tuple[int, ...]] | tuple[int, ...],
    inputsOpt: list[tuple[int, ...]] | tuple[int, ...],
    inputsMax: list[tuple[int, ...]] | tuple[int, ...],
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
                inputName, inputsMin, inputsOpt, inputsMax, strict=False
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
        precision = "FP16" if fp16 else "FP32"
        # A cosmetic console print must never abort the engine build. On a legacy
        # code page the box-draw glyphs can still raise (belt-and-suspenders with
        # the UTF-8 reconfigure in logAndPrint); swallow any console failure --
        # the logging.info below records the shapes regardless.
        try:
            coloredPrint(
                f"╭─ Input: {name} | {precision} \n"
                f"├─ Min: {minShape}\n"
                f"├─ Opt: {optShape}\n"
                f"╰─ Max: {maxShape}",
            )
        except Exception:
            pass
    logging.info(f"Input: {name} - Min: {minShape}, Opt: {optShape}, Max: {maxShape}")


def tensorRTEngineCreator(
    modelPath: str = "",
    enginePath: str = "model.engine",
    fp16: bool = False,
    inputsMin: list[tuple[int, ...]] | tuple[int, ...] | None = None,
    inputsOpt: list[tuple[int, ...]] | tuple[int, ...] | None = None,
    inputsMax: list[tuple[int, ...]] | tuple[int, ...] | None = None,
    inputName: list[str] | None = None,
    maxWorkspaceSize: int = (1 << 30),
    forceStatic: bool = False,
    isMultiInput: bool = False,
) -> tuple[trt.ICudaEngine | None, trt.IExecutionContext | None]:
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
        forceStatic (bool): Force static shapes for all inputs.
        isMultiInput (bool): Whether the model has multiple inputs.

    Returns:
        Tuple of (engine, context) or (None, None) on failure.
    """
    # Input validation
    if not modelPath or not os.path.exists(modelPath):
        logAndPrint(f"Invalid model path: {modelPath}", "red")
        return None, None

    if inputsMin is None:
        inputsMin = []
    if inputsOpt is None:
        inputsOpt = []
    if inputsMax is None:
        inputsMax = []

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

    _logTensorRTBuildDiagnostics(
        modelPath,
        enginePath,
        fp16,
        inputsMin,
        inputsOpt,
        inputsMax,
        maxWorkspaceSize,
        forceStatic,
    )

    try:
        TRTLOGGER = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(TRTLOGGER)
        network, config = createNetworkAndConfig(builder, maxWorkspaceSize)
        timingCachePath = _attachTimingCache(config, maxWorkspaceSize)

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
        # Written even on a failed build: the tactics timed before the failure
        # are still valid and save time on the retry.
        _saveTimingCache(config, timingCachePath)
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
) -> tuple[trt.ICudaEngine | None, trt.IExecutionContext | None]:
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
    optInputShape: list[int] = None,
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
    # Spatial dims are always the trailing two, so this also names the 5D
    # temporal video engine ([1, T, 3, H, W]) correctly instead of encoding
    # "3xH" and colliding across widths.
    height, width = optInputShape[-2], optInputShape[-1]
    batch = optInputShape[0]

    modelPath = Path(modelPath)
    if modelPath.suffix not in [".onnx", ".pth"]:
        raise ValueError(
            f"Unsupported model file extension: {modelPath.suffix}. Only .onnx and .pth are supported."
        )

    nameParts = [f"_{enginePrecision}_{height}x{width}"]

    # Batch-aware suffix so a batch>1 engine never collides with the batch-1
    # cache (which would silently load a static batch-1 engine). batch==1 keeps
    # the historical name so existing engines still load.
    if isinstance(batch, int) and batch > 1:
        nameParts.append(f"_b{batch}")

    if isRife and ensemble:
        nameParts.append("_ensemble")

    engineName = "".join(nameParts) + ".engine"
    return str(modelPath.with_suffix("")) + engineName
