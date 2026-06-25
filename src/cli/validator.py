import logging
import os
import sys

import src.constants as cs
from src.cli.startup import _handleDependencies, _promptDownloadRequirementsSelection
from src.infra.logAndPrint import logAndPrint


def isAnyOtherProcessingMethodEnabled(args):
    return any(
        [
            args.interpolate,
            args.upscale,
            args.segment,
            args.restore,
            args.stabilize,
            args.resize,
            args.dedup,
            args.depth,
            args.autoclip,
            args.obj_detect,
            args.moblur,
        ]
    )


def _autoEnableParentFlags(args):
    logging.info(f"[DEBUG] hasattr(args, '_json_keys'): {hasattr(args, '_json_keys')}")
    if hasattr(args, "_json_keys"):
        logging.info(f"[DEBUG] args._json_keys: {args._json_keys}")

    methodToFlagMapping = {
        "interpolate_method": ("interpolate", "rife4.6"),
        "interpolate_factor": ("interpolate", 2.0),
        "upscale_method": ("upscale", "shufflecugan"),
        "upscale_factor": ("upscale", 2),
        "dedup_method": ("dedup", "ssim"),
        "dedup_sens": ("dedup", 35.0),
        "restore_method": ("restore", ["anime1080fixer"]),
        "segment_method": ("segment", "anime"),
        "depth_method": ("depth", "small_v2"),
        "obj_detect_method": ("obj_detect", "yolov9_small-directml"),
        "resize_factor": ("resize", 2),
        "output_scale": ("resize", ""),
        "moblur_method": ("moblur", "rife4.25"),
        "moblur_factor": ("moblur", 8),
        "moblur_strength": ("moblur", "gaussian_sym"),
        "moblur_shutter_angle": ("moblur", 180.0),
    }

    cliProvided = set()
    for a in sys.argv[1:]:
        if a.startswith("--"):
            name = a[2:].split("=")[0]
            cliProvided.add(name)

    for methodArg, (parentFlag, defaultValue) in methodToFlagMapping.items():
        if hasattr(args, methodArg):
            currentValue = getattr(args, methodArg)

            providedOnCLI = (
                methodArg in cliProvided or methodArg.replace("_", "-") in cliProvided
            )

            isExplicitlyProvided = providedOnCLI or (
                hasattr(args, "_json_keys") and methodArg in args._json_keys
            )

            if methodArg == "interpolate_method":
                logging.info(
                    f"[DEBUG] interpolate_method - providedOnCLI: {providedOnCLI}, isExplicitlyProvided: {isExplicitlyProvided}"
                )

            if isExplicitlyProvided:
                if not getattr(args, parentFlag):
                    setattr(args, parentFlag, True)
                    logging.info(
                        f"Auto-enabling --{parentFlag} because --{methodArg} was provided"
                    )
            else:
                if currentValue != defaultValue:
                    if not getattr(args, parentFlag):
                        setattr(args, parentFlag, True)
                        logging.info(
                            f"Auto-enabling --{parentFlag} because {methodArg} differs from default"
                        )


def _handleDepthSettings(args):
    if args.depth:
        logging.info("Depth enabled, audio processing will be disabled")
        cs.AUDIO = False

    if args.depth_quality not in ["low"] and args.depth_method.split("-")[-1] in [
        "tensorrt",
        "directml",
    ]:
        logAndPrint(
            f"{args.depth_quality.upper()} depth estimation quality is incomaptible with tensorrt and directml, defaulting to low quality",
            "yellow",
        )
        args.depth_quality = "low"

    if args.depth_norm and "video" in args.depth_method:
        logAndPrint(
            "Depth normalization is not compatible with video depth methods, disabling depth_norm",
            "yellow",
        )
        args.depth_norm = False

    if args.depth_method in ["giant_v2", "og_giant_v2"]:
        logAndPrint(
            f"{args.depth_method} is a very large model and may cause out of memory errors on GPUs with less than 16GB of VRAM",
            "yellow",
        )
        if args.half:
            logAndPrint(
                "Half precision does not work with giant models, switching to full precision",
            )
            args.half = False


def _configureProcessingSettings(args):
    if args.slowmo:
        cs.AUDIO = False
        logging.info("Slow motion enabled, audio processing disabled")

    if args.static_step and isinstance(args.interpolate_factor, float):
        logging.info("Interpolate Factor is a float, static step will be disabled")
        args.static_step = False

    if args.dedup:
        if not args.smooth_dedup:
            cs.AUDIO = False
            logging.info(
                "Deduplication enabled and smooth dedup disabled, audio processing disabled"
            )

        if args.dedup_method in ["ssim", "ssim-cuda"]:
            args.dedup_sens = 1.0 - (args.dedup_sens / 1000)
        elif args.dedup_method in ["flownets"]:
            args.dedup_sens = args.dedup_sens / 100

        logging.info(
            f"New dedup sensitivity for {args.dedup_method} is: {args.dedup_sens}"
        )

    if args.autoclip:
        if args.autoclip_method == "pyscenedetect":
            args.autoclip_sens = float(100 - args.autoclip_sens)
        else:
            args.autoclip_sens = float(1.0 - (args.autoclip_sens / 100.0))
        logging.info(
            f"New autoclip sensitivity for {args.autoclip_method} is: {args.autoclip_sens}"
        )

    if args.compile_mode != "default":
        logging.info(
            f"Pytorch Compile mode is set to {args.compile_mode}, this will increase startup time and memory usage and may lead to instability with some models"
        )


def _validateCustomUpscaleModel(args):
    if not args.custom_model:
        return

    args.custom_model = os.path.abspath(args.custom_model)
    if not os.path.isfile(args.custom_model):
        logAndPrint(f"Custom model file not found: {args.custom_model}", "red")
        sys.exit()

    extension = os.path.splitext(args.custom_model)[1].lower()
    pytorchExtensions = {".pt", ".pth", ".ckpt", ".safetensors"}
    onnxExtensions = {".onnx"}
    backendSuffixes = ("-directml", "-openvino", "-tensorrt", "-ncnn")

    selectedBackend = "pytorch"
    baseMethod = args.upscale_method
    for suffix in backendSuffixes:
        if args.upscale_method.endswith(suffix):
            selectedBackend = suffix[1:]
            baseMethod = args.upscale_method[: -len(suffix)]
            break

    if extension in onnxExtensions:
        if selectedBackend not in {"directml", "openvino", "tensorrt"}:
            logAndPrint(
                "Custom ONNX upscale models require an ONNX backend. Use an upscale method ending in -directml, -openvino, or -tensorrt, for example "
                f"{baseMethod}-directml.",
                "red",
            )
            sys.exit()
        return

    if extension in pytorchExtensions:
        if selectedBackend != "pytorch":
            logAndPrint(
                "Custom PyTorch upscale models require a CUDA/PyTorch upscale method without a backend suffix. "
                f"Use {baseMethod} for .pt/.pth/.ckpt/.safetensors files.",
                "red",
            )
            sys.exit()
        return

    logAndPrint(
        "Unsupported custom upscale model format. Supported extensions are .pt, .pth, .ckpt, .safetensors, and .onnx.",
        "red",
    )
    sys.exit()


def _adjustMethodsBasedOnCuda(args):
    isDarwin = cs.SYSTEM == "Darwin"

    def adjustMethod(method, modelsList):
        base = method.lower()
        if isDarwin:
            mpsMethod = f"{base}-mps"
            if mpsMethod in modelsList:
                return mpsMethod
        directML = f"{base}-directml"
        if directML in modelsList:
            return directML
        newMethod = f"{base}-ncnn"
        if newMethod in modelsList:
            return newMethod
        return method

    supportsCuda = getattr(args, "supportsCuda", None)

    if supportsCuda is None:
        from src.infra.isCudaInit import CudaChecker, detectGPUArchitecture

        isCuda = CudaChecker()

        needsFallback = False
        if isCuda.cudaAvailable:
            isModernGPU, gpuName, computeCap = detectGPUArchitecture()
            if not isModernGPU:
                logAndPrint(
                    f"Detected {gpuName} (compute capability: {computeCap}). "
                    f"This GPU may not support modern CUDA kernels. "
                    f"Automatically switching to DirectML/NCNN backends for compatibility.",
                    "yellow",
                )
                needsFallback = True
        else:
            needsFallback = True
    else:
        needsFallback = not supportsCuda

    if needsFallback:
        from src.model.downloadModels import modelsList

        availableModels = modelsList()
        methodAttributes = [
            "interpolate_method",
            "upscale_method",
            "segment_method",
            "depth_method",
            "restore_method",
            "dedup_method",
            "obj_detect_method",
        ]

        methodToFlag = {
            "interpolate_method": "interpolate",
            "upscale_method": "upscale",
            "segment_method": "segment",
            "depth_method": "depth",
            "restore_method": "restore",
            "dedup_method": "dedup",
            "obj_detect_method": "obj_detect",
        }

        for attr in methodAttributes:
            flagName = methodToFlag.get(attr)
            if flagName and not getattr(args, flagName):
                continue

            currentMethod = getattr(args, attr)

            if attr == "restore_method" and isinstance(currentMethod, list):
                adjusted = []
                for method in currentMethod:
                    if any(
                        backend in method.lower()
                        for backend in [
                            "-directml",
                            "-ncnn",
                            "-tensorrt",
                            "-mps",
                            "-openvino",
                        ]
                    ):
                        logging.info(
                            f"{attr} method {method} already using non-default backend"
                        )
                        adjusted.append(method)
                        continue

                    newMethod = adjustMethod(method, availableModels)
                    if newMethod != method:
                        logging.info(
                            f"Adjusted {attr} method from {method} to {newMethod}"
                        )
                    adjusted.append(newMethod)
                setattr(args, attr, adjusted)
            else:
                if any(
                    backend in currentMethod.lower()
                    for backend in [
                        "-directml",
                        "-ncnn",
                        "-tensorrt",
                        "-mps",
                        "-openvino",
                    ]
                ):
                    logging.info(
                        f"{attr} already using non-default backend: {currentMethod}"
                    )
                    continue

                newMethod = adjustMethod(currentMethod, availableModels)
                if newMethod != currentMethod:
                    logging.info(f"Adjusted {attr} from {currentMethod} to {newMethod}")
                    setattr(args, attr, newMethod)
                else:
                    logging.info(
                        f"No adjustment for {attr} ({currentMethod} remains unchanged)"
                    )

        if getattr(args, "moblur", False):
            mob = args.moblur_method
            if not any(
                backend in mob
                for backend in ("-directml", "-openvino", "-mps")
            ):
                base = mob.replace("-tensorrt", "")
                if isDarwin and f"{base}-mps" in availableModels:
                    args.moblur_method = f"{base}-mps"
                else:
                    args.moblur_method = f"{base}-directml"
                logging.info(
                    f"Adjusted moblur_method from {mob} to {args.moblur_method} "
                    f"(no CUDA available)"
                )


def _downloadOfflineModels(args):
    from src.model.downloadModels import downloadModels, modelsList

    logAndPrint(
        f"Offline mode enabled, downloading {args.offline} model(s)...", "green"
    )

    options = modelsList() if args.offline == ["all"] else args.offline
    for option in options:
        for precision in [True, False]:
            try:
                downloadModels(option.lower(), half=precision)
            except Exception:
                logging.error(
                    f"Failed to download model: {option} with precision: {'fp16' if precision else 'fp32'}"
                )
    logAndPrint(
        f"Offline model(s) {', '.join(options)} downloaded successfully!", "green"
    )


def processURL(args, outputPath):
    from urllib.parse import urlparse

    from src.io.inputOutputHandler import generateOutputName
    from src.ytdlp import VideoDownloader

    result = urlparse(args.input)

    if result.netloc.lower() in ["www.youtube.com", "youtube.com", "youtu.be"]:
        logging.info("URL is valid and will be used for processing")

        if args.output is None:
            outputFolder = os.path.join(outputPath, "output")
            os.makedirs(os.path.join(outputFolder), exist_ok=True)
            args.output = os.path.join(
                outputFolder, generateOutputName(args, args.input)
            )
        elif os.path.isdir(args.output):
            outputFolder = args.output
            os.makedirs(os.path.join(outputFolder), exist_ok=True)
        else:
            outputFolder = os.path.dirname(args.output)
            os.makedirs(os.path.join(outputFolder), exist_ok=True)

        tempOutput = os.path.join(outputFolder, generateOutputName(args, args.input))

        VideoDownloader(args.input, tempOutput, args.encode_method, args.custom_encoder)
        logAndPrint(
            f"Video downloaded successfully to {tempOutput}",
            "green",
        )

        if not isAnyOtherProcessingMethodEnabled(args):
            if tempOutput != args.output:
                os.rename(tempOutput, args.output)
                logging.info(f"Renamed output to: {args.output}")
            sys.exit()

        args.input = str(tempOutput)
        logging.info(f"New input path: {args.input}")
    else:
        logging.error(
            "URL is invalid or not a YouTube URL, please check the URL and try again"
        )
        sys.exit()


def argumentsChecker(args, outputPath, parser):
    from src.version import __version__

    args.png_passthrough = False
    args.single_image_input = False

    if args.list_presets:
        from src.server.presetLogic import listPresets

        listPresets()
        sys.exit()

    if args.list_methods is not None:
        from src.cli.parser import _listMethods

        sys.exit(_listMethods(parser, args.list_methods))

    if args.preset:
        from src.server.presetLogic import createPreset

        args = createPreset(args)

    if args.json:
        args, loadedKeys = _loadJsonConfig(args, parser)
        args._json_keys = loadedKeys

    _autoEnableParentFlags(args)

    if args.download_requirements is not None:
        from src.infra.dependencyHandler import DependencyChecker

        _handleDependencies(args)
        selectedProfile = args.download_requirements.strip().lower()
        if not selectedProfile:
            selectedProfile = _promptDownloadRequirementsSelection()

        checker = DependencyChecker()
        if not checker.installProfile(selectedProfile):
            sys.exit(1)

        logAndPrint(
            "All required libraries have been downloaded, you can now run the script freely.",
            "green",
        )
        sys.exit()

    if args.cleanup:
        from src.infra.dependencyHandler import (
            DependencyChecker,
            getDependencyProfile,
            getRequirementsFileForProfile,
            uninstallDependencies,
        )

        checker = DependencyChecker()
        storedProfile = checker.loadStoredProfile()

        if storedProfile:
            try:
                extension = getRequirementsFileForProfile(storedProfile)
            except ValueError:
                storedProfile = None

        if not storedProfile:
            from src.infra.isCudaInit import detectGPUArchitecture, detectNVidiaGPU

            isNvidia = detectNVidiaGPU()
            supportsCuda = False
            if isNvidia:
                supportsCuda, _, _ = detectGPUArchitecture()
            extension = getRequirementsFileForProfile(
                getDependencyProfile(cs.SYSTEM, supportsCuda)
            )

        success, message = uninstallDependencies(extension=extension)
        checker.clearCache()

        logging.info(message)

        if success:
            logAndPrint(
                "Dependencies from the selected runtime profile were uninstalled.",
                "green",
            )
            sys.exit()
        else:
            logAndPrint(
                "Failed to uninstall dependencies, please check the logs for more details",
                "red",
            )
            logging.error("Failed to uninstall dependencies")
            print(message)

    logging.info("============== Version ==============")
    logging.info(f"TAS: {__version__}\n")

    cliArgs = sys.argv[1:]
    providedArgs = set()
    for _i, arg in enumerate(cliArgs):
        if arg.startswith("--"):
            argName = arg[2:]
            providedArgs.add(argName)

    logging.info("============== Arguments ==============")
    for arg, value in vars(args).items():
        if arg in providedArgs and value not in [None, "", "none"]:
            logging.info(f"{arg.upper()}: {value}")

    if not args.benchmark:
        from src.infra.checkSpecs import checkSystem

        checkSystem()

    if args.preview and args.benchmark:
        logAndPrint(
            "Preview cannot be enabled in benchmark mode, disabling preview",
            "yellow",
        )
        args.preview = False

    if args.ae:
        logging.info("After Effects interface detected")
        cs.ADOBE = True
        from src.server.aeComms import startServerInThread

        try:
            startServerInThread(
                host=args.ae,
            )
        except Exception as e:
            logging.error(f"Failed to start AE comms server: {e}")
            logAndPrint(
                "Failed to start AE comms server, please check the logs for more details",
                "red",
            )

    logging.info("\n============== Arguments Checker ==============")
    _handleDependencies(args)

    if args.slowmo and not args.interpolate:
        logAndPrint(
            "Slow motion is enabled but interpolation is not, disabling slowmo",
            "yellow",
        )
        args.slowmo = False

    _handleDepthSettings(args)

    if args.offline != "none":
        _downloadOfflineModels(args)

    _configureProcessingSettings(args)

    _adjustMethodsBasedOnCuda(args)
    _validateCustomUpscaleModel(args)

    if args.custom_encoder:
        logging.info("Custom encoder specified, use with caution")

    if args.bit_depth == "16bit" and args.segment:
        logging.error(
            "16bit input is not supported with segmentation, defaulting to 8bit"
        )
        args.bit_depth = "8bit"

    if args.output:
        outDir = os.path.dirname(os.path.abspath(args.output))
        if outDir and not os.path.exists(outDir):
            os.makedirs(outDir, exist_ok=True)

    if args.encode_method in ["gif", "png"]:
        logging.info(
            f"Encoding method is set to {args.encode_method}, disabling audio processing"
        )
        cs.AUDIO = False

    if not args.input:
        logging.error("No input specified")
        sys.exit()
    elif args.input.startswith(("http", "www")):
        processURL(args, outputPath)
    elif args.input.lower().endswith((".png", ".jpg", ".jpeg")):
        if "%" in args.input:
            logging.info(f"Image sequence pattern detected: {args.input}")
            args.input = os.path.abspath(args.input)
            cs.AUDIO = False
        elif args.input.lower().endswith(".png"):
            args.input = os.path.abspath(args.input)
            cs.AUDIO = False
            args.single_image_input = True
            args.png_passthrough = True
            logging.info("Single PNG input detected, enabling PNG passthrough mode")

            if isAnyOtherProcessingMethodEnabled(args) and args.encode_method != "png":
                logging.info(
                    "Single PNG with processing detected; forcing --encode_method png for valid image output"
                )
                args.encode_method = "png"
        else:
            raise Exception(
                "Single image input is not supported for this format. For image sequences, use a pattern like 'frames_%05d.png' or provide a folder containing PNG files."
            )
    elif args.input.lower().endswith(".gif"):
        if args.encode_method != "gif":
            logging.error(
                "GIF input detected but encoding method is not set to GIF, defaulting to GIF encoding"
            )
            args.encode_method = "gif"
    else:
        try:
            args.input = os.path.abspath(args.input)
        except Exception:
            logging.error("Error processing input")
            sys.exit()

    if args.output_scale:
        try:
            width, height = args.output_scale.split("x")
            args.output_scale_width = int(width)
            args.output_scale_height = int(height)
            if args.output_scale_width <= 0 or args.output_scale_height <= 0:
                raise ValueError("Width and height must be positive integers")
            logging.info(
                f"Output scale set to {args.output_scale_width}x{args.output_scale_height}"
            )
        except (ValueError, AttributeError):
            logAndPrint(
                f"Invalid output_scale format: {args.output_scale}. Expected format: WIDTHxHEIGHT (e.g., 2560x1440)"
            )
            sys.exit()
    else:
        args.output_scale_width = None
        args.output_scale_height = None

    if args.upscale and hasattr(args, "upscale_factor"):
        try:
            if int(args.upscale_factor) < 2:
                logAndPrint(
                    "Upscale factor must be at least 2 when --upscale is enabled; defaulting to 2",
                    "yellow",
                )
                logging.info(
                    f"Adjusted upscale_factor from {args.upscale_factor} to 2 to satisfy minimum requirement"
                )
                args.upscale_factor = 2
        except Exception:
            logAndPrint(
                "Invalid upscale_factor provided; defaulting to 2",
                "yellow",
            )
            logging.info("Invalid upscale_factor value encountered; set to 2")
            args.upscale_factor = 2

    logging.info(
        f"[DEBUG] Before processing check - args.interpolate: {args.interpolate}"
    )

    if not isAnyOtherProcessingMethodEnabled(args) and not args.png_passthrough:
        logAndPrint(
            "No processing methods specified, make sure to use enabler arguments like --upscale, --interpolate, etc.",
            "red",
        )
        sys.exit()

    return args


def _loadJsonConfig(args, parser):
    import json

    cliProvidedArgs = set()
    for arg in sys.argv[1:]:
        if arg.startswith("--") and arg != "--json":
            argName = arg[2:].replace("-", "_")
            cliProvidedArgs.add(argName)

    if len(cliProvidedArgs) > 0:
        logAndPrint(
            "Cannot use --json with other command line arguments. Use --json alone.",
            "red",
        )
        sys.exit()

    jsonPath = os.path.abspath(args.json)

    if not os.path.exists(jsonPath):
        logAndPrint(f"JSON config file not found: {jsonPath}", "red")
        sys.exit()

    try:
        with open(jsonPath, encoding="utf-8") as f:
            jsonConfig = json.load(f)
    except json.JSONDecodeError as e:
        logAndPrint(f"Invalid JSON format in config file: {e}", "red")
        sys.exit()
    except Exception as e:
        logAndPrint(f"Error reading JSON config: {e}", "red")
        sys.exit()

    defaults = {}
    for action in parser._actions:
        if action.dest not in ["help", "version", "json"]:
            defaults[action.dest] = action.default

    loadedKeys = set()
    for key, value in jsonConfig.items():
        if key == "json":
            continue

        if hasattr(args, key):
            currentValue = getattr(args, key)
            defaultValue = defaults.get(key)

            if currentValue == defaultValue:
                setattr(args, key, value)
                logging.info(f"Loaded from JSON: {key} = {value}")
            loadedKeys.add(key)
        else:
            logging.warning(f"Unknown option in JSON config: {key}")

    return args, loadedKeys
