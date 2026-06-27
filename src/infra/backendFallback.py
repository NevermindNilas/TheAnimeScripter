import logging

BACKEND_SUFFIXES = ("-directml", "-ncnn", "-tensorrt", "-mps", "-openvino")

METHOD_ATTRIBUTES = {
    "interpolate_method": "interpolate",
    "upscale_method": "upscale",
    "segment_method": "segment",
    "depth_method": "depth",
    "restore_method": "restore",
    "dedup_method": "dedup",
    "obj_detect_method": "obj_detect",
}


def hasExplicitBackend(method):
    return any(backend in method.lower() for backend in BACKEND_SUFFIXES)


def fallbackMethod(method, availableModels, preferMps=False):
    base = method.lower()
    if preferMps:
        mpsMethod = f"{base}-mps"
        if mpsMethod in availableModels:
            return mpsMethod

    directMlMethod = f"{base}-directml"
    if directMlMethod in availableModels:
        return directMlMethod

    ncnnMethod = f"{base}-ncnn"
    if ncnnMethod in availableModels:
        return ncnnMethod

    return method


def applyBackendFallbacks(args, availableModels, preferMps=False):
    for attr, flagName in METHOD_ATTRIBUTES.items():
        if not getattr(args, flagName, False):
            continue

        currentMethod = getattr(args, attr)

        if attr == "restore_method" and isinstance(currentMethod, list):
            adjusted = []
            for method in currentMethod:
                if hasExplicitBackend(method):
                    logging.info(f"{attr} method {method} already uses a backend")
                    adjusted.append(method)
                    continue

                newMethod = fallbackMethod(method, availableModels, preferMps=preferMps)
                if newMethod != method:
                    logging.info(f"Adjusted {attr} method from {method} to {newMethod}")
                adjusted.append(newMethod)
            setattr(args, attr, adjusted)
            continue

        if hasExplicitBackend(currentMethod):
            logging.info(f"{attr} already uses a backend: {currentMethod}")
            continue

        newMethod = fallbackMethod(currentMethod, availableModels, preferMps=preferMps)
        if newMethod != currentMethod:
            logging.info(f"Adjusted {attr} from {currentMethod} to {newMethod}")
            setattr(args, attr, newMethod)
        else:
            logging.info(
                f"No adjustment for {attr} ({currentMethod} remains unchanged)"
            )

    if getattr(args, "moblur", False):
        moblurMethod = args.moblur_method
        if not any(
            backend in moblurMethod for backend in ("-directml", "-openvino", "-mps")
        ):
            base = moblurMethod.replace("-tensorrt", "")
            if preferMps and f"{base}-mps" in availableModels:
                args.moblur_method = f"{base}-mps"
            else:
                args.moblur_method = f"{base}-directml"
            logging.info(
                f"Adjusted moblur_method from {moblurMethod} to {args.moblur_method} "
                "because CUDA is unavailable"
            )
