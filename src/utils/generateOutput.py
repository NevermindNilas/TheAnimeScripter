import os
import random
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)


def outputNameGenerator(args, videoInput):
    argMap = {
        "resize": f"-Re{getattr(args, 'resize_factor', '')}",
        "dedup": f"-De{getattr(args, 'dedup_sens', '')}",
        "interpolate": f"-Int{getattr(args, 'interpolate_factor', '')}",
        "upscale": f"-Up{getattr(args, 'upscale_factor', '')}",
        "sharpen": f"-Sh{getattr(args, 'sharpen_sens', '')}",
        "denoise": f"-De{getattr(args, 'denoise_method', '')}",
        "segment": "-Segment" if getattr(args, "segment", False) else "",
        "depth": "-Depth" if getattr(args, "depth", False) else "",
        "ytdlp": "-YTDLP" if getattr(args, "ytdlp", False) else "",
    }

    try:
        # Check if videoInput is a URL
        if "https://" in videoInput or "http://" in videoInput:
            name = "TAS" + "-YTDLP" + f"-{random.randint(0, 1000)}" + ".mp4"
            logging.debug(f"Generated name for URL input: {name}")
            return name
        else:
            parts = [
                os.path.splitext(os.path.basename(videoInput))[0]
                if videoInput
                else "TAS"
            ]

        for arg, formatStr in argMap.items():
            if formatStr:
                parts.append(formatStr)

        parts.append(f"-{random.randint(0, 1000)}")

        if getattr(args, "segment", False) or getattr(args, "encode_method", "") in [
            "prores"
        ]:
            extension = ".mov"
        elif videoInput:
            extension = os.path.splitext(videoInput)[1]
        else:
            extension = ".mp4"

        outputName = "".join(parts) + extension
        logging.debug(f"Generated output name: {outputName}")

        return outputName

    except AttributeError as e:
        logging.error(f"AttributeError: {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise
