import os
import time
import json
import re

clipURL = "https://www.youtube.com/watch?v=9vVjAOi1LCg"
dedupMethods = ["ffmpeg", "ssim", "mse"]

upscaleMethods = (
    [
        "shufflecugan",
        "shufflecugan-directml",
        "cugan",
        "cugan-directml",
        "compact",
        "compact-directml",
        "ultracompact",
        "ultracompact-directml",
        "superultracompact",
        "superultracompact-directml",
        "span",
        "span-directml",
        "omnisr",
        "realesrgan",
        "apisr",
    ],
)

interpolateMethods = (
    [
        "rife4.6",
        "rife4.14",
        "rife4.15",
        "rife4.15-lite",
        "rife4.16-lite",
        "gmfss",
    ],
)

denoiseMethods = ["scunet", "nafnet", "dpir", "span"]


def runAllBenchmarks():
    inputVideo = getClip()

    results = {
        "Dedup": runDedupBenchmark(inputVideo),
        "Upscale": runUpscaleBenchmark(inputVideo),
        "Interpolate": runInterpolateBenchmark(inputVideo),
        "Denoise": runDenoiseBenchmark(inputVideo),
    }

    systemInfo = parseSystemInfo()

    with open("benchmarkResults.json", "w") as f:
        json.dump(
            {
                "Testing Methodology": "V2",
                "System Info": systemInfo,
                "Results": results,
            },
            f,
            indent=4,
        )


def getExe():
    if os.path.exists("main.exe"):
        return "main.exe"
    else:
        return "python main.py"


def getClip():
    os.popen(f"{getExe()} --input {clipURL} --output test.mp4").read()
    return os.path.join("output", "test.mp4")


def runDedupBenchmark(inputVideo):
    results = {}
    for method in dedupMethods:
        print(f"Running {method} benchmark...")
        output = os.popen(
            f"{getExe()} --input {inputVideo} --dedup 1 --dedup_method {method} --benchmark 1"
        ).read()

        fps = parseFPS(output)
        results[method] = fps
        time.sleep(1)
    
    return results


def runUpscaleBenchmark(inputVideo):
    results = {}
    for method in upscaleMethods[0]:
        print(f"Running {method} benchmark...")
        output = os.popen(
            f"{getExe()} --input {inputVideo} --upscale 1 --upscale_method {method} --benchmark 1 --outpoint 4"
        ).read()

        fps = parseFPS(output)
        results[method] = fps
        time.sleep(1)

    return results


def runInterpolateBenchmark(inputVideo):
    results = {}
    for method in interpolateMethods[0]:
        print(f"Running {method} benchmark...")
        output = os.popen(
            f"{getExe()} --input {inputVideo} --interpolate 1 --interpolate_method {method} --benchmark 1"
        ).read()

        fps = parseFPS(output)
        results[method] = fps
        time.sleep(1)

        print(f"Running {method} with ensemble benchmark...")
        output = os.popen(
            f"{getExe()} --input test.mp4 --interpolate 1 --interpolate_method {method} --benchmark 1 --ensemble 1"
        ).read()

        fps = parseFPS(output)
        results[f"{method}-ensemble"] = fps
        time.sleep(1)

    return results


def runDenoiseBenchmark(inputVideo):
    results = {}
    for method in denoiseMethods:
        print(f"Running {method} benchmark...")
        output = os.popen(
            f"{getExe()} --input {inputVideo} --denoise 1 --denoise_method {method} --benchmark 1 --outpoint 2"
        ).read()

        fps = parseFPS(output)
        results[method] = fps
        time.sleep(1)

    return results


def parseFPS(output):
    match = re.findall(r"fps=\s*([\d.]+)", output)
    if match:
        return float(match[-1])
    else:
        return None


def parseSystemInfo():
    systemInfo = {}
    with open("log.txt", "r") as file:
        lines = file.readlines()
        start = lines.index("============== System Checker ==============\n") + 1
        end = lines.index("============== Arguments Checker ==============\n")
        for line in lines[start:end]:
            if ": " in line:
                key, value = line.strip().split(": ")
                systemInfo[key] = value
    return systemInfo


if __name__ == "__main__":
    runAllBenchmarks()
