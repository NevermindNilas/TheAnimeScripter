import os
import time
import json
import re

clipURL = "https://www.youtube.com/watch?v=79XMhXPBqsE"
upscaleMethods = (
    [
        "shufflecugan",
        "cugan",
        "compact",
        "ultracompact",
        "superultracompact",
        "span",
        "span-ncnn",
        "cugan-ncnn",
        "omnisr",
        "realesrgan",
        "realesrgan-ncnn",
        "shufflecugan-ncnn",
        "apisr",
    ],
)

interpolateMethods = (
    [
        "rife4.6",
        "rife4.14",
        "rife4.15",
        "rife4.16-lite",
        "rife4.6-ncnn",
        "rife4.14-ncnn",
        "rife4.15-ncnn",
        "gmfss",
    ],
)

denoiseMethods = ["scunet", "nafnet", "dpir", "span"]


def getClip():
    os.popen(f"main.exe --input {clipURL} --output test.mp4").read()
    return os.path.join("output", "test.mp4")


def runAllBenchmarks():
    if not os.path.exists("test.mp4"):
        print("No test.mp4 file found. Downloading test clip...")
        inputVideo = getClip()
    else:
        inputVideo = "test.mp4"

    results = {
        "Upscale": runUpscaleBenchmark(inputVideo),
        "Interpolate": runInterpolateBenchmark(inputVideo),
        "Denoise": runDenoiseBenchmark(inputVideo),
    }

    systemInfo = parseSystemInfo()

    with open("benchmarkResults.json", "w") as f:
        json.dump({"Testing Methodology": "V2", "System Info": systemInfo, "Results": results}, f, indent=4)


def runUpscaleBenchmark(inputVideo):
    results = {}
    for method in upscaleMethods[0]:
        print(f"Running {method} benchmark...")
        output = os.popen(
            f"main.exe --input {input} --upscale 1 --upscale_method {method} --benchmark 1 --outpoint 4"
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
            f"main.exe --input {input} --interpolate 1 --interpolate_method {method} --benchmark 1"
        ).read()

        fps = parseFPS(output)
        results[method] = fps
        time.sleep(1)

        print(f"Running {method} with ensemble benchmark...")
        output = os.popen(
            f"main.exe --input test.mp4 --interpolate 1 --interpolate_method {method} --benchmark 1 --ensemble 1"
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
            f"main.exe --input test.mp4 --denoise 1 --denoise_method {method} --benchmark 1 --outpoint 2"
        ).read()

        fps = parseFPS(output)
        results[method] = fps
        time.sleep(1)

    return results


def parseFPS(output):
    match = re.findall(r"fps=\s*([\d.]+)", output)
    match = match[-1]
    if match:
        return float(match)
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
