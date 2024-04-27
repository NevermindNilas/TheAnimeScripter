import os
import time
import json
import re

clipURL = "https://www.youtube.com/watch?v=kpeUMAVJCig"
dedupMethods = ["ffmpeg", "ssim", "mse"]

upscaleMethods = [
    "shufflecugan",
    "cugan",
    "compact",
    "ultracompact",
    "superultracompact",
    "span",
    "omnisr",
    "realesrgan",
    "apisr",
    "shufflecugan-directml",
    "cugan-directml",
    "compact-directml",
    "ultracompact-directml",
    "superultracompact-directml",
    "span-directml",
]

interpolateMethods = [
    "rife4.6",
    "rife4.14",
    "rife4.15",
    "rife4.15-lite",
    "rife4.16-lite",
    "rife4.6-directml",
    "rife4.14-directml",
    "rife4.15-directml",
    "rife4.15-lite-directml",
    "gmfss",
]

denoiseMethods = ["scunet", "nafnet", "dpir", "span"]


def runAllBenchmarks(executor):
    inputVideo = getClip(executor)

    results = {
        "Dedup": runDedupBenchmark(inputVideo, executor),
        "Upscale": runUpscaleBenchmark(inputVideo, executor),
        "Interpolate": runInterpolateBenchmark(inputVideo, executor),
        "Denoise": runDenoiseBenchmark(inputVideo, executor),
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


def getClip(executor):
    outputPath = "output/test.mp4"
    os.popen(f"{executor} --input {clipURL} --output {outputPath}").read()
    return os.path.abspath(outputPath)

def runDedupBenchmark(inputVideo, executor):
    results = {}
    for method in dedupMethods:
        print(f"Running {method} benchmark...")
        output = os.popen(
            f"{executor} --input {inputVideo} --dedup 1 --dedup_method {method} --benchmark 1"
        ).read()

        fps = parseFPS(output)
        results[method] = fps
        time.sleep(1)
    
    return results


def runUpscaleBenchmark(inputVideo, executor):
    results = {}
    for method in upscaleMethods:
        print(f"Running {method} benchmark...")
        output = os.popen(
            f"{executor} --input {inputVideo} --upscale 1 --upscale_method {method} --benchmark 1 --outpoint 3"
        ).read()

        fps = parseFPS(output)
        results[method] = fps
        time.sleep(1)

    return results


def runInterpolateBenchmark(inputVideo, executor):
    results = {}
    for method in interpolateMethods:
        print(f"Running {method} benchmark...")

        if method != "gmfss":
            output = os.popen(
                f"{executor} --input {inputVideo} --interpolate 1 --interpolate_method {method} --benchmark 1"
            ).read()
        else:
            output = os.popen(
                f"{executor} --input {inputVideo} --interpolate 1 --interpolate_method {method} --benchmark 1 --outpoint 5" # GMFSS is so slow that even this is too much
            ).read()

        fps = parseFPS(output)
        results[method] = fps
        time.sleep(1)

        print(f"Running {method} with ensemble benchmark...")

        if method != "gmfss": # Ensemble is irrelevant for GMFSS
            output = os.popen(
                f"{executor} --input test.mp4 --interpolate 1 --interpolate_method {method} --benchmark 1 --ensemble 1"
            ).read()
        else:
            output = os.popen(
                f"{executor} --input test.mp4 --interpolate 1 --interpolate_method {method} --benchmark 1 --ensemble 1 --outpoint 5" # GMFSS is so slow that even this is too much
            ).read()

        fps = parseFPS(output)
        results[f"{method}-ensemble"] = fps
        time.sleep(1)

    return results


def runDenoiseBenchmark(inputVideo, executor):
    results = {}
    for method in denoiseMethods:
        print(f"Running {method} benchmark...")
        output = os.popen(
            f"{executor} --input {inputVideo} --denoise 1 --denoise_method {method} --benchmark 1 --outpoint 2"
        ).read()

        fps = parseFPS(output)
        results[method] = fps
        time.sleep(1)

    return results


def parseFPS(output):
    match = re.findall(r"fps=\s*([\d.]+)", output)
    if match:
        print(float(match[-1]))
        return float(match[-1])
    else:
        print("None")
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
    runAllBenchmarks(executor=getExe())
