import os
import time
import json
import re
import subprocess

TIMESLEEP = 2
CLIPURL = "https://www.youtube.com/watch?v=kpeUMAVJCig"
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
    "span-ncnn",
    "shufflecugan-ncnn",
    "realesrgan-ncnn",
    "cugan-ncnn",
    "compact-tesorrt",
    "ultracompact-tensorrt",
    "superultracompact-tensorrt",
    "shufflecugan-tensorrt",
    "span-tensorrt",
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
    "rife4.15-ncnn",
    "rife4.14-ncnn",
    "rife4.6-ncnn",
    "rife4.6-tensorrt",
    "rife4.14-tensorrt",
    "rife4.15-tensorrt",
    "gmfss",
]

denoiseMethods = ["scunet", "nafnet", "dpir", "span"]


def runAllBenchmarks(executor):
    print("Running all benchmarks. Depending on your system, this may take a while. Please be patient and keep the terminal at all times in the focus.")
    print("The results will be saved in benchmarkResults.json. Feel free to share this file with the Discord Community at https://discord.gg/2jqfkx3J")
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
    print("Please select 1080p as the desired quality.")
    outputPath = "output/test.mp4"
    # Utilize subprocess Popen with the stdout directed to the terminal
    subprocess.Popen(f"{executor} --input {CLIPURL} --output {outputPath}", shell=True).wait()
    #os.popen(f"{executor} --input {CLIPURL} --output {outputPath}").read()
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
        time.sleep(TIMESLEEP)

    return results


def runUpscaleBenchmark(inputVideo, executor):
    results = {}
    for method in upscaleMethods:
        print(f"Running {method} benchmark...")
        output = os.popen(
            f"{executor} --input {inputVideo} --upscale 1 --upscale_method {method} --benchmark 1 --outpoint 5"
        ).read()

        fps = parseFPS(output)
        results[method] = fps
        time.sleep(TIMESLEEP)

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
                f"{executor} --input {inputVideo} --interpolate 1 --interpolate_method {method} --benchmark 1 --outpoint 3"  # GMFSS is so slow that even this is too much
            ).read()

        fps = parseFPS(output)
        results[method] = fps
        time.sleep(TIMESLEEP)

        print(f"Running {method} with ensemble benchmark...")

        if method != "gmfss":  # Ensemble is irrelevant for GMFSS
            output = os.popen(
                f"{executor} --input {inputVideo} --interpolate 1 --interpolate_method {method} --benchmark 1 --ensemble 1 --outpoint 15"
            ).read()
        else:
            output = os.popen(
                f"{executor} --input {inputVideo} --interpolate 1 --interpolate_method {method} --benchmark 1 --ensemble 1 --outpoint 3"  # GMFSS is so slow that even this is too much
            ).read()

        fps = parseFPS(output)
        results[f"{method}-ensemble"] = fps
        time.sleep(TIMESLEEP)

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
        time.sleep(TIMESLEEP)

    return results


def parseFPS(output):
    match = re.findall(r"fps=\s*([\d.]+)", output)
    if match:
        print("FPS:", float(match[-1]))
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
