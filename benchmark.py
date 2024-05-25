import os
import time
import json
import re
import subprocess
import platform

TIMESLEEP = 2
CLIPURL = "https://www.youtube.com/watch?v=kpeUMAVJCig"
TESTINGVERSION = "V3"

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
    "cugan-directml",
    "compact-directml",
    "ultracompact-directml",
    "superultracompact-directml",
    "span-directml",
    "span-ncnn",
    "shufflecugan-ncnn",
    "realesrgan-ncnn",
    "cugan-ncnn",
    "compact-tensorrt",
    "ultracompact-tensorrt",
    "superultracompact-tensorrt",
    "shufflecugan-tensorrt",
    "span-tensorrt",
]

interpolateMethods = [
    "rife4.6",
    "rife4.15",
    "rife4.15-lite",
    "rife4.16-lite",
    "rife4.6-ncnn",
    "rife4.15-ncnn",
    "rife4.15-lite-ncnn",
    "rife4.16-lite-ncnn",
    "rife4.6-tensorrt",
    "rife4.15-lite-tensorrt",
    "rife4.15-tensorrt",
    "gmfss",
]

denoiseMethods = ["scunet", "nafnet", "dpir", "span"]

if platform.system() == 'Linux':
    upscaleMethods = [method for method in upscaleMethods if 'directml' not in method]

totalTests = len(upscaleMethods) + len(interpolateMethods) * 2 + len(denoiseMethods)
currentTest = 0


def runAllBenchmarks(executor, version):
    print("Running all benchmarks. Depending on your system, this may take a while. Please be patient and keep the terminal at all times in the focus.")
    print("The results will be saved in benchmarkResults.json. Feel free to share this file with the Discord Community at https://discord.gg/2jqfkx3J")
    inputVideo = getClip(executor)

    results = {
        "Upscale": runUpscaleBenchmark(inputVideo, executor),
        "Interpolate": runInterpolateBenchmark(inputVideo, executor),
        "Denoise": runDenoiseBenchmark(inputVideo, executor),
    }

    systemInfo = parseSystemInfo()

    with open("benchmarkResults.json", "w") as f:
        json.dump(
            {
                "Version": version,
                "Testing Methodology": TESTINGVERSION,
                "System Info": systemInfo,
                "Results": results,
            },
            f,
            indent=4,
        )


def getExe():
    if os.path.exists("main.exe"):
        version = subprocess.check_output(["main.exe", "--version"]).decode().strip()
        return "main.exe", version
    else:
        version = subprocess.check_output(["python3.11", "main.py", "--version"]).decode().strip()
        return "python3.11 main.py", version


def getClip(executor):
    print("Please select 1080p as the desired quality.")
    outputPath = "output/test.mp4"
    # Utilize subprocess Popen with the stdout directed to the terminal
    subprocess.Popen(f"{executor} --input {CLIPURL} --output {outputPath}", shell=True).wait()
    #os.popen(f"{executor} --input {CLIPURL} --output {outputPath}").read()
    return os.path.abspath(outputPath)

def runUpscaleBenchmark(inputVideo, executor):
    global currentTest
    results = {}
    for method in upscaleMethods:
        print(f"[{currentTest}/{totalTests}] {method} benchmark...")
        if method == "omnisr":
            output = os.popen(
                f"{executor} --input {inputVideo} --upscale  --upscale_method {method} --benchmark  --outpoint 2"
            ).read()
        else:
            output = os.popen(
                f"{executor} --input {inputVideo} --upscale  --upscale_method {method} --benchmark  --outpoint 4"
            ).read()

        fps = parseFPS(output)
        results[method] = fps
        time.sleep(TIMESLEEP)
        currentTest += 1

    return results

def runInterpolateBenchmark(inputVideo, executor):
    global currentTest
    results = {}
    for method in interpolateMethods:
        print(f"[{currentTest}/{totalTests}] {method} benchmark...")
        currentTest += 1

        if method != "gmfss":
            output = os.popen(
                f"{executor} --input {inputVideo} --interpolate  --interpolate_method {method} --benchmark "
            ).read()
        else:
            output = os.popen(
                f"{executor} --input {inputVideo} --interpolate  --interpolate_method {method} --benchmark  --outpoint 3"  # GMFSS is so slow that even this is too much
            ).read()

        fps = parseFPS(output)
        results[method] = fps
        time.sleep(TIMESLEEP)

        print(f"[{currentTest}/{totalTests}] Running {method} with ensemble benchmark...")
        currentTest += 1

        if method != "gmfss":  # Ensemble is irrelevant for GMFSS
            output = os.popen(
                f"{executor} --input {inputVideo} --interpolate  --interpolate_method {method} --benchmark  --ensemble  --outpoint 15"
            ).read()
        else:
            output = os.popen(
                f"{executor} --input {inputVideo} --interpolate  --interpolate_method {method} --benchmark  --ensemble  --outpoint 3"  # GMFSS is so slow that even this is too much
            ).read()

        fps = parseFPS(output)
        results[f"{method}-ensemble"] = fps
        time.sleep(TIMESLEEP)

    return results


def runDenoiseBenchmark(inputVideo, executor):
    global currentTest
    results = {}
    for method in denoiseMethods:
        print(f"[{currentTest}/{totalTests}] {method} benchmark...")
        currentTest += 1
        output = os.popen(
            f"{executor} --input {inputVideo} --denoise --denoise_method {method} --benchmark --outpoint 2"
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
    executor, version = getExe()
    print(f"Using {executor} version {version}")
    runAllBenchmarks(executor, version)
