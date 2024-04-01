import os
import time
import json
import re

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

def runAllBenchmarks():
    results = {
        "Upscale": runUpscaleBenchmark(),
        "Interpolate": runInterpolateBenchmark(),
        "Denoise": runDenoiseBenchmark(),
    }

    #systemInfo = parseSystemInfo()
    
    with open("benchmarkResults.json", "w") as f:
        json.dump({"Results": results}, f, indent=4)

def runUpscaleBenchmark():
    results = {}
    for method in upscaleMethods[0]:
        print(f"Running {method} benchmark...")
        output = os.popen(
            f"main.exe --input test.mp4 --upscale 1 --upscale_method {method} --benchmark 1"
        ).read()

        fps = parseFPS(output)
        results[method] = fps
        time.sleep(1)

    return results

def runInterpolateBenchmark():
    results = {}
    for method in interpolateMethods[0]:
        print(f"Running {method} benchmark...")
        output = os.popen(
            f"main.exe --input test.mp4 --interpolate 1 --interpolate_method {method} --benchmark 1"
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

def runDenoiseBenchmark():
    results = {}
    for method in denoiseMethods:
        print(f"Running {method} benchmark...")
        output = os.popen(
            f"main.exe --input test.mp4 --denoise 1 --denoise_method {method} --benchmark 1"
        ).read()

        fps = parseFPS(output)
        results[method] = fps
        time.sleep(1)

    return results

def parseFPS(output):
    match = re.findall(r'fps=\s*([\d.]+)', output)
    match = match[-1]
    if match:
        return float(match)
    else:
        return None

def parseSystemInfo():
    systemInfo = {}
    with open('log.txt', 'r') as file:
        lines = file.readlines()
        start = lines.index("============== System Checker ==============\n") + 1
        end = lines.index("============== Arguments Checker ==============\n")
        for line in lines[start:end]:
            if ': ' in line:
                key, value = line.strip().split(': ')
                systemInfo[key] = value
    return systemInfo
    
if __name__ == "__main__":
    runAllBenchmarks()