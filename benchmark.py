import os
import time
import json
import re
import subprocess
import platform

# appdata / Romaing / TheAnimeSCripter / ffmpeg_log.txt
ffmpegLogPath = os.path.join(os.getenv("APPDATA"), "TheAnimeScripter", "ffmpegLog.txt")

def runAllBenchmarks(executor, version, inputVideo=None):
    print(
        "Running all benchmarks. Depending on your system, this may take a while. Please be patient and keep the terminal in focus at all time."
    )
    print(
        "The results will be saved in benchmarkResults.json. Feel free to share this file with the Discord Community at https://discord.gg/WJfyDAVYGZ"
    )
    results = {
        "Upscale": runUpscaleBenchmark(inputVideo, executor),
        "Interpolate": runInterpolateBenchmark(inputVideo, executor),
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
        if platform.system() == "Linux":
            version = (
                subprocess.check_output(["python3.11", "main.py", "--version"])
                .decode()
                .strip()
            )
            return "python3.11 main.py", version
        else:
            version = (
                subprocess.check_output(["python", "main.py", "--version"])
                .decode()
                .strip()
            )
            return "python main.py", version


def getClip(executor):
    print("Please select 1080p as the desired quality.")
    outputPath = "output/test.mp4"
    # Utilize subprocess Popen with the stdout directed to the terminal
    subprocess.Popen(
        f"{executor} --input {CLIPURL} --output {outputPath}", shell=True
    ).wait()
    # os.popen(f"{executor} --input {CLIPURL} --output {outputPath}").read()
    return os.path.abspath(outputPath)


def runUpscaleBenchmark(inputVideo, executor):
    global currentTest
    results = {}
    for method in upscaleMethods:
        print(f"[{currentTest}/{TOTALTESTS}] {method} benchmark...")
        if "-tensorrt" in method:
            subprocess.run(
                f"{executor} --input {inputVideo} --upscale  --upscale_method {method} --benchmark  --outpoint 16"
            )
        else:
            subprocess.run(
                f"{executor} --input {inputVideo} --upscale  --upscale_method {method} --benchmark --outpoint 12"
            )

        fps = parseFPS()
        results[method] = fps
        time.sleep(TIMESLEEP)
        currentTest += 1

    return results


def runInterpolateBenchmark(inputVideo, executor):
    global currentTest
    results = {}
    for method in interpolateMethods:
        print(f"[{currentTest}/{TOTALTESTS}] {method} benchmark...")
        currentTest += 1

        if method != "gmfss":
            subprocess.run(
                f"{executor} --input {inputVideo} --interpolate  --interpolate_method {method} --benchmark --outpoint 75"
            )
        else:
            subprocess.run(
                f"{executor} --input {inputVideo} --interpolate  --interpolate_method {method} --benchmark  --outpoint 4"
            )

        fps = parseFPS()
        results[method] = fps
        time.sleep(TIMESLEEP)

        print(
            f"[{currentTest}/{TOTALTESTS}] Running {method} with ensemble benchmark..."
        )
        currentTest += 1

        if method != "gmfss":  # Ensemble is irrelevant for GMFSS
            subprocess.run(
                f"{executor} --input {inputVideo} --interpolate  --interpolate_method {method} --benchmark  --ensemble  --outpoint 20"
            )
        else:
            subprocess.run(
                f"{executor} --input {inputVideo} --interpolate  --interpolate_method {method} --benchmark  --ensemble  --outpoint 4"  # GMFSS is so slow that even this is too much
            )

        fps = parseFPS()
        results[f"{method}-ensemble"] = fps
        time.sleep(TIMESLEEP)

    return results


def parseFPS():
    with open(ffmpegLogPath, "r") as file:
        output = file.read()
    matches = re.findall(r"fps=\s*([\d.]+)", output)
    # Filter out fps values that are 0.0 or 0
    filtered = [float(fps) for fps in matches if float(fps) > 0]
    if filtered:
        highestFPS = max(filtered)
        averageFPS = round(sum(filtered) / len(filtered), 2)
        print("Highest FPS:", highestFPS, "Average FPS:", averageFPS)
        return ("Highest FPS:", highestFPS, "Average FPS:", averageFPS)
    else:
        print("Couldn't identify FPS value. Skipping...")
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
    TIMESLEEP = 2
    CLIPURL = "https://www.youtube.com/watch?v=kpeUMAVJCig"
    TESTINGVERSION = "V4.2"

    upscaleMethods = [
        "shufflecugan",
        "compact",
        "ultracompact",
        "superultracompact",
        "span",
        "compact-directml",
        "ultracompact-directml",
        "superultracompact-directml",
        "span-directml",
        "span-ncnn",
        "shufflecugan-ncnn",
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
        "rife4.17",
        "rife4.6-ncnn",
        "rife4.15-ncnn",
        "rife4.15-lite-ncnn",
        "rife4.16-lite-ncnn",
        "rife4.17-ncnn",
        "rife4.6-tensorrt",
        "rife4.15-tensorrt",
        "rife4.15-lite-tensorrt",
        "rife4.17-tensorrt",
        "gmfss",
    ]

    currentTest = 0
    executor, version = getExe()
    inputVideo = getClip(executor)

    while True:
        GPUVENDOR = input("Please enter your GPU vendor (NVIDIA, AMD, Intel): ").upper()
        if GPUVENDOR in ["NVIDIA", "AMD", "INTEL"]:
            break
        else:
            print(f"Invalid GPU vendor, >>> {GPUVENDOR} <<<  is not a valid option. Please try again.")

    if GPUVENDOR == "NVIDIA":
        upscaleMethods = [
            method
            for method in upscaleMethods
            if "ncnn" not in method and "directml" not in method
        ]
        interpolateMethods = [
            method
            for method in interpolateMethods
            if "ncnn" not in method and "directml" not in method
        ]
    elif GPUVENDOR == "AMD" or GPUVENDOR == "INTEL":
        upscaleMethods = [
            method
            for method in upscaleMethods
            if "ncnn" in method or "directml" in method
        ]
        interpolateMethods = [
            method
            for method in interpolateMethods
            if "ncnn" in method or "directml" in method
        ]      

    if platform.system() == "Linux":
        upscaleMethods = [
            method for method in upscaleMethods if "directml" not in method
        ]

    TOTALTESTS = len(upscaleMethods) + len(interpolateMethods) * 2
    print(f"GPU Vendor: {GPUVENDOR}")
    print(f"Total models to benchmark: {TOTALTESTS}")
    print(f"Using {executor} version {version}")
    runAllBenchmarks(executor, version, inputVideo)
