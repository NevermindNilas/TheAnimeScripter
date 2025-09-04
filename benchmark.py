import os
import time
import json
import re
import subprocess
import platform
import inquirer
import sys

# Add current directory to path to import src modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if platform.system() == "Windows":
    mainPath = os.path.join(os.getenv("APPDATA"), "TheAnimeScripter")
else:
    mainPath = os.path.join(
        os.getenv("XDG_CONFIG_HOME", os.path.expanduser("~/.config")),
        "TheAnimeScripter",
    )

if not os.path.exists(mainPath):
    os.makedirs(mainPath)

ffmpegLogPath = os.path.join(mainPath, "ffmpeg.log")
tasLogPath = os.path.join(mainPath, "TAS.log")


def runAllBenchmarks(executor, version, inputVideo=None, systemInfo=None):
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
    """Get executor and version with caching for improved performance"""
    try:
        # Try to use cached version first
        from src.utils.dependencyHandler import DependencyChecker
        
        # Set up constants needed by dependencyHandler
        import src.constants as cs
        cs.MAINPATH = mainPath
        
        dependency_checker = DependencyChecker()
        executor, version = dependency_checker.getExecutorCached()
        return executor, version
        
    except Exception as e:
        # Fallback to original implementation if caching fails
        print(f"Caching failed, using fallback: {e}")
        return getExeOriginal()

def getExeOriginal():
    """Original getExe implementation as fallback"""
    if os.path.exists("main.exe"):
        version = subprocess.check_output(["main.exe", "--version"]).decode().strip()
        return ["main.exe"], version
    else:
        if platform.system() == "Linux":
            if os.path.exists("./main"):
                version = (
                    subprocess.check_output(["./main", "--version"]).decode().strip()
                )
                return ["./main"], version
            else:
                version = (
                    subprocess.check_output(["python3.12", "main.py", "--version"])
                    .decode()
                    .strip()
                )
                return ["python3.12", "main.py"], version
        else:
            version = (
                subprocess.check_output(["python", "main.py", "--version"])
                .decode()
                .strip()
            )
            return ["python", "main.py"], version


def getClip(executor):
    print("Please select 1080p as the desired quality.")
    outputPath = "output/test.mp4"
    cmd = executor + ["--input", CLIPURL, "--output", outputPath]
    subprocess.Popen(cmd, shell=False).wait()
    systemInfo = parseSystemInfo()

    return os.path.abspath(outputPath), systemInfo


def runUpscaleBenchmark(inputVideo, executor):
    global currentTest
    results = {}
    for method in upscaleMethods:
        print(f"[{currentTest}/{TOTALTESTS}] {method} benchmark...")
        cmd = executor + [
            "--input",
            inputVideo,
            "--upscale",
            "--upscale_method",
            method,
            "--static",
            "--benchmark",
            "--outpoint",
            "20" if "-tensorrt" in method else "15",
        ]
        print(f"Running command: {' '.join(cmd)}")  # Debugging line
        subprocess.run(cmd, check=True, cwd=os.path.dirname(os.path.abspath(__file__)))

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

        cmd = executor + [
            "--input",
            inputVideo,
            "--interpolate",
            "--interpolate_method",
            method,
            "--benchmark",
        ]
        print(f"Running command: {' '.join(cmd)}")  # Debugging line
        subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))

        fps = parseFPS()
        results[method] = fps
        time.sleep(TIMESLEEP)

        print(
            f"[{currentTest}/{TOTALTESTS}] Running {method} with ensemble benchmark..."
        )
        currentTest += 1

        cmd = executor + [
            "--input",
            inputVideo,
            "--interpolate",
            "--interpolate_method",
            method,
            "--benchmark",
            # "--ensemble",
            "--outpoint",
            "20",
        ]

        if method not in ["rife4.22", "rife4.22-lite"]:
            cmd += ["--ensemble"]
        print(f"Running command: {' '.join(cmd)}")  # Debugging line
        subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))

        fps = parseFPS()
        results[f"{method}-ensemble"] = fps
        time.sleep(TIMESLEEP)

    return results


def parseFPS():
    with open(tasLogPath, "r") as file:
        for line in file:
            if "Total Execution Time" in line:
                match = re.search(
                    r"Total Execution Time: ([\d.]+) seconds - FPS: ([\d.]+)", line
                )
                if match:
                    total_execution_time = float(match.group(1))
                    fps = float(match.group(2))
                    print(f"Total Execution Time: {total_execution_time} seconds")
                    print(f"FPS: {fps}")
                    return total_execution_time, fps
    return None, None


def parseSystemInfo():
    systemInfo = {}
    with open(tasLogPath, "r") as file:
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
    TESTINGVERSION = "V4.4"

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
        "rife4.22",
        "rife4.22-lite",
        "rife4.6-ncnn",
        "rife4.22-lite-ncnn",
        "rife4.22-ncnn",
        "rife4.6-tensorrt",
        "rife4.22-tensorrt",
        "rife4.22-lite-tensorrt",
    ]

    currentTest = 0
    executor, version = getExe()
    inputVideo, systemInfo = getClip(executor)

    # Define the questions
    questions = [
        inquirer.List(
            "GPUVENDOR",
            message="Please select your GPU vendor",
            choices=["NVIDIA", "AMD", "Intel"],
        ),
    ]

    # Ask the questions
    answers = inquirer.prompt(questions)
    GPUVENDOR = answers["GPUVENDOR"].upper()

    # Filter methods based on GPU vendor
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
    elif GPUVENDOR in ["AMD", "INTEL"]:
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
    print(f"Using {' '.join(executor)} version {version}")
    print("Current working directory:", os.getcwd())

    runAllBenchmarks(executor, version, inputVideo, systemInfo)
