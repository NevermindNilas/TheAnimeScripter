import subprocess
import threading
import os

num_threads = int(input("Number of tests to run in parallel: "))

# Check for FFMPEG
dir_path = os.path.dirname(os.path.realpath(__file__))
if not os.path.exists(os.path.join(dir_path, "ffmpeg")):
    bat_path = r"H:\TheAnimeScripter\get_ffmpeg.bat"
    subprocess.call(bat_path)
    

semaphore = threading.Semaphore(num_threads)

def run_command(command):
    semaphore.acquire()
    try:
        command += " --outpoint 1"
        print("Testing command:" + command)
        process = subprocess.Popen(command, shell=True)
        process.wait()
        print('\n')
    except Exception as e:
        print(e)
        counter += 1
        print('\n')
    finally:
        semaphore.release()

commands: list = []
counter = 0

# Segment Test:
commands.append(f"python .\\main.py --input .\\input\\test.mp4 --output output_segment.mp4 --segment 1")

# Depth Test:
commands.append(f"python .\\main.py --input .\\input\\test.mp4 --output output_depth.mp4 --depth 1")

# Dedup tests
for dedup in "light", "medium", "high":
    commands.append(f"python .\\main.py --input .\\input\\test.mp4 --output output_dedup_{dedup}.mp4 --dedup_strenght {dedup}")

# Sharpen tests
for sharpen in "0", "50":
    commands.append(f"python .\\main.py --input .\\input\\test.mp4 --output output_sharpen_{sharpen}.mp4 --sharpen 1 --sharpen_sens {sharpen}")

# Interpolate tests
for interpolate_factor in range(2, 4):
    commands.append(f"python .\\main.py --input .\\input\\test.mp4 --output output_interpolate_{interpolate_factor}.mp4 --interpolate 1 --interpolate_factor {interpolate_factor}")

# Upscale tests
for upscale_method in "shufflecugan", "cugan", "cugan-amd", "swinir", "compact", "ultracompact", "superultracompact":
    if upscale_method == "shufflecugan":
        commands.append(f"python .\\main.py --input .\\input\\test.mp4 --output output_{upscale_method}.mp4 --upscale 1 --upscale_factor 2 --upscale_method {upscale_method}")

    elif upscale_method == "swinir":
        commands.append(f"python .\\main.py --input .\\input\\test.mp4 --output output_{upscale_method}.mp4 --upscale 1 --upscale_factor 2 --upscale_method {upscale_method}")
    
    elif upscale_method == "cugan-amd" or upscale_method == "cugan":
        for scale in [2,3,4]:
            commands.append(f"python .\\main.py --input .\\input\\test.mp4 --output output_{upscale_method}_{scale}.mp4 --upscale 1 --upscale_factor {scale} --upscale_method {upscale_method}")

    else:
        commands.append(f"python .\\main.py --input .\\input\\test.mp4 --output output_{upscale_method}.mp4 --upscale 1 --upscale_factor 2 --upscale_method {upscale_method}")

# Combination with upscale and interpolate
for combination in "shufflecugan", "cugan", "cugan-amd", "swinir", "compact", "ultracompact", "superultracompact":
    commands.append(f"python .\\main.py --input .\\input\\test.mp4 --output output_combination_{combination}.mp4 --upscale_factor 2 --upscale_method {combination} --interpolate 1 --interpolate_factor 2")

# Combination with upscale, interpolate and dedup
for combination_with_dedup in "shufflecugan", "cugan", "cugan-amd", "swinir", "compact", "ultracompact", "superultracompact":
    commands.append(f"python .\\main.py --input .\\input\\test.mp4 --output output_combination_with_dedup_{combination_with_dedup}.mp4 --upscale 1 --upscale_factor 2 --upscale_method {combination_with_dedup} --interpolate 1 --interpolate_factor 2 --dedup_strenght high")
    
# Combination with upscale and sharpen
# TO : DO

threads = []
for command in commands:
    thread = threading.Thread(target=run_command, args=(command,))
    thread.start()
    threads.append(thread)

for thread in threads:
    thread.join()

import time
time.sleep(3)

try:
    if os.path.exists(os.path.join(dir_path, "ffmpeg")):
        os.remove(os.path.join(dir_path, "ffmpeg"))
except Exception as e:
    print(e)
    
print(f"Failed: {counter}")