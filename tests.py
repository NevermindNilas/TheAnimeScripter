import subprocess
import threading

semaphore = threading.Semaphore(2)

def run_command(command):
    semaphore.acquire()
    try:
        command += " --outpoint 1"
        print(command)
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
for interpolate_factor in range(2, 4):
    commands.append(f"python .\\main.py --input .\\input\\test.mp4 --output output_interpolate_{interpolate_factor}.mp4 --interpolate 1 --interpolate_factor {interpolate_factor}")

for upscale_method in "shufflecugan", "cugan", "cugan-amd", "swinir", "compact", "ultracompact", "superultracompact":
    if upscale_method == "shufflecugan":
        commands.append(f"python .\\main.py --input .\\input\\test.mp4 --output output_{upscale_method}.mp4 --upscale_factor 2 --upscale_method {upscale_method}")

    elif upscale_method == "swinir":
        commands.append(f"python .\\main.py --input .\\input\\test.mp4 --output output_{upscale_method}.mp4 --upscale_factor 2 --upscale_method {upscale_method}")
    
    elif upscale_method == "cugan-amd" or upscale_method == "cugan":
        for scale in [2,3,4]:
            commands.append(f"python .\\main.py --input .\\input\\test.mp4 --output output_{upscale_method}_{scale}.mp4 --upscale_factor {scale} --upscale_method {upscale_method}")

    else:
        commands.append(f"python .\\main.py --input .\\input\\test.mp4 --output output_{upscale_method}.mp4 --upscale_factor 2 --upscale_method {upscale_method}")

for dedup in "light", "medium", "high":
    commands.append(f"python .\\main.py --input .\\input\\test.mp4 --output output_dedup_{dedup}.mp4 --dedup_strenght {dedup}")

# Combination with upscale and interpolate
for combination in "shufflecugan", "cugan", "cugan-amd", "swinir", "compact", "ultracompact", "superultracompact":
    commands.append(f"python .\\main.py --input .\\input\\test.mp4 --output output_combination_{combination}.mp4 --upscale_factor 2 --upscale_method {combination} --interpolate 1 --interpolate_factor 2")

# Combination with upscale, interpolate and dedup
for combination_with_dedup in "shufflecugan", "cugan", "cugan-amd", "swinir", "compact", "ultracompact", "superultracompact":
    commands.append(f"python .\\main.py --input .\\input\\test.mp4 --output output_combination_with_dedup_{combination_with_dedup}.mp4 --upscale_factor 2 --upscale_method {combination_with_dedup} --interpolate 1 --interpolate_factor 2 --dedup_strenght high")
    
# Combination with upscale and sharpen
# TO : DO

threads = []
for command in commands:
    thread = threading.Thread(target=run_command, args=(command,))
    thread.start()
    threads.append(thread)

for thread in threads:
    thread.join()

print(f"Failed: {counter}")