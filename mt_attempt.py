from multiprocessing import Process, Queue
import os
import logging
import subprocess
from moviepy.editor import VideoFileClip
import cv2
import numpy as np
from tqdm import tqdm

def get_ffmpeg_path():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    ffmpeg_path = os.path.join(dir_path, "ffmpeg", "ffmpeg.exe")
    
    if not os.path.exists(ffmpeg_path):
        print("Couldn't find FFMPEG, downloading it now")
        print("This might add an aditional 1-5 seconds to the startup time of the process until FFMPEG is downloaded and caches are built, but it will only happen once")
        logging.info("The user doesn't have FFMPEG, downloading it now")
        ffmpeg_bat_location = os.path.join(dir_path, "get_ffmpeg.bat")
        subprocess.call(ffmpeg_bat_location, shell=True)
        
    return ffmpeg_path

def get_video_metadata(input):
    clip = VideoFileClip(input)
    width = clip.size[0]
    height = clip.size[1]
    fps = clip.fps
    nframes = clip.reader.nframes

    logging.info(
        f"Video Metadata: {width}x{height} @ {fps}fps, {nframes} frames")

    return width, height, fps, nframes
       
def build_buffer(read_buffer, input, ffmpeg_path, width, height):
    ffmpeg_command = [
        ffmpeg_path,
        "-i", str(input),
    ]

    ffmpeg_command.extend([
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-v", "quiet",
        "-stats",
        "-",
    ])

    process = subprocess.Popen(
        ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    logging.info(f"Running command: {ffmpeg_command}")

    frame_size = width * height * 3
    frame_count = 0

    for chunk in iter(lambda: process.stdout.read(frame_size), b''):
        if len(chunk) != frame_size:
            logging.error(
                f"Read {len(chunk)} bytes but expected {frame_size}")
            break
        frame = np.frombuffer(chunk, dtype=np.uint8).reshape(
            (height, width, 3))
        read_buffer.put(frame)
        frame_count += 1
        
    stderr = process.stderr.read().decode()
    if stderr:
        if "bitrate=" not in stderr:
            logging.error(f"ffmpeg error: {stderr}")
            
    process.stdout.close()
    process.stderr.close()
    process.terminate()

    for _ in range(1):
        read_buffer.put(None)

    logging.info(f"Read {frame_count} frames")


def start_process(read_buffer, write_buffer, width, height):
    try:
        while True:
            frame = read_buffer.get()
            if frame is None:
                read_buffer.put(None)
                break
            
            frame = cv2.resize(frame, (width, height))
            write_buffer.put(frame)
            
    except Exception as e:
        logging.exception("An error occurred during processing")

def clear_write_buffer(write_buffer, height, width, output, fps, ffmpeg_path, nframes):
    pbar = tqdm(total=nframes, desc="Writing frames")
    command = [ffmpeg_path,
           '-y',
           '-f', 'rawvideo',
           '-vcodec', 'rawvideo',
           '-s', f'{width}x{height}',
           '-pix_fmt', 'rgb24',
           '-r', str(fps),
           '-i', '-',
           '-an',
           '-c:v', 'libx264',
           '-preset', 'veryfast',
           '-crf', '15',
           '-tune', 'animation',
           '-movflags', '+faststart',
           output]
    
    pipe = subprocess.Popen(
        command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        while True:
            frame = write_buffer.get()
            if frame is None:
                write_buffer.put(None)
                break
            
            # Write the frame to FFmpeg
            pipe.stdin.write(frame.tobytes())
            pbar.update(1)

    except Exception as e:
        logging.exception("An error occurred during writing")
    pipe.stdin.close()
    pipe.wait()


if __name__ == "__main__":

    input = r"H:\TheAnimeScripter\input\test.mp4"
    output = r"H:\TheAnimeScripter\input\output.mp4"

    log_file_path = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'log.txt')

    logging.basicConfig(filename=log_file_path, filemode='w',
                        format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)

    ffmpeg_path = get_ffmpeg_path()
    width, height, fps, nframes = get_video_metadata(input)
    
    read_buffer = Queue(maxsize=500)
    write_buffer = Queue(maxsize=500)

    procs = []
    proc = Process(target=build_buffer, args=(
        read_buffer, input, ffmpeg_path, width, height))
    procs.append(proc)
    proc.start()

    width *= 2
    height *= 2

    num_processes = 2

    for _ in range(num_processes):
        proc = Process(target=start_process, args=(read_buffer, write_buffer, width, height))
        procs.append(proc)
        proc.start()

    proc = Process(target=clear_write_buffer, args=(write_buffer, height, width, output, fps, ffmpeg_path, nframes))
    procs.append(proc)
    proc.start()

    for proc in procs:
        proc.join()
