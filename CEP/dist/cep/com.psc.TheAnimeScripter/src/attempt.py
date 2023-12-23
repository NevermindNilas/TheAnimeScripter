from moviepy.editor import VideoFileClip
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import time
from threading import Lock

video_file = r"H:\TheAnimeScripter\input\test.mp4"

read_buffer = Queue(maxsize=10000)
buffer_lock = Lock()

"""
Attempt to multithread the reading of frames from a video file

2 threads is about 25% faster compared to 1 thread

3 threads are about 50% faster compared to 1 thread

There are some race conditions to avoid, still, but it is a good start
"""
def read_frame(clip, start, end):
    for i in range(start, end):
        frame = clip.get_frame(i / clip.fps)
        with buffer_lock:
            read_buffer.put((i, frame))

def main():
    start_time = time.time()
    clip = VideoFileClip(video_file)
    nFrames = clip.reader.nframes
    nThreads = 3

    with ThreadPoolExecutor(max_workers=nThreads) as executor:
        frame_ranges = [(i * nFrames // nThreads, (i + 1) * nFrames // nThreads) for i in range(nThreads)]
        executor.map(lambda args: read_frame(clip, *args), frame_ranges)

    while not read_buffer.empty():
        index, frame = read_buffer.get()
        
        print(f"Frame index: {index}, Frame shape: {frame.shape}")
        
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time} seconds")
    print("Number of frames: ", nFrames)
    
    clip.close()

if __name__ == "__main__":
    main()