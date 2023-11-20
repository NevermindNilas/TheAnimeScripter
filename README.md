# TheAnimeScripter

**TheAnimeScripter** is a Python script designed for all the editors out there. It incorporates various functionalities for video processing.

[Join_The_Discord_Server](https://discord.gg/bFA6xZxM5V)

## Info / Usage

Written in Python 3.11

- pip install -r requirements.txt

- python main.py -video video_name_here

## Special Thanks To

- [SUDO](https://github.com/styler00dollar/VSGAN-tensorrt-docker)
- [HZWER](https://github.com/hzwer/Practical-RIFE)
- [AILAB](https://github.com/bilibili/ailab/tree/main/Real-CUGAN)

## Benchmarks

On the test input from the input folder, I achieve peaks of about 80-90 iterations/second using a 3090 + 13700k. This includes decoding, processing, and encoding with a multi-factor of 2 (60fps -> 120fps).

Approximately 5-10% faster than Practical RIFE based on my testing—this is a rough estimate.

## To-Do

In no particular order:

- Figure out why the output duration of rife is higher than the input
- Add Rife Multithreadding
- Place output in the same folder as input
- Ditch opencv for https://github.com/abhiTronix/vidgear
- Add Rife model download back and make it model-agnostic
- Integrate Dedup
- Create a scriptUI for After Effects

## Done

- Added Rife implementation.
- Added Cugan Upscaling
- Added Cugan Multithreading
- Added Shuffle Cugan - 50% faster for the same quality
- Removed unnecessary implementations like frame-by-frame output
