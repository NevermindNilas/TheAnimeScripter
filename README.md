# TheAnimeScripter

**TheAnimeScripter** is a Python script designed for all the editors out there. It incorporates various functionalities for video processing.

## Info / Usage

Written in Python 3.11

- pip install -r requirements.txt

- python main.py -video video_name_here

## Special Thanks To

- [SUDO](https://github.com/styler00dollar/VSGAN-tensorrt-docker)
- [HZWER](https://github.com/hzwer/Practical-RIFE)

## Benchmarks

On the test input from the input folder, I achieve peaks of about 80-90 iterations/second using a 3090 + 13700k. This includes decoding, processing, and encoding with a multi-factor of 2 (60fps -> 120fps).

Approximately 5-10% faster than Practical RIFE based on my testing—this is a rough estimate.

## To-Do

In no particular order:

- Figure out why the output duration is higher than the input
- Place output in the same folder as input
- Add Rife model download back and make it model-agnostic
- Remove unnecessary implementations like frame-by-frame output
- Increase encoding speed by using Multiple Threads instead of one
- Implement better encoding
- Integrate Cugan
- Integrate Dedup
- Create a scriptUI for After Effects

## Done

- Added Rife implementation.
