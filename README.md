# TheAnimeScripter

**TheAnimeScripter** is a Python script designed for all the editors out there. It incorporates various functionalities for video processing.

[Join_The_Discord_Server](https://discord.gg/bFA6xZxM5V)

## Prerequisites

Written in Python 3.11

- pip install -r requirements.txt

- pip install -U vidgear[core]

## Usage

CUGAN: 
- python main.py -video video_name_here -model_type cugan -half -kind_model shufflecugan -nt 2

RIFE:
- python main.py -video video_name_here -model_type rife -half

DeDup:
- N/A

## Special Thanks To

- [SUDO](https://github.com/styler00dollar/VSGAN-tensorrt-docker)
- [HZWER](https://github.com/hzwer/Practical-RIFE)
- [AILAB](https://github.com/bilibili/ailab/tree/main/Real-CUGAN)

## Benchmarks

- N/A

## To-Do

In no particular order:

- Figure out why the output duration of rife is higher than the input - No clue why this happens
- Add Rife Multithreadding - Too hard to do, not gonna happen anytime soon
- Place output in the same folder as input
- Add Rife model download back and make it model-agnostic
- Create a scriptUI for After Effects

## Done

- Added Rife implementation.
- Added Cugan Upscaling
- Added Cugan Multithreading
- Added Frame deduplication
- Added Shuffle Cugan - 50% faster for the same quality
- Removed unnecessary implementations like frame-by-frame output
