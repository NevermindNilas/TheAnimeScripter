# TheAnimeScripter

**TheAnimeScripter** is a Python script designed for all the editors out there. It incorporates various functionalities for video processing.

[Join_The_Discord_Server](https://discord.gg/bFA6xZxM5V)

## Prerequisites

If Python 3.11 isn't installed in your system, run the file and make sure that Python is added to your system path:

- setup.bat

Otherwise, run:

- update.bat

# Manual installation:

- Download Python 3.11 from: https://www.python.org/downloads/release/python-3110/

- Open a terminal inside the folder

- pip install -r requirements.txt

## Usage

CUGAN: 
- python main.py -video video_name_here -model_type cugan -nt 2

RIFE:
- python main.py -video video_name_here -model_type rife

Dedup:
- python main.py -video video_name_here -model_type dedup -kind_model ffmpeg

## Special Thanks To

- [SUDO](https://github.com/styler00dollar/VSGAN-tensorrt-docker) - for helping me debug my code
- [HZWER](https://github.com/hzwer/Practical-RIFE) - for Rife
- [AILAB](https://github.com/bilibili/ailab/tree/main/Real-CUGAN) - for Cugan

## Benchmarks

- N/A

## To-Do

In no particular order:

- Make a venv automatically.
- Add Rife NCNN Vulkan
- Add SwinIR
- Add RealESRGAN
- Add more ways of dedupping
- Add Dup detect for upscaling
- Add Rife model download back and make it model-agnostic
- Create a scriptUI for After Effects
- Maybe add TRT

## Done

- Added Rife implementation.
- Added Cugan Upscaling
- Added Cugan Multithreading
- Added Frame deduplication
- Added Shuffle Cugan - 50% faster for the same quality
- Removed unnecessary implementations like frame-by-frame output
- Fixed rife output issues
- Increased performance for Rife ever so slightly
- Placed Output in the same folder as input
- Fixed requirements.txt issue where it wouldn't download torch compiled with CUDA
- kind-model now defaults to shufflecugan
- Fixed issue when input path has spaces
- Added scripts to make it easier to get things going
