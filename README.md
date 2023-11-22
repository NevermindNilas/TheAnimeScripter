# TheAnimeScripter

**TheAnimeScripter** is a Python script designed for all the editors out there. It incorporates various functionalities for video processing. This will eventually be a Script for After Effects.

[Join The Discord Server](https://discord.gg/bFA6xZxM5V)

# Prerequisites

### Automated Installation

If Python 3.11 isn't installed in your system, run the file and make sure that Python is added to your system path:

```setup.bat```

Otherwise, run:

```update.bat```

### Manual installation:

- Download and install Python 3.11 from: https://www.python.org/downloads/release/python-3110/ whilst making sure to add it to System Path

- Open a terminal inside the folder

- pip install -r requirements.txt

# Usage

RIFE:
```py
- python main.py -video video_name_here -model_type rife
```

CUGAN:
```py
- python main.py -video video_name_here -model_type cugan -nt 2 -kind_model conservative
```
ShuffleCugan:
```py
-python main.py -video video_name_here -model_type shufflecugan -nt 2
```

SwinIR:
```py
- python main.py -video video_name_here -model_type swinir -kind_model small
```

Dedup:
```py
- python main.py -video video_name_here -model_type dedup -kind_model ffmpeg
```

Segment:
```py
- To be Added
```

## Available Inputs and Models:

```
-video :str      - Takes full path of input file.

-model_type :str - Can be Rife, Cugan, ShuffleCugan, SwinIR, Dedup, Segment (N/A).

-half :bool      - Set to True by default, utilizes FP16, more performance for free generally.

-multi :int      - Used by both Upscaling and Interpolation, 
                   Cugan can utilize scale from 2-4,
                   Shufflecugan only 2, 
                   SwinIR 2 or 4, 
                   Rife.. virtually anything.

-kind_model :str - Cugan: no-denoise, conservative, denoise1x, denoise2x, denoise3x
                   SwinIR: small, medium, large.
                   Dedup: ffmpeg, Hash(N/A), VMAF(N/A), SSIM(N/A)

-pro :bool       - Only for CUGAN, utilize pro models.

-nt :int         - Number of threads to utilize for Upscaling and Segmentation,
                   Really CPU/GPU dependent, with my 3090 I max out at 4 for Cugan / Shufflecugan.
                   As for SwinIR I max out at 2
```

# Acknowledgements

- [SUDO](https://github.com/styler00dollar/VSGAN-tensorrt-docker) - For helping me debug my code
- [HZWER](https://github.com/hzwer/Practical-RIFE) - For Rife
- [AILAB](https://github.com/bilibili/ailab/tree/main/Real-CUGAN) - For Cugan
- [JingyunLiang](https://github.com/JingyunLiang/SwinIR) - For SwinIR

# Benchmarks

- N/A

# To-Do

In no particular order:

- Provide a bundled version with all of the dependencies
- Add Rife NCNN Vulkan
- Add RealESRGAN
- Add Segmentation
- Add Custom Models, specifically custom compact
- Add more ways of dedupping
- Create a scriptUI for After Effects
- Maybe add TRT
- Maybe add HAT

# Done

- Added Rife implementation
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
- Add SwinIR
