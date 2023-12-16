![TAS-Banner](https://github.com/NevermindNilas/TheAnimeScripter/assets/128264457/7eebf967-5466-40a6-a6ba-14e8163c78bc)

# TheAnimeScripter

Welcome to TheAnimeScripter, a comprehensive tool designed for both video processing enthusiasts and professionals all within After Effects. Our tool offers a wide array of functionalities, including interpolation, upscaling, deduplication, segmentation, and more.

[Join The Discord Server](https://discord.gg/bFA6xZxM5V)

## 🚀 Key Features

1. **Interpolation**: Boost your video framerate through frame interpolation.
2. **Upscaling**: Enhance the resolution of your videos for an improved viewing experience.
3. **Deduplication**: Optimize your video size by removing duplicate frames.
4. **Segmentation**: Separate the background from the foreground for easier and faster rotobrushing.
5. **Integration with After Effects**: Seamlessly use our tool directly inside of After Effects.
6. **Model Chaining in After Effects**: Run multiple models all at once.

## 🛠️ Getting Started

### Prerequisites

#### Automated Installation

If Python 3.10 - 3.11 isn't installed on your system, run this file:

- run ```setup.bat``` ( as admin ) 

if it is installed run:

- ```pip install -r requirements.txt```

- ```python download_models.py```

- ```get_ffmpeg.bat```

#### Manual installation

- Download and install Python 3.11 from: https://www.python.org/downloads/release/python-3110/ whilst making sure to add it to System Path

- Open a terminal inside the folder

- ```pip install -r requirements.txt```

- ```python download_models.py```

- Download FFMPEG from: https://github.com/BtbN/FFmpeg-Builds/releases

- inside the folder src make a folder named ffmpeg and drop ffmpeg.exe into it

#### How to use inside of After Effects

- On the top left corner open File -> Scripts -> Install ScriptUI Panel -> (Look for the script.jsx file found within the /src subdirectory )

- After Instalation you will be prompted with a restart of After Effects, do it.

- Now that you've reopened After Effects, go to Window -> And at the bottom of you should have a script.jsx, click it -> Dock the panel wherever you please.

- In the panel, set output to wherever you please and main.py to the main.py file found in the AnimeScripter directory

## 📚 Usage

RIFE:
```py
- python main.py -video video_path_here -model_type rife -multi 2
```

CUGAN:
```py
- python main.py -video video_path_here -model_type cugan -multi 2 -nt 2 -kind_model conservative
```

ShuffleCugan:
```py
- python main.py -video video_path_here -model_type shufflecugan -multi 2 -nt 2
```

Compact:
```py
- python main.py -video video_path_here -model_type compact -multi 2 -nt 2
```

UltraCompact:
```py
- python main.py -video video_path_here -model_type ultracompact -multi 2 -nt 4
```

SwinIR:
```py
- python main.py -video video_path_here -model_type swinir -kind_model small
```

Dedup:
```py
- python main.py -video video_path_here -model_type dedup -kind_model ffmpeg
```

Segment:
```py
- python main.py -video video_path_here -model_type segment -kind_model isnet-anime
```

## 📝 Available Inputs and Models:

```
-video :str      - Takes full path of input file.

-output :str     - Can be left empty for auto generation or takes full path in including file name.

-model_type :str - Can be: Rife, Cugan, ShuffleCugan, Compact, SwinIR, Dedup, Segment, UltraCompact.

-half :bool      - Set to True by default, utilizes FP16, more performance for free generally.

-multi :int      - Used by both Upscaling and Interpolation, 
                   Cugan can utilize scale from 2-4,
                   Shufflecugan only 2, 
                   Compact 2,
                   SwinIR 2 or 4, 
                   Rife.. virtually anything.

-kind_model :str - Cugan: no-denoise, conservative, denoise1x, denoise2x, denoise3x
                   SwinIR: small, medium, large.
                   Dedup: ffmpeg, Hash(N/A), VMAF(N/A), SSIM(N/A)
                   Segment: isnet-anime, isnet-general-use, u2net, silueta, u2netp

-pro :bool       - Set to False by default, Only for CUGAN, utilize pro models.

-nt :int         - Number of threads to utilize for Upscaling and Segmentation,
                   Really CPU/GPU dependent, with my 3090 I max out at 4 for Cugan / Shufflecugan.
                   As for SwinIR I max out at 2

-chain :str      - Not usable yet, for chaining multiple models in one run.
```

## 🙏 Acknowledgements

- [SUDO](https://github.com/styler00dollar/VSGAN-tensorrt-docker) - For helping me debug my code and providing his models.
- [HZWER](https://github.com/hzwer/Practical-RIFE) - For Rife.
- [AILAB](https://github.com/bilibili/ailab/tree/main/Real-CUGAN) - For Cugan.
- [JingyunLiang](https://github.com/JingyunLiang/SwinIR) - For SwinIR.
- [Bubble](https://github.com/Bubblemint864/AI-Models) - For the compact and soon swinir model.
- [Xintao](https://github.com/xinntao/Real-ESRGAN) - for Realesrgan, specifically compact arch.
- [AlixzFX](https://github.com/AlixzFX) - for debugging the scriptUI and inspiration

## 📈 Benchmarks

- N/A

## 📋 To-Do

In no particular order:
- Provide a bundled version with all of the dependencies included ( need assitance with this )
- Add Rife NCNN Vulkan ( probably through ncnn api or directml )
- Fix CPU inference. ( high priority )
- Add testing. ( high priority )
- Add Rife Multithreadding ( halfway done )
- ADd DepthMap process ( high priority )
- Fix slow startup times, probably through lazy loading. ( improved by about 50%, still room for improvement )
- Add Pytorch segmentation. ( current models are all onnx, I will look into a pytorch conversion using onnx2torch )
- Maybe add TRT. ( probably not )

## ✅ Done

- [Massive] Added Rife interpolation
- [Massive] Added Cugan Upscaling.
- [Massive] Added Cugan Multithreading.
- Added Frame deduplication.
- [Massive] Added Shuffle Cugan - 50% faster for the same quality.
- Removed unnecessary implementations like frame-by-frame output.
- Fixed rife output issues.
- Increased performance for Rife ever so slightly.
- Placed Output in the same folder as input.
- Fixed requirements.txt issue where it wouldn't download torch compiled with CUDA.
- kind-model now defaults to shufflecugan.
- Fixed issue when input path has spaces.
- Added scripts to make it easier to get things going.
- [Massive] Added SwinIR.
- [Massive] Added Compact
- [Massive] Added FFMPEG Encoding using moviepy.
- [Massive] Added Ultracompact.
- [Massive] Improved the speed of Inference by upwards of 50% from personal testing!
- Fixed issue where the last nt frames wouldn't be processed.
- [Massive] Added Segmentation.
- Added download_models.py for offline processing.
- [Massive] Added alpha testing of the After Effects script.
- Fixed Path issues associated with spaces inside the script.
- Introduced lazy loading for faster start-up times, more work needed.
- [Massive] Fixed the encoder, now it should be much higher resolution for less space.
- The output will now default to .m4v to comply with After Effects' codec support.
- [Massive] After Effects will now only upscale the trimmed part, not the whole clip.
- [Massive] Increased Rife performance by 50% and bumped up the version to 4.13.
- [Massive] Added Model chaining in the script, now multiple models can be ran at once.
