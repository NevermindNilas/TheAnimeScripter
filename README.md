![TasImage](https://github.com/NevermindNilas/TheAnimeScripter/assets/128264457/9b6dd3b9-3af6-4749-b317-49cb94e85539)

# TheAnimeScripter

Welcome to TheAnimeScripter, a comprehensive tool designed for both video processing enthusiasts and professionals all within After Effects. Our tool offers a wide array of functionalities, including interpolation, upscaling, deduplication, segmentation, and more.

[Join The Discord Server](https://discord.gg/bFA6xZxM5V)

## 🚀 Key Features

1. **Interpolation**: Boost your video framerate through frame interpolation.
2. **Upscaling**: Enhance the resolution of your videos for an improved viewing experience.
3. **Deduplication**: Optimize your video size by removing duplicate frames.
4. **Segmentation**: Separate the background from the foreground for easier and faster rotobrushing.
5. **Integration with After Effects**: Seamlessly use our tool directly inside of After Effects.
6. **Model Chaining in After Effects**: Run Deduplication, Upscaling, Interpolation all in one.
7. **No Frame Extraction**: The script does 0 frame extraction, everything is being done in memory without additional decode and encode cycles.

## 🛠️ Getting Started

### Prerequisites

#### How to utilize

- Download one of the freshest releases from: 
- [Here](https://github.com/NevermindNilas/TheAnimeScripter/releases)


#### Or Manual installation and Build

- Download and install Python 3.11 from: https://www.python.org/downloads/release/python-3110/ whilst making sure to add it to System Path

- Open a terminal inside the folder

- ```python setup.py```

- open a terminal, activate the VENV

- ```pip install auto-py-to-exe``` (that's what I've used for building, simplifies everything)

- in the terminal type ```auto-py-to-exe```

- Select: One Directory, input folder is going to be main.py and additional files the folder src and get_ffmpeg.bat

#### How to use inside of After Effects

- On the top left corner open File -> Scripts -> Install ScriptUI Panel -> (Look for the TheAnimeScripter.jsx file found in folder )

- After Instalation you will be prompted with a restart of After Effects, do it.

- Now that you've reopened After Effects, go to Window -> And at the bottom of you should have a TheAnimeScripter.jsx, click it -> Dock the panel wherever you please.

- In the settings panel, set folder to the same directory as The Anime Scripter and output to wherever you please

## 📚 Usage

- N/A

## 📝 Available Inputs and Models:

- N/A

## 🙏 Acknowledgements

- [SUDO](https://github.com/styler00dollar/VSGAN-tensorrt-docker) - For helping me debug my code and shufflecugan model
- [HZWER](https://github.com/hzwer/Practical-RIFE) - For Rife.
- [AILAB](https://github.com/bilibili/ailab/tree/main/Real-CUGAN) - For Cugan.
- [JingyunLiang](https://github.com/JingyunLiang/SwinIR) - For SwinIR.
- [Bubble](https://github.com/Bubblemint864/AI-Models) - For the SwinIR Model
- [Xintao](https://github.com/xinntao/Real-ESRGAN) - for Realesrgan, specifically compact arch.
- [the-database](https://github.com/the-database/mpv-upscale-2x_animejanai) - For Compact, UltraCompact, SuperUltraCompact models
- [Tohrusky](https://github.com/Tohrusky/realcugan-ncnn-py) - For RealCugan-NCNN-Vulkan wrapper.

## 📈 Benchmarks

- N/A

## 📋 To-Do

In no particular order:
- Add testing.
- Fix Rife padding, again.
- Play around with better mpdecimate params
- Find a way to add awarpsharp2 to the pipeline
- Add bounding box support for Segmentation

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
- [Massive] New and Improved UI within After Effects.
- About 20x faster FFMPEG download.
- [Massive] Better start-up times and overall processing times.
- [Massive] Upscaling and Interpolating now requires only 1 decode and encode cycle meaning faster processing times.
- Fixed CPU Inference
- The models can now be downloaded on the go.
- Added logging for easier debugging
- [Massive] Official 0.1 Release (19/12/2023).
- Fixed Compact, Ultracompact issues
- [Massive] Added SuperUltraCompact
- [Massive] Added CUGAN NCNN Vulkan support, now AMD/Intel GPU/iGPU users will be able to take advantage of their systems.
- [Massive] Added --inpoint and --outpoint variables for AE users, now you will ever need at most 2 decode encode processes instead of 3.
- [Massive] Release 0.1.1 (21/12/2023)
- [Massive] Release 0.1.2 (22/12/2023)
- [Massive] Lowered total processing steps from 2 to 1 meaning faster processing and less encoding needed
- Some minor performance improvements, not very noticeable for high end computers.
- [Massive] Increased performance by ~25% by piping output directly to FFMPEG.
- [Massive] Added Anime Segmentation for Auto Rotobrushing. Still some work left to do