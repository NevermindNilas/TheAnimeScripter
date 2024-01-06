
# TheAnimeScripter

Welcome to TheAnimeScripter, a comprehensive tool designed for both video processing enthusiasts and professionals all within After Effects. Our tool offers a wide array of functionalities, including interpolation, upscaling, deduplication, segmentation, and more.

[Join The Discord Server](https://discord.gg/bFA6xZxM5V)

## 🔥 Deduplicated, Upscaled, Interpolated and Sharpened Demo
https://github.com/NevermindNilas/TheAnimeScripter/assets/128264457/7bca1442-2e49-47fa-99a7-73ba5c7d197b


## 🚀 Key Features

1. **Interpolation**: Boost your video framerate through frame interpolation.
2. **Upscaling**: Enhance the resolution of your videos for an improved viewing experience.
3. **Deduplication**: Optimize your video size by removing duplicate frames.
4. **Segmentation**: Separate the background from the foreground for easier and faster rotobrushing.
5. **Auto Clip Cutting**: Increase your productivity by automatically cutting your clips using a Scene Change Filter.
6. **Integration with After Effects**: Seamlessly use our tool directly inside of After Effects.
7. **Model Chaining in After Effects**: Run Deduplication, Upscaling, Interpolation all in one.
8. **No Frame Extraction**: The script does 0 frame extraction, everything is being done in memory without additional decode and encode cycles.

## 🛠️ Getting Started

#### How to download

- Download one of the latest releases from [here](https://github.com/NevermindNilas/TheAnimeScripter/releases)

#### Or Manually build

- **Notation needed, the code in the repository can include unexpected bugs, if you want stable builds then download from the releases page, but if you want the cutting edge builds, then follow along.**

- Download and install Python 3.11 from: https://www.python.org/downloads/release/python-3110/ whilst making sure to add it to System Path

- Open a terminal inside the folder

- run: ```python build.py```

#### How to use inside of After Effects

- On the top left corner open File -> Scripts -> Install ScriptUI Panel -> (Look for the TheAnimeScripter.jsx file found in folder )

- After Instalation you will be prompted with a restart of After Effects, do it.

- Now that you've reopened After Effects, go to Window -> And at the bottom of you should have a TheAnimeScripter.jsx, click it -> Dock the panel wherever you please.

- In the settings panel, set folder to the same directory as The Anime Scripter and output to wherever you please

## 📚 Available Inputs

- `--input` : (str, required) Absolute path of the input video.
- `--output` : (str, required) Output string of the video, can be absolute path or just a name.
- `--interpolate` : (int, default=0) Set to 1 if you want to enable interpolation, 0 to disable.
- `--interpolate_factor` : (int, default=2) Factor by which to interpolate.
- `--interpolate_method` : (str, default="rife") Method to use for interpolation. Options: "Rife", "Rife-ncnn".
- `--upscale` : (int, default=0) Set to 1 if you want to enable upscaling, 0 to disable.
- `--upscale_factor` : (int, default=2) Factor by which to upscale.
- `--upscale_method` : (str, default="ShuffleCugan") Method to use for upscaling.
- `--cugan_kind` : (str, default="no-denoise") Kind of Cugan to use.
- `--dedup` : (int, default=0) Set to 1 if you want to enable deduplication, 0 to disable.
- `--dedup_method` : (str, default="ffmpeg") Method to use for deduplication.
- `--dedup_strenght` : (str, default="light") Strength of deduplication.
- `--nt` : (int, default=1) Number of threads to use.
- `--half` : (int, default=1) Set to 1 to use half precision, 0 for full precision.
- `--inpoint` : (float, default=0) Inpoint for the video.
- `--outpoint` : (float, default=0) Outpoint for the video.
- `--sharpen` : (int, default=0) Set to 1 if you want to enable sharpening, 0 to disable.
- `--sharpen_sens` : (float, default=50) Sensitivity of sharpening.
- `--segment` : (int, default=0) Set to 1 if you want to enable segmentation, 0 to disable.
- `--scenechange` : (int, default=0) Set to 1 if you want to enable scene change detection, 0 to disable.
- `--scenechange_sens` : (float, default=40) Sensitivity of scene change detection.
- `--depth` : (int, default=0) Generate Depth Maps, 1 to enable, 0 to disable
- `--encode_method` : (str, default="x264") Method to use for encoding. Options: x264, nvenc_h264, nvenc_h265, qsv_h264, qsv_h265 ( only available for processeses that include Upscaling or Interpolation ).

## 🙏 Acknowledgements

- [SUDO](https://github.com/styler00dollar/VSGAN-tensorrt-docker) - For helping me debug my code and shufflecugan model
- [HZWER](https://github.com/hzwer/Practical-RIFE) - For Rife.
- [AILAB](https://github.com/bilibili/ailab/tree/main/Real-CUGAN) - For Cugan.
- [JingyunLiang](https://github.com/JingyunLiang/SwinIR) - For SwinIR.
- [Bubble](https://github.com/Bubblemint864/AI-Models) - For the SwinIR Model
- [Xintao](https://github.com/xinntao/Real-ESRGAN) - for Realesrgan, specifically compact arch.
- [the-database](https://github.com/the-database/mpv-upscale-2x_animejanai) - For Compact, UltraCompact, SuperUltraCompact models
- [Tohrusky](https://github.com/Tohrusky/realcugan-ncnn-py) - For RealCugan-NCNN-Vulkan wrapper.
- [SkyTNT](https://github.com/SkyTNT/anime-segmentation) - For Anime Segmentation.
- [ISL-ORG](https://github.com/isl-org/MiDaS) - For Depth Map Processing.

## 📈 Benchmarks

The following benchmarks were conducted on a system with a 13700k and 3090 GPU for 1920x1080p inputs and take x264 encoding into account:

With FP16 on for every test except NCNN.
- **Interpolation**: 
    - Rife ( v4.13 ): ~91 FPS ( Fastmode True, Essemble False )
    - Rife NCNN: N/A

- **Upscaling 2x**: 
    - Shufflecugan: ~20 FPS
    - Compact: ~19 FPS
    - UltraCompact: ~23 FPS
    - SuperUltraCompact: ~27 FPS
    - SwinIR: ~1.5 FPS
    - Cugan: N/A
    - Cugan-NCNN: ~7 FPS

- **Depth Map Generation**:
    - DPT-Hybrid: ~17 FPS
    - DPT-Large: N/A

- **Segmentation**:
    - Isnet-Anime: ~10 FPS ( with BF16 / AMP Autocast )

Please note that these benchmarks are approximate and actual performance may vary based on specific video content, settings, and other factors.

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
- [Massive] Release 0.1.3 (25/12/2023)
- [Massive] Added Automated Scene Change Detect and cut inside of After Effects
- Added dedup strenght chooser for more accurate dead frame removal.
- [Massive] Release 0.1.4 (28/12/2023)
- Increased segmentation performance by 5X
- [Massive] Added Depth Processing
- [Massive] Huge increase in performance, 2X in Upscale, 1.5x in Interpolation
- [Massive] Release 0.1.5 (06/01/2024)
- Fixed relative timeline issues for inPoint / outPoint inside of AE
